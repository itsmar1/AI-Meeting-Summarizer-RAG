"""
RAG chain — retrieve relevant transcript chunks, build a grounded
context window, and stream an answer from Ollama.

The chain follows the standard RAG pattern:
    User question
        → embed question
        → retrieve top-k chunks
        → build prompt (question + chunks as context)
        → stream answer from Ollama
        → yield tokens to the UI
"""

import json
from collections.abc import AsyncGenerator

import httpx

from core.config import settings
from core.exceptions import OllamaConnectionError, RAGError, SummarizationError
from core.logger import get_logger
from rag.retriever import retrieve, retrieve_across_sessions

logger = get_logger(__name__)

_OLLAMA_GENERATE_URL = f"{settings.ollama_url}/api/generate"
_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)

# ── Prompt template ───────────────────────────────────────────────────────────

_RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions about meeting transcripts.
Answer the question using ONLY the provided transcript excerpts as context.
If the answer cannot be found in the excerpts, say so clearly — do not make up information.
Be concise and cite which speaker said what when relevant.

Transcript excerpts:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:"""


def _build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Args:
        chunks: List of result dicts from the retriever.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "No relevant transcript excerpts found."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[Excerpt {i}]\n{chunk['text']}")
    return "\n\n".join(lines)


# ── Main chain functions ──────────────────────────────────────────────────────

async def answer_question(
    question: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Answer a question about a single meeting session, streaming the
    response token-by-token.

    Args:
        question:   The user's natural language question.
        session_id: UUID of the session to query.

    Yields:
        Answer tokens as they stream from Ollama.

    Raises:
        RAGError:              If retrieval fails.
        OllamaConnectionError: If the Ollama server is unreachable.
        SummarizationError:    If the Ollama call fails.
    """
    logger.info("RAG question for session %s: %s", session_id, question[:80])

    chunks = await retrieve(session_id, question)
    context = _build_context(chunks)

    prompt = _RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    async for token in _stream_answer(prompt):
        yield token


async def answer_across_sessions(
    question: str,
    session_ids: list[str],
) -> AsyncGenerator[str, None]:
    """
    Answer a question by searching across multiple meeting sessions.
    Useful for cross-meeting queries like "What was said about the budget
    across all meetings this month?".

    Args:
        question:    The user's natural language question.
        session_ids: List of session UUIDs to search.

    Yields:
        Answer tokens as they stream from Ollama.
    """
    logger.info(
        "Cross-session RAG question (%d sessions): %s",
        len(session_ids),
        question[:80],
    )

    chunks = await retrieve_across_sessions(session_ids, question)
    context = _build_context(chunks)

    prompt = _RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    async for token in _stream_answer(prompt):
        yield token


async def _stream_answer(prompt: str) -> AsyncGenerator[str, None]:
    """
    Internal helper — send a prompt to Ollama and yield response tokens.

    Args:
        prompt: Fully built prompt string.

    Yields:
        Token strings from the Ollama streaming response.

    Raises:
        OllamaConnectionError / SummarizationError on failure.
    """
    payload = {
        "model": settings.ollama_llm_model,
        "prompt": prompt,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream(
                "POST", _OLLAMA_GENERATE_URL, json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise SummarizationError(
                        f"Ollama returned HTTP {resp.status_code}: "
                        f"{body.decode()[:200]}"
                    )
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    except httpx.ConnectError as exc:
        raise OllamaConnectionError(
            f"Cannot reach Ollama at {settings.ollama_url}. "
            "Make sure Ollama is running: ollama serve"
        ) from exc
    except (OllamaConnectionError, SummarizationError, RAGError):
        raise
    except Exception as exc:
        raise SummarizationError(
            f"Unexpected error during RAG answer generation: {exc}"
        ) from exc