import json
from collections.abc import AsyncGenerator

import httpx

from core.config import settings
from core.exceptions import OllamaConnectionError, SummarizationError
from core.logger import get_logger
from pipeline.output_parser import MeetingExtraction, parse_extraction
from pipeline.translation_prompt import build_extraction_prompt, build_summary_prompt

logger = get_logger(__name__)

_OLLAMA_GENERATE_URL = f"{settings.ollama_url}/api/generate"
_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)

# llama3.2 has an 8k context window (8192 tokens ≈ ~6000 words ≈ ~32000 chars).
# We reserve ~2000 tokens for the prompt template and the model's response,
# leaving ~24000 characters of transcript that can safely be passed in.
_MAX_TRANSCRIPT_CHARS = 24_000


def _truncate_transcript(transcript: str) -> str:
    """
    Truncate a transcript to fit within the LLM's context window.

    Truncates from the end (keeps the beginning of the meeting) and
    appends a note so the model knows the transcript was cut.
    """
    if len(transcript) <= _MAX_TRANSCRIPT_CHARS:
        return transcript
    truncated = transcript[:_MAX_TRANSCRIPT_CHARS]
    # Cut at the last newline to avoid splitting mid-sentence
    last_newline = truncated.rfind("\n")
    if last_newline > _MAX_TRANSCRIPT_CHARS * 0.8:
        truncated = truncated[:last_newline]
    note = "\n\n[Transcript truncated to fit model context window]"
    logger.warning(
        "Transcript truncated from %d to %d chars to fit context window.",
        len(transcript),
        len(truncated),
    )
    return truncated + note


async def _stream_ollama(prompt: str) -> str:
    """
    Send a prompt to Ollama and collect the full streamed response.

    Args:
        prompt: The complete prompt string.

    Returns:
        The full response text from the model.

    Raises:
        OllamaConnectionError: If the server is unreachable.
        SummarizationError:    If the server returns a non-200 status.
    """
    payload = {
        "model": settings.ollama_llm_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": 8192,    # explicitly set context window for llama3.2
        },
    }

    logger.debug("Sending prompt to Ollama (%d chars)…", len(prompt))

    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream("POST", _OLLAMA_GENERATE_URL, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise SummarizationError(
                        f"Ollama returned HTTP {resp.status_code}: {body.decode()[:300]}"
                    )
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        full_response += chunk.get("response", "")
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.debug("Non-JSON line from Ollama: %s", line[:80])

    except httpx.ConnectError as exc:
        raise OllamaConnectionError(
            f"Cannot reach Ollama at {settings.ollama_url}. "
            "Make sure Ollama is running: ollama serve"
        ) from exc
    except (OllamaConnectionError, SummarizationError):
        raise
    except Exception as exc:
        # Log the full exception type and repr so empty-message errors are visible
        logger.error(
            "Ollama call failed: type=%s repr=%s",
            type(exc).__name__,
            repr(exc),
        )
        raise SummarizationError(
            f"Unexpected error during Ollama call: "
            f"{type(exc).__name__}: {repr(exc)}"
        ) from exc

    if not full_response:
        raise SummarizationError(
            "Ollama returned an empty response. "
            "The prompt may be too long or the model may have run out of context. "
            f"Prompt length: {len(prompt)} chars."
        )

    return full_response


async def stream_summary_tokens(prompt: str) -> AsyncGenerator[str, None]:
    """
    Yield individual tokens from Ollama as they arrive.
    Used by the UI to stream the narrative summary into the Gradio textbox.
    """
    payload = {
        "model": settings.ollama_llm_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": 8192,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream("POST", _OLLAMA_GENERATE_URL, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise SummarizationError(
                        f"Ollama returned HTTP {resp.status_code}: {body.decode()[:300]}"
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
    except (OllamaConnectionError, SummarizationError):
        raise
    except Exception as exc:
        logger.error(
            "Ollama streaming failed: type=%s repr=%s",
            type(exc).__name__,
            repr(exc),
        )
        raise SummarizationError(
            f"Unexpected error during Ollama streaming: "
            f"{type(exc).__name__}: {repr(exc)}"
        ) from exc


async def extract_structure(
    transcript: str,
    language_name: str,
    context: str | None = None,
) -> MeetingExtraction:
    """
    Stage 1 — Extract structured data (decisions, action items, etc.)
    from the transcript using a JSON-constrained prompt.
    """
    logger.info("Stage 1: extracting structured data (language=%s)", language_name)

    # Truncate before building the prompt so we know it fits
    safe_transcript = _truncate_transcript(transcript)
    prompt = build_extraction_prompt(safe_transcript, language_name, context)
    logger.debug("Extraction prompt length: %d chars", len(prompt))

    raw = await _stream_ollama(prompt)
    return parse_extraction(raw)


async def generate_summary(
    extraction: MeetingExtraction,
    language_name: str,
) -> AsyncGenerator[str, None]:
    """
    Stage 2 — Stream a narrative executive summary based on the structured
    extraction from Stage 1.
    """
    logger.info("Stage 2: generating narrative summary (language=%s)", language_name)
    structured_data = extraction.model_dump_json(indent=2)
    prompt = build_summary_prompt(structured_data, language_name)
    logger.debug("Summary prompt length: %d chars", len(prompt))
    async for token in stream_summary_tokens(prompt):
        yield token