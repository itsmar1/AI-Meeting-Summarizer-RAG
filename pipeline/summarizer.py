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
    }

    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream("POST", _OLLAMA_GENERATE_URL, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise SummarizationError(
                        f"Ollama returned HTTP {resp.status_code}: {body.decode()[:200]}"
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
        raise SummarizationError(f"Unexpected error during Ollama call: {exc}") from exc

    return full_response


async def stream_summary_tokens(prompt: str) -> AsyncGenerator[str, None]:
    """
    Yield individual tokens from Ollama as they arrive.
    Used by the UI to stream the narrative summary into the Gradio textbox.

    Args:
        prompt: The Stage 2 summary prompt.

    Yields:
        Token strings as they stream from Ollama.

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
            async with client.stream("POST", _OLLAMA_GENERATE_URL, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise SummarizationError(
                        f"Ollama returned HTTP {resp.status_code}: {body.decode()[:200]}"
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


async def extract_structure(
    transcript: str,
    language_name: str,
    context: str | None = None,
) -> MeetingExtraction:
    """
    Stage 1 — Extract structured data (decisions, action items, etc.)
    from the transcript using a JSON-constrained prompt.

    Args:
        transcript:     Diarized transcript text.
        language_name:  Detected language, e.g. "French".
        context:        Optional user-provided context.

    Returns:
        Validated MeetingExtraction instance.
    """
    logger.info("Stage 1: extracting structured data (language=%s)", language_name)
    prompt = build_extraction_prompt(transcript, language_name, context)
    raw = await _stream_ollama(prompt)
    return parse_extraction(raw)


async def generate_summary(
    extraction: MeetingExtraction,
    language_name: str,
) -> AsyncGenerator[str, None]:
    """
    Stage 2 — Stream a narrative executive summary based on the structured
    extraction from Stage 1.

    Args:
        extraction:    MeetingExtraction from Stage 1.
        language_name: Detected language for the output.

    Yields:
        Summary tokens as they stream from Ollama.
    """
    logger.info("Stage 2: generating narrative summary (language=%s)", language_name)
    structured_data = extraction.model_dump_json(indent=2)
    prompt = build_summary_prompt(structured_data, language_name)
    async for token in stream_summary_tokens(prompt):
        yield token