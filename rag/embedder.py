"""
Embedding generation via Ollama's /api/embeddings endpoint.

We use Ollama's local embedding model (default: nomic-embed-text) so
no external API key or network call is needed.  The embedder is kept
stateless — it just takes text and returns a vector — so it is easy
to swap for a different provider later.
"""

import httpx

from core.config import settings
from core.exceptions import EmbeddingError, OllamaConnectionError
from core.logger import get_logger

logger = get_logger(__name__)

_EMBED_URL = f"{settings.ollama_url}/api/embeddings"
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)


async def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a single text string.

    Args:
        text: The text to embed.  Should be under ~2000 tokens for
              nomic-embed-text.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        EmbeddingError:        If Ollama returns an error response.
        OllamaConnectionError: If the Ollama server is unreachable.
    """
    if not text.strip():
        raise EmbeddingError("Cannot embed empty text.")

    payload = {
        "model": settings.ollama_embed_model,
        "prompt": text,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(_EMBED_URL, json=payload)
    except httpx.ConnectError as exc:
        raise OllamaConnectionError(
            f"Cannot reach Ollama at {settings.ollama_url}. "
            "Make sure Ollama is running: ollama serve"
        ) from exc
    except Exception as exc:
        raise EmbeddingError(f"Embedding request failed: {exc}") from exc

    if response.status_code != 200:
        raise EmbeddingError(
            f"Ollama embedding returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    data = response.json()
    embedding = data.get("embedding")

    if not embedding or not isinstance(embedding, list):
        raise EmbeddingError(
            f"Unexpected embedding response shape: {str(data)[:200]}"
        )

    return embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts sequentially.

    Ollama's embedding endpoint processes one text at a time, so we
    loop rather than batching.  For large corpora this could be
    parallelised with asyncio.gather but sequential is safe enough
    for meeting-length transcripts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors in the same order as the input.
    """
    logger.debug("Embedding %d chunk(s)…", len(texts))
    embeddings = []
    for i, text in enumerate(texts):
        vec = await embed_text(text)
        embeddings.append(vec)
        if (i + 1) % 10 == 0:
            logger.debug("Embedded %d / %d chunks", i + 1, len(texts))
    logger.debug("Batch embedding complete.")
    return embeddings