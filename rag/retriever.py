"""
Retriever — splits transcripts into chunks, builds the vector index,
and retrieves the most relevant chunks for a query.

Chunking strategy
─────────────────
We split the diarized transcript on speaker-turn boundaries first
(each "SPEAKER_XX: …" block becomes a candidate chunk), then merge
short turns and split long turns to stay within settings.rag_chunk_size.
This preserves speaker context, which matters for questions like
"What did the second speaker say about the budget?".
"""

import re

from core.config import settings
from core.exceptions import RAGError
from core.logger import get_logger
from rag.embedder import embed_batch, embed_text
from rag.vector_store import (
    add_chunks,
    collection_exists,
    get_or_create_collection,
    query_collection,
)
from storage.session_repo import save_chroma_ref

logger = get_logger(__name__)

# Regex that matches speaker-turn lines: "SPEAKER_00: text…"
_SPEAKER_TURN_RE = re.compile(r"^(SPEAKER_\w+):\s*(.+)$", re.MULTILINE)


def _split_into_chunks(transcript: str) -> list[str]:
    """
    Split a diarized transcript into overlapping chunks suitable for
    embedding and retrieval.

    Algorithm
    ─────────
    1. Split on speaker-turn boundaries.
    2. Merge consecutive turns from the same speaker into a single block
       until the block would exceed chunk_size.
    3. Apply character-level overlap between consecutive chunks.

    Args:
        transcript: Full diarized transcript string.

    Returns:
        List of chunk strings.
    """
    chunk_size = settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap

    # Extract all turns as (speaker, text) pairs
    turns = [
        (m.group(1), m.group(2).strip())
        for m in _SPEAKER_TURN_RE.finditer(transcript)
    ]

    if not turns:
        # Fallback for plain (non-diarized) transcripts
        return _character_chunks(transcript, chunk_size, overlap)

    # Build blocks by merging short consecutive turns
    blocks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for speaker, text in turns:
        line = f"{speaker}: {text}"
        if current_len + len(line) > chunk_size and current_lines:
            blocks.append("\n".join(current_lines))
            # Keep the last line for overlap
            current_lines = current_lines[-1:] if overlap > 0 else []
            current_len = sum(len(l) for l in current_lines)
        current_lines.append(line)
        current_len += len(line)

    if current_lines:
        blocks.append("\n".join(current_lines))

    logger.debug("Transcript split into %d chunk(s).", len(blocks))
    return blocks


def _character_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback: plain character-level chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


async def build_index(session_id: str, transcript: str) -> str:
    """
    Chunk a transcript, embed all chunks, and store them in ChromaDB.

    This is called once after a meeting is processed.  If an index
    already exists for the session it is skipped (idempotent).

    Args:
        session_id:  Session UUID — used as the ChromaDB collection name.
        transcript:  Full diarized transcript string.

    Returns:
        The ChromaDB collection name (stored in SQLite via save_chroma_ref).

    Raises:
        RAGError: If embedding or storage fails.
    """
    from rag.vector_store import collection_name_for

    col_name = collection_name_for(session_id)

    if collection_exists(session_id):
        logger.info("RAG index already exists for session %s — skipping.", session_id)
        return col_name

    logger.info("Building RAG index for session %s…", session_id)

    chunks = _split_into_chunks(transcript)
    if not chunks:
        raise RAGError("Transcript produced no chunks — cannot build index.")

    embeddings = await embed_batch(chunks)

    collection = get_or_create_collection(session_id)
    add_chunks(collection, chunks, embeddings, session_id)

    # Persist the collection reference in SQLite
    await save_chroma_ref(session_id, col_name)

    logger.info(
        "RAG index built: %d chunk(s) stored in collection '%s'.",
        len(chunks),
        col_name,
    )
    return col_name


async def retrieve(session_id: str, query: str) -> list[dict]:
    """
    Retrieve the most relevant transcript chunks for a user query.

    Args:
        session_id: Session UUID to search within.
        query:      The user's natural language question.

    Returns:
        List of result dicts (see vector_store.query_collection for schema).

    Raises:
        RAGError: If the index doesn't exist or the query fails.
    """
    if not collection_exists(session_id):
        raise RAGError(
            f"No RAG index found for session {session_id}. "
            "The index may still be building — please wait a moment."
        )

    logger.debug("Retrieving chunks for query: %s…", query[:60])

    query_embedding = await embed_text(query)
    collection = get_or_create_collection(session_id)
    results = query_collection(collection, query_embedding)

    logger.debug("Retrieved %d chunk(s).", len(results))
    return results


async def retrieve_across_sessions(
    session_ids: list[str],
    query: str,
) -> list[dict]:
    """
    Retrieve relevant chunks across multiple sessions (for cross-meeting
    questions like "What was discussed about the budget last month?").

    Args:
        session_ids: List of session UUIDs to search.
        query:       The user's natural language question.

    Returns:
        Combined and distance-sorted list of result dicts.
    """
    query_embedding = await embed_text(query)

    all_results: list[dict] = []
    for sid in session_ids:
        if not collection_exists(sid):
            continue
        collection = get_or_create_collection(sid)
        results = query_collection(collection, query_embedding)
        all_results.extend(results)

    # Sort by cosine distance (lower = more similar)
    all_results.sort(key=lambda r: r["distance"])
    return all_results[: settings.rag_top_k]