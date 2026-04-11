"""
ChromaDB vector store — one collection per meeting session.

Each session gets its own ChromaDB collection named after its UUID so
collections are isolated and can be deleted individually when a session
is removed from history.

ChromaDB is configured to persist to disk at settings.chroma_path so
embeddings survive application restarts.
"""

from pathlib import Path

import chromadb
from chromadb import Collection

from core.config import settings
from core.exceptions import RAGError
from core.logger import get_logger

logger = get_logger(__name__)

# Module-level client — one persistent client shared across all operations
_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Return (or lazily initialise) the ChromaDB persistent client."""
    global _client
    if _client is None:
        chroma_path = settings.chroma_path
        chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info("Initialising ChromaDB at: %s", chroma_path)
        _client = chromadb.PersistentClient(path=str(chroma_path))
    return _client


def collection_name_for(session_id: str) -> str:
    """
    Derive a ChromaDB collection name from a session UUID.

    ChromaDB collection names must be 3–63 characters and match
    [a-zA-Z0-9_-]+, so we prefix with 'session_' and strip the dashes
    from the UUID.
    """
    return f"session_{session_id.replace('-', '')}"


def get_or_create_collection(session_id: str) -> Collection:
    """
    Return the ChromaDB collection for a session, creating it if needed.

    Args:
        session_id: The session UUID.

    Returns:
        ChromaDB Collection object.

    Raises:
        RAGError: If ChromaDB raises an unexpected error.
    """
    name = collection_name_for(session_id)
    try:
        collection = _get_client().get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
    except Exception as exc:
        raise RAGError(f"Failed to get/create collection '{name}': {exc}") from exc

    logger.debug("Using collection: %s", name)
    return collection


def add_chunks(
    collection: Collection,
    chunks: list[str],
    embeddings: list[list[float]],
    session_id: str,
) -> None:
    """
    Add transcript chunks and their embeddings to a ChromaDB collection.

    Each chunk is stored with:
        - A deterministic ID (session_id + chunk index)
        - The pre-computed embedding vector
        - The raw text as a document (for display in search results)
        - Metadata: session_id and chunk index (for filtering)

    Args:
        collection:  Target ChromaDB collection.
        chunks:      List of transcript text chunks.
        embeddings:  Pre-computed embedding vectors (same length as chunks).
        session_id:  Session UUID used in document IDs and metadata.

    Raises:
        RAGError: If ChromaDB raises during the add operation.
    """
    if len(chunks) != len(embeddings):
        raise RAGError(
            f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch."
        )

    ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"session_id": session_id, "chunk_index": i} for i in range(len(chunks))]

    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise RAGError(f"Failed to add chunks to ChromaDB: {exc}") from exc

    logger.debug("Added %d chunk(s) to collection %s.", len(chunks), collection.name)


def query_collection(
    collection: Collection,
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for a query embedding.

    Args:
        collection:      The ChromaDB collection to query.
        query_embedding: Embedding vector of the user's question.
        top_k:           Number of results to return (defaults to settings.rag_top_k).

    Returns:
        List of dicts with keys: 'text', 'session_id', 'chunk_index', 'distance'.

    Raises:
        RAGError: If ChromaDB raises during the query.
    """
    k = top_k or settings.rag_top_k

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RAGError(f"ChromaDB query failed: {exc}") from exc

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "session_id": meta.get("session_id", ""),
            "chunk_index": meta.get("chunk_index", -1),
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def delete_collection(session_id: str) -> None:
    """
    Delete the ChromaDB collection for a session (called when the session
    is removed from history).

    Args:
        session_id: The session UUID.
    """
    name = collection_name_for(session_id)
    try:
        _get_client().delete_collection(name)
        logger.info("Deleted ChromaDB collection: %s", name)
    except Exception as exc:
        # Log but don't raise — a missing collection is not a fatal error
        logger.warning("Could not delete collection %s: %s", name, exc)


def collection_exists(session_id: str) -> bool:
    """
    Check whether a ChromaDB collection exists for a given session.

    Args:
        session_id: The session UUID.

    Returns:
        True if the collection exists and contains at least one document.
    """
    name = collection_name_for(session_id)
    try:
        col = _get_client().get_collection(name)
        return col.count() > 0
    except Exception:
        return False