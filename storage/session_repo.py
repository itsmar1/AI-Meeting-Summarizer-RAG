"""
Repository layer for meeting sessions.

All database access goes through this module — nothing else imports
aiosqlite directly.  Functions are thin async wrappers around SQL so
they are easy to test with an in-memory SQLite database.
"""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from core.exceptions import StorageError
from core.logger import get_logger
from pipeline.output_parser import ActionItem, MeetingOutput
from storage.db import get_db, utc_now

logger = get_logger(__name__)


# ── Read models ───────────────────────────────────────────────────────────────

@dataclass
class SessionSummary:
    """Lightweight row used for the history list in the UI."""
    id: str
    created_at: str
    audio_filename: str
    language_name: str
    speaker_count: int
    summary: str            # truncated for list display


@dataclass
class SessionDetail:
    """Full session data used for the detail view and re-download."""
    id: str
    created_at: str
    audio_filename: str
    language_code: str
    language_name: str
    speaker_count: int
    transcript: str
    summary: str
    key_topics: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    open_questions: list[str]
    context: str | None
    chroma_collection: str | None   # None if RAG index not built yet


# ── Write operations ──────────────────────────────────────────────────────────

async def save_session(
    audio_filename: str,
    transcript: str,
    output: MeetingOutput,
    speaker_count: int,
    context: str | None = None,
) -> str:
    """
    Persist a completed meeting session and its action items.

    Args:
        audio_filename: Original filename of the uploaded audio.
        transcript:     Full diarized transcript string.
        output:         MeetingOutput from the pipeline.
        speaker_count:  Number of speakers identified by pyannote.
        context:        Optional user-provided context string.

    Returns:
        The new session UUID string.

    Raises:
        StorageError: On any database write failure.
    """
    session_id = str(uuid.uuid4())
    now = utc_now()

    extraction = output.extraction

    logger.info("Saving session %s (%s)", session_id, audio_filename)

    try:
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO sessions (
                    id, created_at, audio_filename, language_code, language_name,
                    speaker_count, transcript, summary, key_topics, decisions,
                    open_questions, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    now,
                    audio_filename,
                    output.language_code,
                    output.language_name,
                    speaker_count,
                    transcript,
                    output.summary,
                    json.dumps(extraction.key_topics, ensure_ascii=False),
                    json.dumps(extraction.decisions, ensure_ascii=False),
                    json.dumps(extraction.open_questions, ensure_ascii=False),
                    context,
                ),
            )

            # Insert action items into the normalised table
            if extraction.action_items:
                await db.executemany(
                    """
                    INSERT INTO action_items (session_id, task, owner, due_date)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (session_id, item.task, item.owner, item.due_date)
                        for item in extraction.action_items
                    ],
                )

            await db.commit()

    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"Failed to save session: {exc}") from exc

    logger.info("Session saved: %s", session_id)
    return session_id


async def save_chroma_ref(session_id: str, collection_name: str) -> None:
    """
    Link a ChromaDB collection name to a session after the RAG index
    has been built.

    Args:
        session_id:       The session UUID.
        collection_name:  The ChromaDB collection name.
    """
    try:
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO chroma_refs (session_id, collection_name)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET collection_name = excluded.collection_name
                """,
                (session_id, collection_name),
            )
            await db.commit()
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"Failed to save Chroma ref: {exc}") from exc

    logger.debug("Chroma ref saved: session=%s collection=%s", session_id, collection_name)


# ── Read operations ───────────────────────────────────────────────────────────

async def list_sessions(limit: int = 50) -> list[SessionSummary]:
    """
    Return the most recent sessions for the history list.

    Args:
        limit: Maximum number of sessions to return (newest first).

    Returns:
        List of SessionSummary objects ordered by created_at DESC.
    """
    try:
        async with get_db() as db:
            async with db.execute(
                """
                SELECT id, created_at, audio_filename, language_name,
                       speaker_count, summary
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"Failed to list sessions: {exc}") from exc

    return [
        SessionSummary(
            id=row["id"],
            created_at=row["created_at"],
            audio_filename=row["audio_filename"],
            language_name=row["language_name"],
            speaker_count=row["speaker_count"],
            summary=row["summary"][:200] + "…" if len(row["summary"]) > 200 else row["summary"],
        )
        for row in rows
    ]


async def get_session(session_id: str) -> SessionDetail | None:
    """
    Fetch full details for a single session including action items and
    the ChromaDB collection reference.

    Args:
        session_id: UUID of the session.

    Returns:
        SessionDetail or None if the session does not exist.
    """
    try:
        async with get_db() as db:
            # Main session row
            async with db.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                return None

            # Action items
            async with db.execute(
                "SELECT task, owner, due_date FROM action_items WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                action_rows = await cursor.fetchall()

            # Chroma ref
            async with db.execute(
                "SELECT collection_name FROM chroma_refs WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                chroma_row = await cursor.fetchone()

    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"Failed to fetch session {session_id}: {exc}") from exc

    return SessionDetail(
        id=row["id"],
        created_at=row["created_at"],
        audio_filename=row["audio_filename"],
        language_code=row["language_code"],
        language_name=row["language_name"],
        speaker_count=row["speaker_count"],
        transcript=row["transcript"],
        summary=row["summary"],
        key_topics=json.loads(row["key_topics"]),
        decisions=json.loads(row["decisions"]),
        action_items=[
            ActionItem(task=r["task"], owner=r["owner"], due_date=r["due_date"])
            for r in action_rows
        ],
        open_questions=json.loads(row["open_questions"]),
        context=row["context"],
        chroma_collection=chroma_row["collection_name"] if chroma_row else None,
    )


async def delete_session(session_id: str) -> bool:
    """
    Delete a session and all its related rows (CASCADE handles action_items
    and chroma_refs automatically).

    Args:
        session_id: UUID of the session to delete.

    Returns:
        True if a row was deleted, False if the session did not exist.
    """
    try:
        async with get_db() as db:
            cursor = await db.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            await db.commit()
            deleted = cursor.rowcount > 0
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"Failed to delete session {session_id}: {exc}") from exc

    if deleted:
        logger.info("Session deleted: %s", session_id)
    return deleted


async def save_transcript_file(
    session_id: str,
    transcript: str,
    audio_filename: str,
) -> Path:
    """
    Write the diarized transcript to a .txt file in the transcript directory
    and return the path (used for the Gradio download button).

    Args:
        session_id:     Used as part of the filename for uniqueness.
        transcript:     Full diarized transcript text.
        audio_filename: Original audio filename (for the header comment).

    Returns:
        Path to the saved transcript file.
    """
    settings_import()   # lazy import to avoid circular at module load
    from core.config import settings

    dest = settings.transcript_dir / f"{session_id}.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)

    header = f"# Meeting Transcript\n# Source: {audio_filename}\n# Session: {session_id}\n\n"
    dest.write_text(header + transcript, encoding="utf-8")

    logger.debug("Transcript saved to: %s", dest)
    return dest


def settings_import():
    """Dummy function — exists only to allow the lazy import pattern above."""
    pass