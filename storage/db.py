"""
Database initialisation and connection management.

Schema
──────
sessions          — one row per processed meeting
action_items      — normalised action items linked to a session
chroma_refs       — ChromaDB collection names linked to a session

All timestamps are stored as ISO-8601 strings (UTC) so they are
human-readable in any SQLite browser without extra tooling.
"""

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite

from core.config import settings
from core.exceptions import StorageError
from core.logger import get_logger

logger = get_logger(__name__)

# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,          -- UUID
    created_at      TEXT NOT NULL,             -- ISO-8601 UTC
    audio_filename  TEXT NOT NULL,             -- original uploaded filename
    language_code   TEXT NOT NULL,             -- e.g. "fr"
    language_name   TEXT NOT NULL,             -- e.g. "French"
    speaker_count   INTEGER NOT NULL DEFAULT 0,
    transcript      TEXT NOT NULL,             -- full diarized transcript
    summary         TEXT NOT NULL,             -- narrative summary paragraph
    key_topics      TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
    decisions       TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
    open_questions  TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
    context         TEXT                         -- user-provided context (nullable)
);
"""

_CREATE_ACTION_ITEMS = """
CREATE TABLE IF NOT EXISTS action_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task        TEXT NOT NULL,
    owner       TEXT NOT NULL DEFAULT 'Unassigned',
    due_date    TEXT                    -- nullable
);
"""

_CREATE_CHROMA_REFS = """
CREATE TABLE IF NOT EXISTS chroma_refs (
    session_id       TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    collection_name  TEXT NOT NULL      -- ChromaDB collection name for this session
);
"""

_ENABLE_WAL = "PRAGMA journal_mode=WAL;"
_ENABLE_FK  = "PRAGMA foreign_keys=ON;"

# ── Init ──────────────────────────────────────────────────────────────────────

async def init_db(db_path: Path | None = None) -> None:
    """
    Create the database file and all tables if they do not already exist.
    Safe to call multiple times (idempotent).

    Args:
        db_path: Override for the database path (defaults to settings.db_path).
    """
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising database at: %s", path)

    async with aiosqlite.connect(path) as db:
        await db.execute(_ENABLE_WAL)
        await db.execute(_ENABLE_FK)
        await db.execute(_CREATE_SESSIONS)
        await db.execute(_CREATE_ACTION_ITEMS)
        await db.execute(_CREATE_CHROMA_REFS)
        await db.commit()

    logger.info("Database ready.")


# ── Connection context manager ────────────────────────────────────────────────

@asynccontextmanager
async def get_db(
    db_path: Path | None = None,
) -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Async context manager that yields a connected aiosqlite.Connection
    with WAL mode and foreign keys enabled.

    Usage:
        async with get_db() as db:
            await db.execute("SELECT …")

    Args:
        db_path: Override for the database path.

    Raises:
        StorageError: If the connection cannot be established.
    """
    path = db_path or settings.db_path

    try:
        async with aiosqlite.connect(path) as db:
            db.row_factory = aiosqlite.Row   # rows accessible as dicts
            await db.execute(_ENABLE_FK)
            await db.execute(_ENABLE_WAL)
            yield db
    except sqlite3.Error as exc:
        raise StorageError(f"Database error on {path}: {exc}") from exc


# ── Utility ───────────────────────────────────────────────────────────────────

def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()