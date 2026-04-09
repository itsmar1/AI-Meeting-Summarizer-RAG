"""
Custom exceptions for the Meeting Summariser.

Hierarchy
─────────
MeetingSummarizerError          ← base for everything in this project
├── AudioProcessingError        ← ffmpeg conversion failures
├── TranscriptionError          ← faster-whisper failures
├── DiarizationError            ← pyannote failures
├── LanguageDetectionError      ← language detection failures
├── SummarizationError          ← Ollama LLM call failures
│   └── OllamaConnectionError   ← Ollama server unreachable
├── OutputParsingError          ← Pydantic / JSON parse failures
├── StorageError                ← SQLite read/write failures
└── RAGError                    ← ChromaDB / embedding failures
    └── EmbeddingError          ← Ollama embedding call failures
"""


class MeetingSummarizerError(Exception):
    """Base class for all project-specific exceptions."""


# ── Audio ─────────────────────────────────────────────────────────────────────

class AudioProcessingError(MeetingSummarizerError):
    """Raised when ffmpeg fails to convert or read an audio file."""


# ── Pipeline ──────────────────────────────────────────────────────────────────

class TranscriptionError(MeetingSummarizerError):
    """Raised when faster-whisper fails to transcribe audio."""


class DiarizationError(MeetingSummarizerError):
    """Raised when pyannote fails to diarise the audio."""


class LanguageDetectionError(MeetingSummarizerError):
    """Raised when language detection produces no usable result."""


# ── LLM ───────────────────────────────────────────────────────────────────────

class SummarizationError(MeetingSummarizerError):
    """Raised when the Ollama summarisation call fails."""


class OllamaConnectionError(SummarizationError):
    """Raised when the Ollama server cannot be reached."""


# ── Parsing ───────────────────────────────────────────────────────────────────

class OutputParsingError(MeetingSummarizerError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


# ── Storage ───────────────────────────────────────────────────────────────────

class StorageError(MeetingSummarizerError):
    """Raised when a SQLite read or write operation fails."""


# ── RAG ───────────────────────────────────────────────────────────────────────

class RAGError(MeetingSummarizerError):
    """Raised when a ChromaDB or retrieval operation fails."""


class EmbeddingError(RAGError):
    """Raised when the Ollama embedding call fails."""