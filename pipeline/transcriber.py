from dataclasses import dataclass, field
from pathlib import Path

from faster_whisper import WhisperModel

from core.config import settings
from core.exceptions import TranscriptionError
from core.logger import get_logger
from pipeline.language_detector import DetectionResult, detect_language

logger = get_logger(__name__)


@dataclass
class WordTimestamp:
    """A single word with its start/end time in seconds."""
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    """
    One continuous speech segment produced by Whisper.
    Each segment contains word-level timestamps needed to align
    speaker labels from the diarizer.
    """
    text: str
    start: float                            # segment start (seconds)
    end: float                              # segment end   (seconds)
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Full output from a transcription run."""
    segments: list[TranscriptSegment]
    language: DetectionResult
    raw_text: str                           # plain concatenation of all segments


# Module-level cache so the model is loaded once per process
_model_cache: dict[str, WhisperModel] = {}


def _load_model() -> WhisperModel:
    """
    Load (or return cached) WhisperModel using settings.

    The model is keyed by size + device + compute_type so different
    configurations in tests don't collide.
    """
    cache_key = (
        f"{settings.whisper_model_size}"
        f"-{settings.whisper_device}"
        f"-{settings.whisper_compute_type}"
    )
    if cache_key not in _model_cache:
        logger.info(
            "Loading Whisper model: size=%s device=%s compute_type=%s",
            settings.whisper_model_size,
            settings.whisper_device,
            settings.whisper_compute_type,
        )
        _model_cache[cache_key] = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper model loaded.")
    return _model_cache[cache_key]


def transcribe(audio_path: str | Path) -> TranscriptionResult:
    """
    Transcribe an audio file to text with word-level timestamps.

    Language is auto-detected from the first 30 seconds of audio and then
    passed back to Whisper so the full transcription uses the correct
    vocabulary.  The word timestamps are preserved so the diarizer can
    later assign speaker labels to each word.

    Args:
        audio_path: Path to the pre-processed 16 kHz mono WAV file.

    Returns:
        TranscriptionResult containing segments, detected language, and
        the full plain-text transcript.

    Raises:
        TranscriptionError: If faster-whisper raises any exception.
    """
    audio_path = Path(audio_path)
    logger.info("Starting transcription: %s", audio_path.name)

    try:
        model = _load_model()

        # Step 1 — detect language (reuses the loaded model, no extra cost)
        language_result = detect_language(audio_path, model)

        # Step 2 — transcribe with word timestamps
        raw_segments, _ = model.transcribe(
            str(audio_path),
            language=language_result.language_code,
            word_timestamps=True,       # required for diarization alignment
            vad_filter=True,            # skip silent parts → faster + cleaner
            vad_parameters={
                "min_silence_duration_ms": 500,
            },
        )

        segments: list[TranscriptSegment] = []
        for seg in raw_segments:
            words = [
                WordTimestamp(word=w.word, start=w.start, end=w.end)
                for w in (seg.words or [])
            ]
            segments.append(
                TranscriptSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    words=words,
                )
            )

    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    raw_text = " ".join(s.text for s in segments)
    logger.info(
        "Transcription complete: %d segments, %d words",
        len(segments),
        sum(len(s.words) for s in segments),
    )

    return TranscriptionResult(
        segments=segments,
        language=language_result,
        raw_text=raw_text,
    )