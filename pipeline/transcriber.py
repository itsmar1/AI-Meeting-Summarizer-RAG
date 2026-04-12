from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from core.config import settings
from core.exceptions import TranscriptionError
from core.logger import get_logger
from pipeline.language_detector import DetectionResult, detect_language, load_audio


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

    The audio file is decoded once into a numpy array via decode_audio()
    and that array is reused for both language detection and transcription.
    This avoids the "'str' object has no attribute 'dtype'" error that
    occurs when passing a file path string to faster-whisper v1.x APIs
    that expect a pre-decoded array.


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

        # Step 1 — decode audio to numpy array ONCE
        # Both detect_language() and model.transcribe() receive the same
        # array — no double file read, no path-string type mismatch.
        audio: np.ndarray = load_audio(audio_path)
        logger.debug("Audio decoded: %d samples at 16 kHz", len(audio))

        # Step 2 — detect language from the decoded array
        language_result = detect_language(audio, model)

        # Step 3 — transcribe from the same decoded array
        raw_segments, _ = model.transcribe(
            audio,  # ← numpy array, not a path string
            language=language_result.language_code,
            word_timestamps=True,  # required for diarization alignment
            vad_filter=True,  # skip silent parts → faster + cleaner
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