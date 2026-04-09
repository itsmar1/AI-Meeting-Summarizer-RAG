from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from core.exceptions import LanguageDetectionError
from core.logger import get_logger

logger = get_logger(__name__)

# ISO 639-1 codes that Whisper supports, mapped to human-readable names
# used for display in the UI and for selecting the right prompt template.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

# Minimum probability below which we consider detection unreliable
_MIN_CONFIDENCE = 0.60


@dataclass
class DetectionResult:
    """Result of a language detection pass."""
    language_code: str          # ISO 639-1, e.g. "fr"
    language_name: str          # human-readable, e.g. "French"
    confidence: float           # 0.0 – 1.0 probability from Whisper


def detect_language(audio_path: str | Path, model: WhisperModel) -> DetectionResult:
    """
    Detect the spoken language in an audio file using faster-whisper's
    built-in language identification.

    faster-whisper analyses the first 30 seconds of audio and returns a
    probability distribution over all languages it knows.  We take the
    highest-probability language and validate it meets a minimum confidence
    threshold.

    Args:
        audio_path: Path to the pre-processed WAV file.
        model:      An already-loaded WhisperModel instance (shared with
                    the transcriber to avoid loading the model twice).

    Returns:
        DetectionResult with the detected language code, name and confidence.

    Raises:
        LanguageDetectionError: If detection confidence is below the threshold
                                or the detected language is unsupported.
    """
    audio_path = Path(audio_path)
    logger.info("Detecting language in: %s", audio_path.name)

    # detect_language() returns (language_code, probabilities_dict)
    detected_code, probabilities = model.detect_language(str(audio_path))

    confidence = probabilities.get(detected_code, 0.0)
    logger.debug(
        "Language detection: code=%s confidence=%.2f", detected_code, confidence
    )

    if confidence < _MIN_CONFIDENCE:
        raise LanguageDetectionError(
            f"Language detection confidence too low ({confidence:.0%}). "
            f"Detected '{detected_code}' but not confident enough to proceed. "
            "Try a larger Whisper model or check the audio quality."
        )

    # Resolve the human-readable name; fall back gracefully for unsupported codes
    language_name = SUPPORTED_LANGUAGES.get(detected_code, detected_code.upper())

    if detected_code not in SUPPORTED_LANGUAGES:
        logger.warning(
            "Language '%s' is not in the curated supported list — "
            "transcription may still work but prompt templates may fall back "
            "to English.",
            detected_code,
        )

    result = DetectionResult(
        language_code=detected_code,
        language_name=language_name,
        confidence=confidence,
    )
    logger.info(
        "Detected language: %s (%s) — confidence %.0f%%",
        language_name,
        detected_code,
        confidence * 100,
    )
    return result