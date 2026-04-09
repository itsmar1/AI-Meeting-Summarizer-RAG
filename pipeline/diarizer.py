from dataclasses import dataclass, field
from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.core import Annotation

from core.config import settings
from core.exceptions import DiarizationError
from core.logger import get_logger
from pipeline.transcriber import TranscriptionResult, WordTimestamp

logger = get_logger(__name__)

# Module-level cache — pyannote pipeline is expensive to load
_pipeline_cache: Pipeline | None = None


def _load_pipeline() -> Pipeline:
    """Load (or return cached) pyannote diarization pipeline."""
    global _pipeline_cache
    if _pipeline_cache is None:
        if not settings.hf_token:
            raise DiarizationError(
                "HF_TOKEN is not set. A Hugging Face token is required to "
                "download the pyannote diarization model.\n"
                "1. Get a free token at https://huggingface.co/settings/tokens\n"
                "2. Accept the model licence at "
                "   https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Add HF_TOKEN=<your_token> to your .env file."
            )
        logger.info("Loading pyannote speaker diarization pipeline…")
        _pipeline_cache = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )
        logger.info("Pyannote pipeline loaded.")
    return _pipeline_cache


@dataclass
class DiarizedWord:
    """A single word annotated with its speaker label."""
    word: str
    start: float
    end: float
    speaker: str            # e.g. "SPEAKER_00", "SPEAKER_01"


@dataclass
class DiarizedSegment:
    """
    A contiguous block of speech by a single speaker, built by merging
    consecutive DiarizedWords from the same speaker.
    """
    speaker: str
    text: str
    start: float
    end: float


@dataclass
class DiarizationResult:
    """Full diarization output."""
    segments: list[DiarizedSegment]
    speaker_count: int
    diarized_transcript: str    # formatted "SPEAKER_XX: text\n…" string


def _assign_speaker_to_word(
    word: WordTimestamp,
    annotation: Annotation,
) -> str:
    """
    Find the speaker whose turn overlaps the most with a word's time span.

    We use the midpoint of the word as the primary probe because very short
    words can straddle a speaker boundary.  If no turn covers the midpoint
    we fall back to a small window search.

    Args:
        word:       Word with start/end timestamps from Whisper.
        annotation: pyannote Annotation (diarization result).

    Returns:
        Speaker label string, e.g. "SPEAKER_00".  Returns "UNKNOWN" if no
        match is found.
    """
    mid = (word.start + word.end) / 2

    # Primary probe: which turn contains the word's midpoint?
    turns = annotation.get_labels(
        annotation.support().crop(
            {"start": mid - 0.01, "end": mid + 0.01}
        )
    )
    if turns:
        return next(iter(turns))

    # Fallback: largest overlap in the word's full span
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        overlap = min(turn.end, word.end) - max(turn.start, word.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


def _merge_into_segments(diarized_words: list[DiarizedWord]) -> list[DiarizedSegment]:
    """
    Merge consecutive words from the same speaker into segments.

    A new segment is started whenever the speaker changes.

    Args:
        diarized_words: List of words with speaker labels.

    Returns:
        List of DiarizedSegment objects.
    """
    if not diarized_words:
        return []

    segments: list[DiarizedSegment] = []
    current_speaker = diarized_words[0].speaker
    current_words: list[DiarizedWord] = []

    for dw in diarized_words:
        if dw.speaker != current_speaker and current_words:
            segments.append(
                DiarizedSegment(
                    speaker=current_speaker,
                    text=" ".join(w.word.strip() for w in current_words),
                    start=current_words[0].start,
                    end=current_words[-1].end,
                )
            )
            current_words = []
            current_speaker = dw.speaker
        current_words.append(dw)

    # flush last group
    if current_words:
        segments.append(
            DiarizedSegment(
                speaker=current_speaker,
                text=" ".join(w.word.strip() for w in current_words),
                start=current_words[0].start,
                end=current_words[-1].end,
            )
        )

    return segments


def diarize(
    audio_path: str | Path,
    transcription: TranscriptionResult,
) -> DiarizationResult:
    """
    Run speaker diarization and align the result with Whisper's word
    timestamps to produce a speaker-labelled transcript.

    Strategy
    ────────
    1. Run pyannote on the audio to get speaker turn boundaries.
    2. For each word from Whisper, find which speaker turn it falls in.
    3. Merge consecutive same-speaker words into readable segments.

    Args:
        audio_path:     Path to the 16 kHz mono WAV (same file used for
                        transcription).
        transcription:  TranscriptionResult from transcriber.transcribe().

    Returns:
        DiarizationResult with speaker-labelled segments and a formatted
        transcript string.

    Raises:
        DiarizationError: If pyannote fails or produces no output.
    """
    audio_path = Path(audio_path)
    logger.info("Starting diarization: %s", audio_path.name)

    try:
        pipeline = _load_pipeline()
        annotation: Annotation = pipeline(str(audio_path))
    except DiarizationError:
        raise
    except Exception as exc:
        raise DiarizationError(f"pyannote diarization failed: {exc}") from exc

    speakers = annotation.labels()
    logger.info("Diarization found %d speaker(s).", len(speakers))

    # Flatten all words from all Whisper segments into one list
    all_words: list[WordTimestamp] = [
        word
        for segment in transcription.segments
        for word in segment.words
    ]

    if not all_words:
        raise DiarizationError(
            "Transcription produced no word-level timestamps. "
            "Make sure word_timestamps=True is set in the transcriber."
        )

    # Assign a speaker to every word
    diarized_words: list[DiarizedWord] = [
        DiarizedWord(
            word=w.word,
            start=w.start,
            end=w.end,
            speaker=_assign_speaker_to_word(w, annotation),
        )
        for w in all_words
    ]

    segments = _merge_into_segments(diarized_words)

    # Build the human-readable transcript string
    lines = [f"{seg.speaker}: {seg.text}" for seg in segments]
    diarized_transcript = "\n".join(lines)

    logger.info(
        "Diarization complete: %d segments across %d speaker(s).",
        len(segments),
        len(speakers),
    )

    return DiarizationResult(
        segments=segments,
        speaker_count=len(speakers),
        diarized_transcript=diarized_transcript,
    )