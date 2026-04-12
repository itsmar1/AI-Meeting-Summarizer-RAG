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
            token=settings.hf_token,
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


def _build_turn_index(annotation: Annotation) -> list[tuple[float, float, str]]:
    """
    Pre-build a flat sorted list of (start, end, speaker) turns from the
    pyannote annotation.

    Building this once and reusing it for every word lookup reduces the
    complexity from O(words × turns) with expensive pyannote API calls
    to O(turns) build + O(turns) per word worst case — fast enough for
    any meeting-length audio.

    Args:
        annotation: pyannote Annotation from the diarization pipeline.

    Returns:
        List of (start, end, speaker) tuples sorted by start time.
    """
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t[0])
    return turns




def _assign_speaker_to_word(
    word: WordTimestamp,
    turns: list[tuple[float, float, str]],
) -> str:
    """
    Find the speaker whose turn has the most overlap with a word's
    time span, using a pre-sorted turn index.

    Uses the word midpoint as the primary probe (fast path), then
    falls back to maximum-overlap search for words that straddle
    a speaker boundary.

    Args:
        word:  Word with start/end timestamps from Whisper.
        turns: Pre-sorted list of (start, end, speaker) from _build_turn_index().


    Returns:
        Speaker label string e.g. "SPEAKER_00", or "UNKNOWN" if no turn
        overlaps the word at all.
    """
    mid = (word.start + word.end) / 2

    # Fast path: find first turn containing the midpoint
    for start, end, speaker in turns:
        if start > mid:
            break  # turns are sorted — no point continuing
        if start <= mid <= end:
            return speaker

    # Fallback: maximum overlap for words at speaker boundaries
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for start, end, speaker in turns:
        if start > word.end:
            break  # past the word entirely
        overlap = min(end, word.end) - max(start, word.start)
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

    # Build the turn index ONCE — O(turns) — then reuse for every word.
    # This replaces the old approach that called annotation.support().crop()
    # once per word, which was O(words × turns) and caused multi-hour hangs.
    turn_index = _build_turn_index(annotation)
    logger.info(
        "Turn index built: %d speaker turns for %d words.",
        len(turn_index),
        len(all_words),
    )

    # Assign a speaker to every word using the fast index lookup
    diarized_words: list[DiarizedWord] = [
        DiarizedWord(
            word=w.word,
            start=w.start,
            end=w.end,
            speaker=_assign_speaker_to_word(w, turn_index),
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