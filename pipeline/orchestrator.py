import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from core.exceptions import MeetingSummarizerError
from core.logger import get_logger
from pipeline.audio_preprocessor import cleanup_audio, preprocess_audio
from pipeline.diarizer import DiarizationResult, diarize
from pipeline.output_parser import MeetingExtraction, MeetingOutput
from pipeline.summarizer import extract_structure, generate_summary
from pipeline.transcriber import TranscriptionResult, transcribe

logger = get_logger(__name__)


@dataclass
class PipelineProgress:
    """
    Emitted at each pipeline stage so the UI can show live progress.
    """
    stage: str          # human-readable stage label
    detail: str = ""    # optional extra detail


async def run_pipeline(
    audio_path: str | Path,
    context: str | None = None,
) -> AsyncGenerator[PipelineProgress | MeetingOutput | str, None]:
    """
    Full end-to-end pipeline:
        Audio → Preprocess → Transcribe → Diarize
        → Extract structure → Stream summary

    This is an async generator that yields:
        - PipelineProgress objects at each stage (for the UI progress bar)
        - Individual str tokens while the summary is streaming
        - A final MeetingOutput object when everything is complete

    The caller (Gradio app) iterates over this generator and updates
    the UI incrementally.

    Args:
        audio_path: Path to the uploaded audio/video file.
        context:    Optional user-provided context string.

    Yields:
        PipelineProgress | str | MeetingOutput

    Raises:
        MeetingSummarizerError subclasses on failure (callers should catch
        MeetingSummarizerError to display a user-friendly error message).
    """
    audio_path = Path(audio_path)
    converted_path: Path | None = None

    try:
        # ── Stage 1: Audio preprocessing ─────────────────────────────────────
        yield PipelineProgress(stage="Preprocessing audio…")
        converted_path = await preprocess_audio(audio_path)

        # ── Stage 2: Transcription + language detection ───────────────────────
        yield PipelineProgress(stage="Transcribing audio…", detail="Detecting language")

        # transcribe() is CPU-bound — run in a thread to avoid blocking the loop
        transcription: TranscriptionResult = await asyncio.to_thread(
            transcribe, converted_path
        )
        yield PipelineProgress(
            stage="Transcription complete",
            detail=(
                f"Language: {transcription.language.language_name} "
                f"({transcription.language.confidence:.0%} confidence)"
            ),
        )

        # ── Stage 3: Speaker diarization ──────────────────────────────────────
        yield PipelineProgress(stage="Identifying speakers…")
        diarization: DiarizationResult = await asyncio.to_thread(
            diarize, converted_path, transcription
        )
        yield PipelineProgress(
            stage="Diarization complete",
            detail=f"{diarization.speaker_count} speaker(s) identified",
        )

        # ── Stage 4: Structured extraction (Stage 1 of LLM chain) ────────────
        yield PipelineProgress(stage="Extracting decisions & action items…")
        extraction: MeetingExtraction = await extract_structure(
            transcript=diarization.diarized_transcript,
            language_name=transcription.language.language_name,
            context=context,
        )
        yield PipelineProgress(
            stage="Extraction complete",
            detail=(
                f"{len(extraction.decisions)} decision(s), "
                f"{len(extraction.action_items)} action item(s)"
            ),
        )

        # ── Stage 5: Streaming narrative summary (Stage 2 of LLM chain) ──────
        yield PipelineProgress(stage="Generating summary…")
        summary_tokens: list[str] = []
        async for token in generate_summary(extraction, transcription.language.language_name):
            summary_tokens.append(token)
            yield token    # stream tokens live to the UI

        summary_text = "".join(summary_tokens)

        # ── Final output ──────────────────────────────────────────────────────
        yield MeetingOutput(
            summary=summary_text,
            extraction=extraction,
            language_code=transcription.language.language_code,
            language_name=transcription.language.language_name,
        )

    except MeetingSummarizerError:
        raise
    except Exception as exc:
        raise MeetingSummarizerError(
            f"Unexpected pipeline error: {exc}"
        ) from exc
    finally:
        # Always clean up the converted WAV regardless of success/failure
        if converted_path is not None:
            await cleanup_audio(converted_path)


async def run_pipeline_collect(
    audio_path: str | Path,
    context: str | None = None,
) -> tuple[MeetingOutput, DiarizationResult | None]:
    """
    Convenience wrapper that runs the full pipeline and collects all output,
    returning the final MeetingOutput without streaming.

    Useful for CLI usage, tests, and batch processing.

    Returns:
        Tuple of (MeetingOutput, DiarizationResult).
        DiarizationResult may be None if diarization was skipped.
    """
    output: MeetingOutput | None = None

    async for item in run_pipeline(audio_path, context):
        if isinstance(item, MeetingOutput):
            output = item

    if output is None:
        raise MeetingSummarizerError("Pipeline completed without producing output.")

    return output