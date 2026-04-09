import asyncio
import shutil
from pathlib import Path

from core.config import settings
from core.exceptions import AudioProcessingError
from core.logger import get_logger

logger = get_logger(__name__)


def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is found on PATH."""
    return shutil.which("ffmpeg") is not None


async def preprocess_audio(input_path: str | Path) -> Path:
    """
    Convert any audio/video file to a 16 kHz mono WAV that faster-whisper
    can consume directly.

    The output file is written next to the input with a '_converted' suffix
    so the original is never modified.  The caller is responsible for
    deleting the output file when it is no longer needed.

    Args:
        input_path: Path to the source audio/video file.

    Returns:
        Path to the converted WAV file.

    Raises:
        AudioProcessingError: If ffmpeg is missing or the conversion fails.
    """
    if not _ffmpeg_available():
        raise AudioProcessingError(
            "ffmpeg is not installed or not on PATH. "
            "Install it with: sudo apt install ffmpeg"
        )

    input_path = Path(input_path)
    if not input_path.exists():
        raise AudioProcessingError(f"Input file not found: {input_path}")

    output_path = input_path.with_stem(input_path.stem + "_converted").with_suffix(".wav")

    cmd = [
        "ffmpeg",
        "-y",                             # overwrite output without asking
        "-i", str(input_path),            # input file
        "-ar", str(settings.audio_sample_rate),   # sample rate  → 16000 Hz
        "-ac", str(settings.audio_channels),      # channels     → 1 (mono)
        "-sample_fmt", "s16",             # sample format → 16-bit PCM
        str(output_path),
    ]

    logger.info("Converting audio: %s → %s", input_path.name, output_path.name)
    logger.debug("ffmpeg command: %s", " ".join(cmd))

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
    except OSError as exc:
        raise AudioProcessingError(f"Failed to launch ffmpeg: {exc}") from exc

    if process.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace").strip()
        raise AudioProcessingError(
            f"ffmpeg exited with code {process.returncode}.\n{error_msg}"
        )

    logger.info("Audio conversion complete: %s", output_path.name)
    return output_path


async def cleanup_audio(path: Path) -> None:
    """
    Delete a temporary audio file.  Logs a warning if deletion fails but
    does not raise — a leftover temp file is not a fatal error.

    Args:
        path: Path to the file to delete.
    """
    try:
        path.unlink(missing_ok=True)
        logger.debug("Deleted temp audio file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete temp audio file %s: %s", path, exc)