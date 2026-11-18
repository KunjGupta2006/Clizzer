"""Utility functions and media processing helpers for Clizzer.

Provides filesystem workspace management, FFmpeg/FFprobe subprocess wrappers
for duration and resolution extraction, audio track isolation, timestamp formatting,
and system dependency verification.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import uuid
from pathlib import Path
from typing import Any

try:
    from backend.config import config
except ImportError:
    from config import config  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def get_temp_dir() -> Path:
    """Create a unique temporary directory under config.TEMP_DIR using UUID4.

    Returns:
        Path: Path to the newly created unique temporary directory.
    """
    temp_dir = config.TEMP_DIR / f"clip_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp(path: Path | str) -> None:
    """Safely remove a temporary directory and its contents using shutil.rmtree.

    Handles read-only file permissions (e.g. on Windows) and suppresses
    unhandled deletion errors with a warning log.

    Args:
        path: Path to the directory or file to remove.
    """
    target = Path(path)
    if not target.exists():
        return

    def _on_rm_error(func: Any, file_path: str, exc_info: Any) -> None:
        """Clear read-only file attribute and retry deletion."""
        try:
            os.chmod(file_path, stat.S_IWRITE)
            func(file_path)
        except Exception:
            pass

    try:
        if target.is_dir():
            shutil.rmtree(target, onerror=_on_rm_error)
        else:
            target.unlink(missing_ok=True)
    except Exception as err:
        logger.warning("Failed to clean up temporary path %s: %s", target, err)


def get_video_duration(path: str | Path) -> float:
    """Extract media duration in seconds using ffprobe.

    Executes: ffprobe -v error -show_entries format=duration ...

    Args:
        path: Filesystem path to the video or audio file.

    Returns:
        float: Total duration in seconds.

    Raises:
        FileNotFoundError: If the media file does not exist or ffprobe is not installed.
        RuntimeError: If ffprobe command fails or duration cannot be parsed.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Media file not found: {file_path}")

    if not shutil.which("ffprobe"):
        raise FileNotFoundError("ffprobe executable not found on system PATH.")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw_duration = result.stdout.strip()

        # Fallback to stream duration if format duration is absent/not available
        if not raw_duration or raw_duration.upper() == "N/A":
            stream_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ]
            stream_res = subprocess.run(stream_cmd, capture_output=True, text=True, check=True)
            raw_duration = stream_res.stdout.strip()

        return float(raw_duration)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"ffprobe failed for {file_path}: {err.stderr.strip()}") from err
    except (ValueError, TypeError) as err:
        raise RuntimeError(
            f"Failed to parse duration from ffprobe output: {raw_duration!r}"
        ) from err


def get_video_resolution(path: str | Path) -> tuple[int, int]:
    """Retrieve video dimensions (width, height) using ffprobe.

    Args:
        path: Filesystem path to the video file.

    Returns:
        tuple[int, int]: (width, height) resolution in pixels.

    Raises:
        FileNotFoundError: If the video file does not exist or ffprobe is not installed.
        RuntimeError: If resolution extraction or parsing fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    if not shutil.which("ffprobe"):
        raise FileNotFoundError("ffprobe executable not found on system PATH.")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(file_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw_output = result.stdout.strip()
        cleaned = raw_output.replace("x", ",").replace(":", ",")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))
        raise ValueError(f"Unexpected resolution output format: {raw_output!r}")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"ffprobe failed for {file_path}: {err.stderr.strip()}") from err
    except Exception as err:
        raise RuntimeError(f"Failed to parse video resolution: {err}") from err


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 22050,
) -> str:
    """Extract mono audio track from a video file to WAV format using ffmpeg.

    Executes: ffmpeg -i <video> -ar <sr> -ac 1 -f wav <output>

    Args:
        video_path: Filesystem path to the source video file.
        output_path: Target filesystem path for the extracted WAV audio file.
        sample_rate: Target audio sampling rate in Hz (default: 22050).

    Returns:
        str: Filesystem path string to the generated WAV audio file.

    Raises:
        FileNotFoundError: If the source video file is missing or ffmpeg is not found.
        RuntimeError: If ffmpeg audio extraction command fails.
    """
    src_path = Path(video_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source video file not found: {src_path}")

    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg executable not found on system PATH.")

    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_path),
        "-vn",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        str(dst_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return str(dst_path)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"ffmpeg audio extraction failed: {err.stderr.strip()}") from err


def format_timestamp(seconds: float) -> str:
    """Format time in seconds into an 'HH:MM:SS.mmm' timestamp string.

    Args:
        seconds: Time offset or duration in seconds.

    Returns:
        str: Formatted timestamp string (HH:MM:SS.mmm).
    """
    prefix = "-" if seconds < 0 else ""
    total_ms = int(round(abs(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{prefix}{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def ensure_ffmpeg() -> bool:
    """Check whether ffmpeg executable is installed and available on system PATH.

    Returns:
        bool: True if ffmpeg is found on PATH, False otherwise.
    """
    return shutil.which("ffmpeg") is not None
