import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, Optional
from backend.app.errors import SourceUnavailable
from backend.app.logging import logger

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass


def run_ffmpeg(
    args: list[str],
    timeout: float = 60.0,
    suppress_error_log: bool = False,
    cwd: Optional[Path | str] = None,
) -> subprocess.CompletedProcess:
    logger.debug("executing_ffmpeg", args=args)
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            shell=False,
            cwd=str(cwd) if cwd else None,
        )
        if proc.returncode != 0 and not suppress_error_log:
            logger.error("ffmpeg_failed", stderr=proc.stderr, returncode=proc.returncode)
        return proc
    except subprocess.TimeoutExpired as e:
        logger.error("ffmpeg_timeout", timeout=timeout)
        raise SourceUnavailable(f"FFmpeg operation timed out after {timeout}s") from e
    except FileNotFoundError as e:
        logger.error("ffmpeg_not_found")
        raise SourceUnavailable("FFmpeg/ffprobe executable not found on system") from e


def validate_source(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise SourceUnavailable(f"Source file {file_path} does not exist or is empty")

    # Primary: ffprobe
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]
        proc = run_ffmpeg(cmd, timeout=15.0)
        if proc.returncode == 0:
            try:
                info = json.loads(proc.stdout)
                if info.get("streams"):
                    return info
            except Exception:
                pass

    # Fallback: parse ffmpeg -i output
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg_bin, "-i", str(file_path)]
    proc = run_ffmpeg(cmd, timeout=15.0, suppress_error_log=True)
    stderr = proc.stderr or ""

    streams = []
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
    if not dur_match and "Invalid data found" in stderr:
        raise SourceUnavailable(f"Corrupted media file: {file_path}")

    duration = 0.0
    if dur_match:
        h, m, s = dur_match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)

    for line in stderr.splitlines():
        if "Stream #" in line:
            if "Video:" in line:
                m_res = re.search(r"(\d{3,5})x(\d{3,5})", line)
                streams.append({
                    "codec_type": "video",
                    "height": int(m_res.group(2)) if m_res else 360,
                    "width": int(m_res.group(1)) if m_res else 640,
                    "duration": duration,
                })
            elif "Audio:" in line:
                m_sr = re.search(r"(\d{4,6})\s*Hz", line)
                channels = 1 if "mono" in line else 2
                streams.append({
                    "codec_type": "audio",
                    "sample_rate": m_sr.group(1) if m_sr else "16000",
                    "channels": channels,
                    "duration": duration,
                })

    if not streams:
        raise SourceUnavailable(f"No audio or video streams found in {file_path}")

    return {
        "format": {"duration": duration},
        "streams": streams,
    }


def get_duration(file_path: Path) -> float:
    info = validate_source(file_path)
    fmt = info.get("format", {})
    if "duration" in fmt:
        return float(fmt["duration"])
    for s in info.get("streams", []):
        if "duration" in s:
            return float(s["duration"])
    raise SourceUnavailable("Could not determine media duration")


def extract_audio(source_path: Path, out_wav: Path, duration: float = 0.0) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(source_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_wav),
    ]
    calc_timeout = max(300.0, float(duration) * 0.35)
    proc = run_ffmpeg(cmd, timeout=calc_timeout)
    if proc.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        raise SourceUnavailable("Failed to extract audio track as 16kHz mono WAV")
    return out_wav


def create_proxy_video(source_path: Path, out_proxy: Path, duration: float = 0.0) -> Path:
    out_proxy.parent.mkdir(parents=True, exist_ok=True)
    fps_val = "1" if duration > 1800.0 else "2"
    cmd = [
        "ffmpeg",
        "-y",
        "-threads", "0",
        "-i", str(source_path),
        "-an",
        "-vf", f"scale=-2:360,fps={fps_val}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "30",
        str(out_proxy),
    ]
    calc_timeout = max(300.0, float(duration) * 0.5)
    proc = run_ffmpeg(cmd, timeout=calc_timeout)
    if proc.returncode != 0 or not out_proxy.exists() or out_proxy.stat().st_size == 0:
        raise SourceUnavailable("Failed to generate 360p proxy video")
    return out_proxy


def generate_synthetic_video(out_path: Path, duration: float = 30.0, with_audio: bool = True) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=duration={duration}:size=640x360:rate=25",
    ]
    if with_audio:
        cmd.extend([
            "-f", "lavfi",
            "-i", f"sine=frequency=1000:duration={duration}:sample_rate=16000",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-shortest",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
        ])
    cmd.append(str(out_path))

    proc = run_ffmpeg(cmd, timeout=30.0)
    if proc.returncode != 0 or not out_path.exists():
        raise SourceUnavailable("Failed to synthesize test video fixture")
    return out_path
