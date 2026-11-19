import json
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional
from backend.app.models import Clip, FaceTrack, Word
from backend.app.render.reframe import build_reframe_filter
from backend.app.render.captions import generate_ass, generate_srt
from backend.app.ingest.ffmpeg_utils import run_ffmpeg, validate_source
from backend.app.errors import RenderFailed
from backend.app.logging import logger

_ENCODER_CHECKED = None


def get_video_encoder() -> str:
    global _ENCODER_CHECKED
    if _ENCODER_CHECKED is not None:
        return _ENCODER_CHECKED

    try:
        # Test if h264_nvenc can actually initialize on this machine
        test_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.04",
            "-c:v", "h264_nvenc",
            "-f", "null", "-",
        ]
        proc = run_ffmpeg(test_cmd, timeout=3.0, suppress_error_log=True)
        if proc.returncode == 0:
            _ENCODER_CHECKED = "h264_nvenc"
            return _ENCODER_CHECKED
    except Exception:
        pass

    _ENCODER_CHECKED = "libx264"
    return _ENCODER_CHECKED


def extract_thumbnail(
    source_path: Path,
    peak_t: float,
    out_thumb: Path,
    aspect: str = "9:16",
) -> Path:
    out_thumb.parent.mkdir(parents=True, exist_ok=True)
    # Target dimensions
    scale_str = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" if aspect == "9:16" else "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{peak_t:.3f}",
        "-i", str(source_path),
        "-vf", scale_str,
        "-vframes", "1",
        "-q:v", "2",
        str(out_thumb),
    ]
    run_ffmpeg(cmd, timeout=15.0)
    return out_thumb


def render_clip(
    job_id: str,
    clip_idx: int,
    source_path: Path,
    clip: Clip,
    words: List[Word],
    face_tracks: Optional[List[FaceTrack]] = None,
    aspect: str = "9:16",
    out_dir: Path = Path("data/out"),
    attribution: str = "",
    captions: bool = True,
    mask_profanity_toggle: bool = True,
) -> Clip:
    clip_dir = out_dir / job_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_file = clip_dir / f"clip_{clip_idx:02d}.mp4"
    srt_file = clip_dir / f"clip_{clip_idx:02d}.srt"
    ass_file = clip_dir / f"clip_{clip_idx:02d}.ass"
    thumb_file = clip_dir / f"clip_{clip_idx:02d}_thumb.jpg"
    json_file = clip_dir / f"clip_{clip_idx:02d}.json"

    # 1. Source dimensions
    src_info = validate_source(source_path)
    v_stream = next((s for s in src_info["streams"] if s["codec_type"] == "video"), None)
    source_w = int(v_stream.get("width", 1920)) if v_stream else 1920
    source_h = int(v_stream.get("height", 1080)) if v_stream else 1080

    target_w = 1080 if aspect in ("9:16", "1:1") else 1920
    target_h = 1920 if aspect == "9:16" else 1080

    # 2. Captions & SRT
    generate_srt(
        words=words,
        clip_start=clip.start,
        clip_end=clip.end,
        out_srt_path=srt_file,
        mask_curse=mask_profanity_toggle,
    )

    if captions:
        generate_ass(
            words=words,
            clip_start=clip.start,
            clip_end=clip.end,
            out_ass_path=ass_file,
            mask_curse=mask_profanity_toggle,
            creator_credit=attribution,
            video_width=target_w,
            video_height=target_h,
        )

    # 3. Build reframe filter
    reframe_filter = build_reframe_filter(
        aspect=aspect,
        source_w=source_w,
        source_h=source_h,
        target_w=target_w,
        target_h=target_h,
        face_tracks=face_tracks,
        start_t=clip.start,
        end_t=clip.end,
    )

    has_ass = captions and ass_file.exists()
    ass_target = ass_file.name if has_ass else ""

    is_complex = ";" in reframe_filter
    if is_complex:
        if has_ass:
            filter_complex = f"{reframe_filter}[v_ov];[v_ov]subtitles={ass_target}[v_out]"
        else:
            filter_complex = f"{reframe_filter}[v_out]"
    else:
        if has_ass:
            simple_vf = f"{reframe_filter},subtitles={ass_target}"
        else:
            simple_vf = reframe_filter

    # 4. Loudnorm audio filter
    af_chain = "loudnorm=I=-14:LRA=11:TP=-1.5"

    encoder = get_video_encoder()
    enc_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]
    if encoder == "h264_nvenc":
        enc_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21"]

    clip_duration = clip.end - clip.start

    # Render command with -ss after -i for accurate seeking as per spec
    # Run with cwd=clip_dir so subtitles file is referenced locally without path/colon escaping issues
    source_resolved = str(source_path.resolve())
    clip_filename = clip_file.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i", source_resolved,
        "-ss", f"{clip.start:.3f}",
        "-t", f"{clip_duration:.3f}",
    ]

    if is_complex:
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[v_out]",
            "-map", "0:a?",
            "-af", af_chain,
            *enc_args,
            "-c:a", "aac",
            "-b:a", "192k",
            clip_filename,
        ])
    else:
        cmd.extend([
            "-vf", simple_vf,
            "-af", af_chain,
            *enc_args,
            "-c:a", "aac",
            "-b:a", "192k",
            clip_filename,
        ])

    proc = run_ffmpeg(cmd, timeout=180.0, cwd=clip_dir)
    if proc.returncode != 0 or not clip_file.exists() or clip_file.stat().st_size == 0:
        logger.error("render_failed_stderr", stderr=proc.stderr)
        raise RenderFailed(f"Failed to render clip {clip_idx} to {clip_file}")

    # 5. Extract thumbnail
    extract_thumbnail(source_path, clip.peak_t, thumb_file, aspect=aspect)

    # 6. Save clip metadata JSON
    clip.file = str(clip_file)
    clip.srt = str(srt_file)
    clip.thumb = str(thumb_file)
    clip.attribution = attribution

    meta = clip.model_dump()
    json_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return clip
