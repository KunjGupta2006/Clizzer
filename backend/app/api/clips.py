import json
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from backend.app.config import settings
from backend.app.models import Clip
from backend.app.render.render import render_clip
from backend.app.logging import logger

router = APIRouter(prefix="/clips", tags=["clips"])


@router.post("/{clip_id}/trim")
def trim_clip(clip_id: str, payload: Dict[str, float]) -> Dict[str, Any]:
    # clip_id format: job_id_clip_01 or search in data/out
    parts = clip_id.rsplit("_clip_", 1)
    if len(parts) == 2:
        job_id = parts[0]
        clip_num = int(parts[1])
    else:
        # Try finding json file
        matches = list(settings.OUT_DIR.glob(f"**/{clip_id}.json"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")
        json_file = matches[0]
        job_id = json_file.parent.name
        clip_num = int(json_file.stem.replace("clip_", ""))

    json_path = settings.OUT_DIR / job_id / f"clip_{clip_num:02d}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Clip metadata not found")

    clip_dict = json.loads(json_path.read_text(encoding="utf-8"))
    clip = Clip.model_validate(clip_dict)

    start_delta = float(payload.get("start_delta", 0.0))
    end_delta = float(payload.get("end_delta", 0.0))

    # Clamp deltas to +/- 5s
    start_delta = max(-5.0, min(5.0, start_delta))
    end_delta = max(-5.0, min(5.0, end_delta))

    clip.start = max(0.0, clip.start + start_delta)
    clip.end = max(clip.start + 5.0, clip.end + end_delta)

    source_mp4 = settings.JOBS_DIR / job_id / "source.mp4"
    if not source_mp4.exists():
        raise HTTPException(status_code=404, detail="Original source video missing for re-render")

    rendered = render_clip(
        job_id=job_id,
        clip_idx=clip_num,
        source_path=source_mp4,
        clip=clip,
        words=[],
        out_dir=settings.OUT_DIR,
    )
    return rendered.model_dump()


@router.get("/{clip_id}/download")
def download_clip(clip_id: str):
    # If clip_id is like job_12345_clip_01 or job_12345_clip_01.mp4
    if "_clip_" in clip_id:
        parts = clip_id.rsplit("_clip_", 1)
        job_id = parts[0]
        clip_name = f"clip_{parts[1].replace('.mp4', '')}.mp4"
        target = settings.OUT_DIR / job_id / clip_name
        if target.exists():
            return FileResponse(path=str(target), media_type="video/mp4", filename=target.name)

    matches = list(settings.OUT_DIR.glob(f"**/{clip_id}.mp4"))
    if not matches:
        # Check if clip_id is filename
        matches = list(settings.OUT_DIR.glob(f"**/{clip_id}"))
    if not matches or not matches[0].exists():
        raise HTTPException(status_code=404, detail="Clip video file not found")

    # Prefer the latest rendered clip if multiple match
    file_path = sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=file_path.name,
    )
