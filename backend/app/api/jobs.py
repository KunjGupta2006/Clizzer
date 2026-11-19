import json
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from backend.app.config import settings
from backend.app.db import get_session
from backend.app.models import Job, JobParams, Clip, SeriesBundle
from backend.app.core.orchestrator import submit_job_async
from backend.app.core.cache import pipeline_cache
from backend.app.rank.select import select_top_clips
from backend.app.render.render import render_clip
from backend.app.logging import logger

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_job(
    request: Request,
    session: Session = Depends(get_session),
) -> Dict[str, str]:
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from e

        try:
            params = JobParams(**body)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))
    else:
        # Form / Multipart
        form = await request.form()
        permission_confirmed = str(form.get("permission_confirmed", "false")).lower() in ("true", "1")
        if not permission_confirmed:
            raise HTTPException(status_code=400, detail="permission_confirmed must be true")

        source_url = form.get("source_url")
        file = form.get("file")
        saved_file_path = None

        if file and hasattr(file, "filename") and file.filename:
            job_temp_dir = settings.DATA_DIR / "uploads"
            job_temp_dir.mkdir(parents=True, exist_ok=True)
            saved_file_path = str(job_temp_dir / f"{uuid.uuid4()}_{file.filename}")
            content = await file.read()
            Path(saved_file_path).write_bytes(content)

        kw_raw = form.get("keywords")
        kw_list = [k.strip() for k in kw_raw.split(",")] if kw_raw else []

        try:
            params = JobParams(
                source_url=str(source_url) if source_url else None,
                file_path=saved_file_path,
                num_clips=int(form.get("num_clips", 5)),
                min_len=int(form.get("min_len", 20)),
                max_len=int(form.get("max_len", 60)),
                aspect=str(form.get("aspect", "9:16")),
                mode=str(form.get("mode", "auto")),
                keywords=kw_list,
                captions=str(form.get("captions", "true")).lower() in ("true", "1"),
                permission_confirmed=permission_confirmed,
            )
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    new_job = Job(
        id=job_id,
        status="queued",
        progress=0.0,
        stage="queued",
    )
    new_job.params = params.model_dump()
    session.add(new_job)
    session.commit()

    # Submit to worker pool
    submit_job_async(job_id, params)
    return {"job_id": job_id}


@router.get("/{job_id}")
def get_job(job_id: str, session: Session = Depends(get_session)) -> Dict[str, Any]:
    job = session.exec(select(Job).where(Job.id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "timings": job.timings,
        "error": job.error,
        "created_at": job.created_at,
        "params": job.params,
    }


@router.get("/{job_id}/clips")
def get_job_clips(job_id: str, session: Session = Depends(get_session)) -> List[Dict[str, Any]]:
    job = session.exec(select(Job).where(Job.id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    out_dir = settings.OUT_DIR / job_id
    if not out_dir.exists():
        return []

    clips = []
    for json_file in sorted(out_dir.glob("clip_*.json")):
        try:
            clips.append(json.loads(json_file.read_text(encoding="utf-8")))
        except Exception:
            pass
    return clips


@router.get("/{job_id}/clips/{clip_name}/download")
def download_job_clip(job_id: str, clip_name: str):
    filename = clip_name if clip_name.endswith(".mp4") else f"{clip_name}.mp4"
    target = settings.OUT_DIR / job_id / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Clip {filename} not found for job {job_id}")
    return FileResponse(path=str(target), media_type="video/mp4", filename=target.name)


@router.get("/{job_id}/curve")
def get_job_curve(job_id: str, session: Session = Depends(get_session)) -> Dict[str, Any]:
    job = session.exec(select(Job).where(Job.id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    params = job.params
    cache_key = f"{params.get('source_url') or params.get('file_path')}_{params.get('mode', 'auto')}_{params.get('aspect', '9:16')}"
    cached = pipeline_cache.load_stage(cache_key, "bundle")
    if not cached:
        raise HTTPException(status_code=404, detail="Curve data not yet available for this job")

    return {
        "duration": cached.get("duration", 0.0),
        "length": cached.get("length", 0),
        "fused": cached.get("fused", []),
        "audio": cached.get("audio", []),
        "chat": cached.get("chat", []),
        "text": cached.get("text", []),
        "visual": cached.get("visual", []),
        "face": cached.get("face", []),
    }


@router.post("/{job_id}/rerank")
def rerank_job(
    job_id: str,
    payload: Dict[str, Any],
    session: Session = Depends(get_session),
) -> List[Dict[str, Any]]:
    job = session.exec(select(Job).where(Job.id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    params = job.params
    cache_key = f"{params.get('source_url') or params.get('file_path')}_{params.get('mode', 'auto')}_{params.get('aspect', '9:16')}"
    bundle_data = pipeline_cache.load_stage(cache_key, "bundle")
    if not bundle_data:
        raise HTTPException(status_code=400, detail="Cannot rerank: SeriesBundle not cached")

    bundle = SeriesBundle.model_validate(bundle_data)
    num_clips = payload.get("num_clips", params.get("num_clips", 5))
    mode = payload.get("mode", params.get("mode", "auto"))
    weights = payload.get("weights")

    ranked_clips, _ = select_top_clips(
        bundle=bundle,
        words=[],
        num_clips=num_clips,
        min_len=float(params.get("min_len", 20)),
        max_len=float(params.get("max_len", 60)),
        mode=mode,
        custom_weights=weights,
        quality_floor=-5.0,
    )

    # Re-render updated clips
    source_mp4 = settings.JOBS_DIR / job_id / "source.mp4"
    rendered_clips = []
    for idx, c in enumerate(ranked_clips):
        rc = render_clip(
            job_id=job_id,
            clip_idx=idx + 1,
            source_path=source_mp4,
            clip=c,
            words=[],
            aspect=params.get("aspect", "9:16"),
            out_dir=settings.OUT_DIR,
            captions=params.get("captions", True),
        )
        rendered_clips.append(rc.model_dump())

    return rendered_clips
