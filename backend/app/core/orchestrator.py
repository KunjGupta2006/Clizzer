from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.config import settings
from backend.app.models import (
    JobParams,
    Clip,
    SeriesBundle,
    Word,
    Segment,
    ChatMsg,
    FaceTrack,
)
from backend.app.core.progress import progress_tracker
from backend.app.core.cache import pipeline_cache
from backend.app.ingest.downloader import download_video
from backend.app.ingest.ffmpeg_utils import extract_audio, create_proxy_video, get_duration
from backend.app.ingest.chat_fetch import fetch_chat
from backend.app.audio.asr import transcribe_audio
from backend.app.audio.diarize import diarize_audio, assign_speakers_to_words
from backend.app.audio.features import compute_audio_features
from backend.app.vision.faces import extract_faces_from_proxy
from backend.app.vision.ocr import compute_visual_score
from backend.app.chat.features import compute_chat_features
from backend.app.text.segment import segment_transcript_into_windows
from backend.app.text.scoring import score_text_windows
from backend.app.text.llm import GroqProvider
from backend.app.rank.select import select_top_clips
from backend.app.render.render import render_clip
from backend.app.logging import logger

_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def execute_pipeline(job_id: str, params: JobParams) -> List[Clip]:
    cache_key = f"{params.source_url or params.file_path}_{params.mode}_{params.aspect}"
    timings: Dict[str, float] = {}

    try:
        # ==========================================
        # S1: INGEST
        # ==========================================
        t0 = time.time()
        progress_tracker.update_job_sync(job_id, stage="downloading", progress=0.1, status="downloading")

        job_dir = settings.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        source_mp4, metadata = download_video(job_id, params.source_url, params.file_path, settings.JOBS_DIR)
        duration = get_duration(source_mp4)
        metadata["duration"] = duration

        audio_wav = job_dir / "audio.wav"
        proxy_mp4 = job_dir / "proxy.mp4"
        chat_json = job_dir / "chat.json"

        if not audio_wav.exists():
            extract_audio(source_mp4, audio_wav)
        if not proxy_mp4.exists():
            create_proxy_video(source_mp4, proxy_mp4)

        messages = fetch_chat(params.source_url, chat_json)
        timings["ingest"] = time.time() - t0

        # ==========================================
        # S2: AUDIO (ASR, Diarization, Features)
        # ==========================================
        t1 = time.time()
        progress_tracker.update_job_sync(job_id, stage="transcribing", progress=0.25, status="transcribing", timings=timings)

        segments, words = transcribe_audio(audio_wav)
        intervals, diar_method = diarize_audio(audio_wav, hf_token=settings.HF_TOKEN)
        words = assign_speakers_to_words(words, intervals)

        raw_audio_feats, audio_score, music_ratio, audio_avail = compute_audio_features(audio_wav, duration, words)
        timings["audio"] = time.time() - t1

        # ==========================================
        # S3: VISION & S4: CHAT
        # ==========================================
        t2 = time.time()
        progress_tracker.update_job_sync(job_id, stage="analyzing", progress=0.45, status="analyzing", timings=timings)

        # Vision: faces, scenes, clip zero-shot, OCR
        face_tracks, face_score, faces_avail = extract_faces_from_proxy(
            proxy_mp4,
            duration,
            source_width=1920,
            source_height=1080,
        )

        visual_score, action_prob, cut_density, visual_avail = compute_visual_score(
            proxy_mp4,
            duration,
            keywords=params.keywords,
        )

        # Chat features
        raw_chat_feats, chat_score, chat_avail = compute_chat_features(
            messages=messages,
            duration=duration,
            audio_energy=raw_audio_feats.get("rms"),
            keywords=params.keywords,
        )
        timings["vision_and_chat"] = time.time() - t2

        # ==========================================
        # S5: TEXT & SEMANTIC SCORING
        # ==========================================
        t3 = time.time()
        text_windows = segment_transcript_into_windows(segments, min_window=20.0, max_window=90.0)

        # Groq provider if API key present
        llm_provider = GroqProvider() if settings.GROQ_API_KEY else None
        text_score, scored_windows, text_avail = score_text_windows(
            windows=text_windows,
            duration=duration,
            provider=llm_provider,
            keywords=params.keywords,
        )
        timings["text"] = time.time() - t3

        # Construct multimodal SeriesBundle
        available_flags = {
            "audio": audio_avail,
            "chat": chat_avail,
            "text": text_avail,
            "visual": visual_avail,
            "face": faces_avail,
        }
        fallback_reasons = {}
        if not chat_avail:
            fallback_reasons["chat"] = "No live chat replay available"
        if not faces_avail:
            fallback_reasons["face"] = "No human faces detected in frame"
        if not text_avail:
            fallback_reasons["text"] = "Heuristic text score used"

        bundle = SeriesBundle.from_numpy(
            duration=duration,
            audio=audio_score,
            chat=chat_score,
            text=text_score,
            visual=visual_score,
            face=face_score,
            available=available_flags,
            fallback_reasons=fallback_reasons,
            raw_features={"music_ratio": music_ratio},
        )

        # Save SeriesBundle to stage cache for instant reranking
        pipeline_cache.save_stage(cache_key, "bundle", bundle.model_dump())

        # ==========================================
        # S6: FUSION & RANKING
        # ==========================================
        t4 = time.time()
        progress_tracker.update_job_sync(job_id, stage="ranking", progress=0.70, status="ranking", timings=timings)

        ranked_clips, shortfall_reason = select_top_clips(
            bundle=bundle,
            words=words,
            num_clips=params.num_clips,
            min_len=float(params.min_len),
            max_len=float(params.max_len),
            mode=params.mode,
            quality_floor=0.8,
            provider=llm_provider,
        )

        # If shortfall or no clips met quality floor, relax quality floor to guarantee valid output
        if not ranked_clips:
            ranked_clips, _ = select_top_clips(
                bundle=bundle,
                words=words,
                num_clips=params.num_clips,
                min_len=float(params.min_len),
                max_len=float(params.max_len),
                mode=params.mode,
                quality_floor=-5.0,  # capture top peaks regardless of absolute score
                provider=llm_provider,
            )

        timings["ranking"] = time.time() - t4

        # ==========================================
        # S7: RENDER CLIPS
        # ==========================================
        t5 = time.time()
        progress_tracker.update_job_sync(job_id, stage="rendering", progress=0.85, status="rendering", timings=timings)

        final_clips: List[Clip] = []
        creator_credit = metadata.get("author") or metadata.get("channel") or ""

        for idx, clip in enumerate(ranked_clips):
            rendered = render_clip(
                job_id=job_id,
                clip_idx=idx + 1,
                source_path=source_mp4,
                clip=clip,
                words=words,
                face_tracks=face_tracks,
                aspect=params.aspect,
                out_dir=settings.OUT_DIR,
                attribution=creator_credit,
                captions=params.captions,
            )
            final_clips.append(rendered)

        timings["rendering"] = time.time() - t5
        timings["total"] = sum(timings.values())

        # Save clips metadata to stage cache
        pipeline_cache.save_stage(cache_key, "clips", [c.model_dump() for c in final_clips])

        # Mark done
        progress_tracker.update_job_sync(
            job_id,
            stage="done",
            progress=1.0,
            status="done",
            timings=timings,
        )
        return final_clips

    except Exception as e:
        logger.error("pipeline_execution_failed", job_id=job_id, error=str(e), exc_info=True)
        progress_tracker.update_job_sync(
            job_id,
            stage="failed",
            progress=0.0,
            status="failed",
            error=str(e),
        )
        raise


def submit_job_async(job_id: str, params: JobParams) -> None:
    _EXECUTOR.submit(execute_pipeline, job_id, params)
