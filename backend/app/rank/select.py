import random
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from backend.app.models import Candidate, Clip, SeriesBundle, Word
from backend.app.rank.fusion import fuse_signals
from backend.app.rank.candidates import find_candidate_peaks
from backend.app.rank.snap import snap_candidate_boundaries
from backend.app.rank.dedup import deduplicate_candidates
from backend.app.text.llm import LLMProvider
from backend.app.text.prompts import GTA6_SYSTEM_PROMPT, CLIP_METADATA_PROMPT
from backend.app.logging import logger


class ClipMetadataOutput(BaseModel):
    title: str = Field(..., max_length=70)
    hook: str = ""
    hashtags: List[str] = Field(default_factory=list)
    reason: str = ""


def generate_metadata_for_clips(
    candidates: List[Candidate],
    words: List[Word],
    provider: Optional[LLMProvider] = None,
) -> List[Clip]:
    clips: List[Clip] = []

    for i, cand in enumerate(candidates):
        # Extract transcript excerpt for candidate
        clip_words = [w for w in words if cand.start <= w.start <= cand.end]
        transcript_excerpt = " ".join([w.text for w in clip_words]).strip() or "Epic gameplay moment and viral stream reaction."

        title = f"GTA 6 Insane Highlight #{i + 1}"
        hook = "You won't believe what happened next!"
        hashtags = ["#GTA6", "#Gaming", "#ViceCity", "#ViralClips"]
        reason = cand.reason

        if cand.music_ratio > 0.6:
            reason += " [Notice: High music content (>60%), possible copyright risk]"

        if provider is not None:
            try:
                user_prompt = CLIP_METADATA_PROMPT.format(transcript=transcript_excerpt[:500])
                resp = provider.complete(GTA6_SYSTEM_PROMPT, user_prompt, schema=ClipMetadataOutput)
                if resp.get("title"):
                    title = resp["title"][:70]
                if resp.get("hook"):
                    hook = resp["hook"]
                if resp.get("hashtags"):
                    hashtags = resp["hashtags"][:5]
                if resp.get("reason"):
                    reason = resp["reason"]
            except Exception as e:
                logger.debug("llm_metadata_fallback_used", error=str(e))

        clips.append(
            Clip(
                start=cand.start,
                end=cand.end,
                peak_t=cand.peak_t,
                fused_score=cand.fused_score,
                breakdown=cand.breakdown,
                reason=reason,
                music_ratio=cand.music_ratio,
                title=title,
                hook=hook,
                hashtags=hashtags,
            )
        )

    return clips


def select_top_clips(
    bundle: SeriesBundle,
    words: List[Word],
    num_clips: int = 5,
    min_len: float = 20.0,
    max_len: float = 60.0,
    mode: str = "auto",
    custom_weights: Optional[Dict[str, float]] = None,
    quality_floor: float = 0.8,
    provider: Optional[LLMProvider] = None,
    seed: int = 42,
) -> Tuple[List[Clip], Optional[str]]:
    # Set seeds for deterministic ranking
    random.seed(seed)
    np.random.seed(seed)

    duration = bundle.duration

    # 1. Fuse signals with weights
    fused, _ = fuse_signals(bundle, mode=mode, custom_weights=custom_weights)

    # 2. Peak picking
    candidates = find_candidate_peaks(
        fused_series=fused,
        duration=duration,
        bundle=bundle,
        min_len=min_len,
        max_len=max_len,
    )

    # Attach music ratio from raw features if available
    raw_music = bundle.raw_features.get("music_ratio", [])
    for cand in candidates:
        sec = int(round(cand.peak_t))
        if sec < len(raw_music):
            cand.music_ratio = float(raw_music[sec])

    # 3. Boundary snapping
    snapped = [
        snap_candidate_boundaries(c, words, duration=duration)
        for c in candidates
    ]

    # 4. Deduplicate (NMS + MMR)
    deduped = deduplicate_candidates(snapped, min_spacing_sec=45.0, iou_threshold=0.3, mmr_lambda=0.7)

    # 5. Quality floor check (fused >= 0.8)
    qualified = [c for c in deduped if c.fused_score >= quality_floor]

    shortfall_reason = None
    if len(qualified) < num_clips:
        shortfall_reason = f"Only {len(qualified)}/{num_clips} candidate moments met the quality floor (>= {quality_floor})."

    # Select top N (never pad!)
    selected_cands = qualified[:num_clips]

    # Generate metadata / titles
    clips = generate_metadata_for_clips(selected_cands, words, provider=provider)
    return clips, shortfall_reason
