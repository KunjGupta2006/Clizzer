from typing import List, Optional
import numpy as np
from backend.app.models import Candidate


def temporal_iou(c1: Candidate, c2: Candidate) -> float:
    intersect_start = max(c1.start, c2.start)
    intersect_end = min(c1.end, c2.end)
    intersect = max(0.0, intersect_end - intersect_start)

    union_start = min(c1.start, c2.start)
    union_end = max(c1.end, c2.end)
    union = max(1e-6, union_end - union_start)

    return float(intersect / union)


def deduplicate_candidates(
    candidates: List[Candidate],
    min_spacing_sec: float = 45.0,
    iou_threshold: float = 0.3,
    mmr_lambda: float = 0.7,
) -> List[Candidate]:
    if len(candidates) <= 1:
        return candidates

    # 1. Non-Maximum Suppression (NMS)
    # Candidates are expected to be pre-sorted descending by score
    kept: List[Candidate] = []

    for cand in candidates:
        discard = False
        for k in kept:
            # Check temporal IoU
            iou = temporal_iou(cand, k)
            peak_dist = abs(cand.peak_t - k.peak_t)
            if iou > iou_threshold or peak_dist < min_spacing_sec:
                discard = True
                break
        if not discard:
            kept.append(cand)

    # 2. MMR Reranking
    if len(kept) <= 1:
        return kept

    mmr_selected: List[Candidate] = [kept[0]]
    remaining = kept[1:]

    while remaining:
        best_cand = None
        best_mmr_score = -999999.0
        best_idx = -1

        for idx, cand in enumerate(remaining):
            # Relevance score
            rel = cand.fused_score
            # Max similarity to already selected candidates
            max_sim = max(temporal_iou(cand, sel) for sel in mmr_selected)

            mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
            if mmr > best_mmr_score:
                best_mmr_score = mmr
                best_cand = cand
                best_idx = idx

        if best_cand is not None:
            mmr_selected.append(best_cand)
            remaining.pop(best_idx)

    return mmr_selected
