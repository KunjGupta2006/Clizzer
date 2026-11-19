from typing import List, Optional
from backend.app.models import Candidate, Word


def snap_candidate_boundaries(
    candidate: Candidate,
    words: List[Word],
    duration: float,
    max_snap_sec: float = 3.0,
    pre_roll: float = 0.3,
    tail_pad: float = 0.5,
) -> Candidate:
    if not words:
        # Just apply pre-roll and tail pad clamped to duration
        start = max(0.0, candidate.start - pre_roll)
        end = min(duration, candidate.end + tail_pad)
        candidate.start = start
        candidate.end = end
        return candidate

    new_start = candidate.start
    new_end = candidate.end

    # 1. Snap start boundary: look for word boundary <= 3s before candidate.start
    candidate_start_words = [
        w for w in words
        if abs(w.start - candidate.start) <= max_snap_sec
    ]
    if candidate_start_words:
        # Find closest word start
        best_word = min(candidate_start_words, key=lambda w: abs(w.start - candidate.start))
        new_start = best_word.start
    else:
        # Check if candidate.start is mid-word; if so, move to word start
        mid_word = next((w for w in words if w.start <= candidate.start <= w.end), None)
        if mid_word:
            new_start = mid_word.start

    # 2. Snap end boundary: look for word boundary <= 3s near candidate.end
    candidate_end_words = [
        w for w in words
        if abs(w.end - candidate.end) <= max_snap_sec
    ]
    if candidate_end_words:
        best_word_end = min(candidate_end_words, key=lambda w: abs(w.end - candidate.end))
        new_end = best_word_end.end
    else:
        mid_word_end = next((w for w in words if w.start <= candidate.end <= w.end), None)
        if mid_word_end:
            new_end = mid_word_end.end

    # 3. Add +0.3s pre-roll and +0.5s tail pad
    final_start = max(0.0, new_start - pre_roll)
    final_end = min(duration, new_end + tail_pad)

    if final_end <= final_start:
        final_end = min(duration, final_start + 20.0)

    candidate.start = float(final_start)
    candidate.end = float(final_end)
    return candidate
