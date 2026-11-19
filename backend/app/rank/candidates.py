from typing import Dict, List, Optional
import numpy as np
from scipy.signal import find_peaks
from backend.app.models import Candidate, SeriesBundle


def find_candidate_peaks(
    fused_series: np.ndarray,
    duration: float,
    bundle: Optional[SeriesBundle] = None,
    min_len: float = 20.0,
    max_len: float = 60.0,
    prominence: float = 0.5,
    distance_sec: int = 30,
) -> List[Candidate]:
    if len(fused_series) == 0:
        return []

    # Detect peaks
    peaks, properties = find_peaks(
        fused_series,
        prominence=prominence,
        distance=distance_sec,
    )

    # If no prominent peaks found, fall back to top local maxima
    if len(peaks) == 0:
        peaks, _ = find_peaks(fused_series, distance=distance_sec)
        if len(peaks) == 0:
            # Fall back to argmax
            peaks = np.array([int(np.argmax(fused_series))])

    arrays = bundle.to_arrays() if bundle else {}
    candidates: List[Candidate] = []

    for p in peaks:
        peak_t = float(p)
        peak_score = float(fused_series[p])

        # Adaptive length: default to center of [min_len, max_len]
        adaptive_len = (min_len + max_len) / 2.0
        clip_len = float(np.clip(adaptive_len, min_len, max_len))

        # Window: start = peak - 0.6*len, end = peak + 0.4*len
        start = max(0.0, peak_t - 0.6 * clip_len)
        end = min(duration, peak_t + 0.4 * clip_len)

        # Signal breakdown at peak
        breakdown: Dict[str, float] = {}
        for mod in ["audio", "chat", "text", "visual", "face"]:
            if mod in arrays and p < len(arrays[mod]):
                breakdown[mod] = float(arrays[mod][p])
            else:
                breakdown[mod] = 0.0

        candidates.append(
            Candidate(
                start=start,
                end=end,
                peak_t=peak_t,
                fused_score=peak_score,
                breakdown=breakdown,
                reason=f"Peak at {peak_t:.1f}s with score {peak_score:.2f}",
                music_ratio=0.0,
            )
        )

    # Sort descending by fused score
    candidates.sort(key=lambda c: c.fused_score, reverse=True)
    return candidates
