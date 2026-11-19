from typing import Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
from backend.app.config import settings
from backend.app.models import SeriesBundle
from backend.app.logging import logger


def get_mode_weights(mode: str = "auto", custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if custom_weights:
        return dict(custom_weights)
    presets = settings.WEIGHT_PRESETS
    return dict(presets.get(mode, presets["auto"]))


def fuse_signals(
    bundle: SeriesBundle,
    mode: str = "auto",
    custom_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    weights = get_mode_weights(mode, custom_weights)
    arrays = bundle.to_arrays()
    length = bundle.length

    # Check available modalities
    available_weights: Dict[str, float] = {}
    for mod in ["audio", "chat", "text", "visual", "face"]:
        if bundle.available.get(mod, True) and mod in weights:
            available_weights[mod] = float(weights[mod])
        else:
            available_weights[mod] = 0.0
            logger.info("modality_zero_weighted", modality=mod, reason=bundle.fallback_reasons.get(mod))

    # Renormalize available weights so sum == 1.0
    total_w = sum(available_weights.values())
    if total_w > 1e-6:
        renorm_weights = {k: v / total_w for k, v in available_weights.items()}
    else:
        # Fallback to audio + visual if all zeroed
        renorm_weights = {"audio": 0.5, "visual": 0.5, "chat": 0.0, "text": 0.0, "face": 0.0}

    # S(t) = wa*A + wc*C + wt*T + wv*V + wf*F
    fused = np.zeros(length, dtype=np.float32)
    for mod, w in renorm_weights.items():
        if w > 0.0 and mod in arrays:
            fused += float(w) * arrays[mod][:length]

    # Smooth with Gaussian filter (sigma=1.2 corresponds to ~5s kernel)
    if length > 5:
        smoothed = gaussian_filter1d(fused, sigma=1.2)
    else:
        smoothed = fused

    # Update bundle fused series
    bundle.fused = [float(x) for x in smoothed]
    return smoothed, renorm_weights
