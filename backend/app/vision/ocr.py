from pathlib import Path
from typing import List, Optional, Tuple
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from backend.app.vision.scenes import detect_scene_cuts, compute_cut_density
from backend.app.vision.clipzs import classify_proxy_frames
from backend.app.audio.features import rolling_median_mad
from backend.app.logging import logger

_OCR_ENGINE = None


def get_ocr_engine() -> Optional[object]:
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
        return _OCR_ENGINE
    except Exception as e:
        logger.debug("rapidocr_not_available", error=str(e), fallback="heuristic_ocr")
        return None


def run_ocr_on_proxy(
    proxy_path: Path,
    duration: float,
    keywords: List[str],
) -> np.ndarray:
    length = int(np.ceil(duration))
    ocr_hits = np.zeros(length, dtype=np.float32)

    if cv2 is None or not proxy_path.exists() or not keywords:
        return ocr_hits

    kw_lower = [k.lower() for k in keywords]
    engine = get_ocr_engine()

    cap = cv2.VideoCapture(str(proxy_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        sec = int(np.floor(t))
        if sec >= length:
            break

        # Run OCR every 3 seconds
        if sec % 3 == 0 and engine is not None:
            try:
                result, _ = engine(frame)
                if result:
                    detected_text = " ".join([line[1].lower() for line in result])
                    for kw in kw_lower:
                        if kw in detected_text:
                            # Apply hit to surrounding 3 seconds
                            for s in range(max(0, sec - 1), min(length, sec + 2)):
                                ocr_hits[s] = 1.0
                            break
            except Exception:
                pass

        frame_idx += int(fps)  # skip forward ~ 1 sec

    cap.release()
    return ocr_hits


def compute_visual_score(
    proxy_path: Path,
    duration: float,
    keywords: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    # Returns (visual_score, action_prob, cut_density, available)
    length = int(np.ceil(duration))

    # 1. Cuts
    cuts = detect_scene_cuts(proxy_path, duration)
    cut_density = compute_cut_density(cuts, duration, window_sec=10.0)

    # 2. CLIP Zero-shot
    action_prob, trailer_prob, clip_avail = classify_proxy_frames(proxy_path, duration)

    # 3. OCR hits
    ocr_hits = run_ocr_on_proxy(proxy_path, duration, keywords)

    # Formula: visual_score = .4 action_labels + .3 cut_density + .3 trailer_prob
    # trailer_prob>0.6 adds +1.0; keyword hit +0.5 to visual
    raw_score = (
        0.4 * action_prob
        + 0.3 * cut_density
        + 0.3 * trailer_prob
        + np.where(trailer_prob > 0.6, 1.0, 0.0)
        + (0.5 * ocr_hits)
    )

    norm_score = rolling_median_mad(raw_score, window_size=120)
    return norm_score, action_prob, cut_density, True
