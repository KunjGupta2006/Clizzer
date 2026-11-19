from typing import Any, Dict, List, Optional
import numpy as np
from backend.app.models import Segment
from backend.app.logging import logger

_MINILM_MODEL = None


def get_minilm_model() -> Optional[Any]:
    global _MINILM_MODEL
    if _MINILM_MODEL is not None:
        return _MINILM_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        # Check if model is cached locally
        _MINILM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _MINILM_MODEL
    except Exception as e:
        logger.debug("minilm_not_loaded", error=str(e), fallback="heuristic_segmenter")
        return None


def segment_transcript_into_windows(
    segments: List[Segment],
    min_window: float = 20.0,
    max_window: float = 90.0,
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    windows: List[Dict[str, Any]] = []
    current_texts: List[str] = []
    start_t = segments[0].start
    speaker = segments[0].speaker
    model = get_minilm_model()

    prev_embedding = None

    for i, seg in enumerate(segments):
        current_texts.append(seg.text)
        current_dur = seg.end - start_t

        is_break = False
        # If reached max window, force break
        if current_dur >= max_window:
            is_break = True
        elif current_dur >= min_window:
            # Check semantic break or speaker turn
            if seg.speaker != speaker:
                is_break = True
            elif model is not None:
                try:
                    emb = model.encode(seg.text)
                    if prev_embedding is not None:
                        sim = float(np.dot(emb, prev_embedding) / (np.linalg.norm(emb) * np.linalg.norm(prev_embedding) + 1e-6))
                        if sim < 0.45:  # Topic boundary shift
                            is_break = True
                    prev_embedding = emb
                except Exception:
                    pass
            elif seg.text.endswith((".", "!", "?")):
                # Punctuation boundary
                is_break = True

        if is_break or i == len(segments) - 1:
            joined_text = " ".join(current_texts).strip()
            # If this is the last element and duration is too short, merge with previous window
            if i == len(segments) - 1 and current_dur < min_window and windows:
                windows[-1]["end"] = seg.end
                windows[-1]["text"] = (windows[-1]["text"] + " " + joined_text).strip()
                windows[-1]["duration"] = windows[-1]["end"] - windows[-1]["start"]
            elif joined_text:
                windows.append({
                    "id": len(windows) + 1,
                    "start": start_t,
                    "end": seg.end,
                    "text": joined_text,
                    "speaker": speaker,
                    "duration": seg.end - start_t,
                })
            current_texts = []
            if i + 1 < len(segments):
                start_t = segments[i + 1].start
                speaker = segments[i + 1].speaker
            prev_embedding = None

    return windows
