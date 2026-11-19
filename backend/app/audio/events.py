from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from backend.app.logging import logger

_AST_PIPELINE = None


def get_ast_pipeline() -> Optional[object]:
    global _AST_PIPELINE
    if _AST_PIPELINE is not None:
        return _AST_PIPELINE

    try:
        from transformers import pipeline
        # Fast initialization if weights cached
        _AST_PIPELINE = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
        return _AST_PIPELINE
    except Exception as e:
        logger.debug("ast_pipeline_not_loaded", error=str(e), fallback="heuristic_events")
        return None


def detect_audio_events(
    wav_path: Path,
    duration: float,
    y: Optional[np.ndarray] = None,
    sr: int = 16000,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    # Returns (event_excitement, music_ratio, available)
    length = int(np.ceil(duration))
    excitement = np.zeros(length, dtype=np.float32)
    music_ratio = np.zeros(length, dtype=np.float32)

    if y is None or len(y) == 0:
        try:
            import soundfile as sf
            y, sr = sf.read(str(wav_path))
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception:
            return excitement, music_ratio, False

    pipeline = get_ast_pipeline()
    if pipeline is not None:
        try:
            # Run 1s windows
            for sec in range(length):
                start_sample = int(sec * sr)
                end_sample = min(int((sec + 1) * sr), len(y))
                if end_sample <= start_sample:
                    break
                chunk = y[start_sample:end_sample]
                if len(chunk) < sr // 4:
                    continue
                results = pipeline(chunk, top_k=8)
                score_dict = {r["label"].lower(): r["score"] for r in results}

                # Excitement classes: laughter, screaming, applause, gunshot, explosion, siren
                ex_sum = 0.0
                for cat in ["laughter", "screaming", "applause", "gunshot", "explosion", "siren", "cheering", "yell"]:
                    ex_sum += score_dict.get(cat, 0.0)

                music_score = score_dict.get("music", 0.0)
                speech_score = score_dict.get("speech", 0.0)

                excitement[sec] = float(np.clip(ex_sum, 0.0, 1.0))
                denom = music_score + speech_score + 1e-6
                music_ratio[sec] = float(np.clip(music_score / denom, 0.0, 1.0))

            return excitement, music_ratio, True
        except Exception as e:
            logger.warning("ast_inference_failed", error=str(e), fallback="heuristic_events")

    # Heuristic fallback using spectral energy and periodicity
    for sec in range(length):
        start_sample = int(sec * sr)
        end_sample = min(int((sec + 1) * sr), len(y))
        if end_sample <= start_sample:
            break
        chunk = y[start_sample:end_sample]
        rms_val = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
        # If sudden spike in amplitude
        excitement[sec] = float(np.clip(rms_val * 4.0, 0.0, 1.0))
        # Default low music ratio
        music_ratio[sec] = 0.05

    return excitement, music_ratio, False
