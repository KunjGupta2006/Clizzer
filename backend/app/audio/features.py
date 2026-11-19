from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from backend.app.models import Word
from backend.app.audio.events import detect_audio_events
from backend.app.logging import logger

try:
    import soundfile as sf
except ImportError:
    sf = None


def rolling_median_mad(arr: np.ndarray, window_size: int = 120) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr

    half = window_size // 2
    z_scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        sub = arr[s:e]
        med = np.median(sub)
        mad = np.median(np.abs(sub - med))
        # 1.4826 is normal consistency factor; eps = 1e-6
        scale = mad * 1.4826 + 1e-6
        z = (arr[i] - med) / scale
        z_scores[i] = float(np.clip(z, -3.0, 3.0))

    # Replace any potential NaNs or Infs
    z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=3.0, neginf=-3.0)
    return z_scores


def compute_audio_features(
    wav_path: Path,
    duration: float,
    words: List[Word],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, bool]:
    length = int(np.ceil(duration))
    raw_features: Dict[str, np.ndarray] = {
        "rms": np.zeros(length, dtype=np.float32),
        "spectral_flux": np.zeros(length, dtype=np.float32),
        "pitch_var": np.zeros(length, dtype=np.float32),
        "speech_rate": np.zeros(length, dtype=np.float32),
        "overlap_ratio": np.zeros(length, dtype=np.float32),
        "event_excitement": np.zeros(length, dtype=np.float32),
    }

    y = None
    sr = 16000
    available = True

    try:
        if sf is not None and wav_path.exists():
            y, sr = sf.read(str(wav_path))
            if y.ndim > 1:
                y = np.mean(y, axis=1)
    except Exception as e:
        logger.warning("failed_to_read_wav", wav=str(wav_path), error=str(e))
        available = False

    if y is not None and len(y) > 0:
        # 1. RMS per second
        for sec in range(length):
            s_idx = int(sec * sr)
            e_idx = min(int((sec + 1) * sr), len(y))
            if e_idx > s_idx:
                chunk = y[s_idx:e_idx]
                raw_features["rms"][sec] = float(np.sqrt(np.mean(chunk**2)))

        # 2. Spectral Flux (difference in power spectrum between consecutive 1s frames)
        prev_spec = None
        for sec in range(length):
            s_idx = int(sec * sr)
            e_idx = min(int((sec + 1) * sr), len(y))
            if e_idx > s_idx:
                chunk = y[s_idx:e_idx]
                spec = np.abs(np.fft.rfft(chunk, n=512))
                if prev_spec is not None:
                    diff = np.maximum(0.0, spec - prev_spec)
                    raw_features["spectral_flux"][sec] = float(np.sum(diff**2))
                prev_spec = spec

        # 3. Pitch variance proxy (zero-crossing rate variance across subframes)
        for sec in range(length):
            s_idx = int(sec * sr)
            e_idx = min(int((sec + 1) * sr), len(y))
            if e_idx > s_idx:
                chunk = y[s_idx:e_idx]
                # Subdivide into 4 subchunks
                sub_zcrs = []
                sub_size = max(1, len(chunk) // 4)
                for sc in range(4):
                    sub = chunk[sc * sub_size : (sc + 1) * sub_size]
                    if len(sub) > 1:
                        zc = np.sum(np.abs(np.diff(np.signbit(sub)))) / len(sub)
                        sub_zcrs.append(zc)
                if sub_zcrs:
                    raw_features["pitch_var"][sec] = float(np.var(sub_zcrs))

    # 4. Speech rate: count words that intersect [sec, sec+1)
    for w in words:
        s_sec = max(0, min(length - 1, int(np.floor(w.start))))
        e_sec = max(0, min(length - 1, int(np.floor(w.end))))
        for sec in range(s_sec, e_sec + 1):
            raw_features["speech_rate"][sec] += 1.0

    # 5. Overlap ratio: detect multiple active words at same time
    word_coverage = np.zeros(length, dtype=np.int32)
    for w in words:
        s_sec = max(0, min(length - 1, int(np.floor(w.start))))
        e_sec = max(0, min(length - 1, int(np.floor(w.end))))
        for sec in range(s_sec, e_sec + 1):
            word_coverage[sec] += 1
    raw_features["overlap_ratio"] = np.where(word_coverage > 1, 1.0, 0.0).astype(np.float32)

    # 6. Events excitement and music ratio
    excitement, music_ratio, _ = detect_audio_events(wav_path, duration, y=y, sr=sr)
    raw_features["event_excitement"] = excitement

    # Normalize each component with 120s rolling median/MAD
    norm_rms = rolling_median_mad(raw_features["rms"], window_size=120)
    norm_flux = rolling_median_mad(raw_features["spectral_flux"], window_size=120)
    norm_pitch = rolling_median_mad(raw_features["pitch_var"], window_size=120)
    norm_rate = rolling_median_mad(raw_features["speech_rate"], window_size=120)
    norm_events = rolling_median_mad(raw_features["event_excitement"], window_size=120)

    # Formula: audio_score = .35 rms + .2 flux + .15 pitch_var + .15 speech_rate + .15 event_excitement
    audio_score = (
        0.35 * norm_rms
        + 0.20 * norm_flux
        + 0.15 * norm_pitch
        + 0.15 * norm_rate
        + 0.15 * norm_events
    )
    # Clip audio_score to [-3.0, 3.0]
    audio_score = np.clip(audio_score, -3.0, 3.0).astype(np.float32)

    return raw_features, audio_score, music_ratio, available
