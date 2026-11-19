import numpy as np


def cross_correlate_lag(
    chat_velocity: np.ndarray,
    audio_energy: np.ndarray,
    max_lag: int = 15,
) -> int:
    if len(chat_velocity) == 0 or len(audio_energy) == 0:
        return 0

    n = min(len(chat_velocity), len(audio_energy))
    c = chat_velocity[:n] - np.mean(chat_velocity[:n])
    a = audio_energy[:n] - np.mean(audio_energy[:n])

    c_norm = np.linalg.norm(c)
    a_norm = np.linalg.norm(a)

    if c_norm < 1e-6 or a_norm < 1e-6:
        return 0

    best_lag = 0
    max_corr = -1.0

    # Test lag from 0 to max_lag seconds (chat reacts after audio event)
    for lag in range(0, min(max_lag + 1, n // 2)):
        if n - lag <= 0:
            break
        # chat delayed by lag: compare audio[0 : n-lag] with chat[lag : n]
        corr = np.dot(a[: n - lag], c[lag:n]) / (c_norm * a_norm)
        if corr > max_corr:
            max_corr = corr
            best_lag = lag

    return best_lag


def shift_chat_series(chat_series: np.ndarray, lag_sec: int) -> np.ndarray:
    if lag_sec <= 0 or len(chat_series) == 0:
        return chat_series

    shifted = np.zeros_like(chat_series)
    # Shift backwards so reaction event lines up with cause
    shifted[: -lag_sec] = chat_series[lag_sec:]
    # Fill tail with last value or zero
    shifted[-lag_sec:] = 0.0
    return shifted
