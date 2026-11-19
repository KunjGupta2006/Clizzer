import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from backend.app.models import ChatMsg
from backend.app.chat.lexicons import HYPE_LEXICON, LAUGH_LEXICON, SHOCK_LEXICON, CLIP_IT_PHRASES
from backend.app.chat.lag import cross_correlate_lag, shift_chat_series
from backend.app.audio.features import rolling_median_mad
from backend.app.logging import logger

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_ANALYZER = SentimentIntensityAnalyzer()
except ImportError:
    _VADER_ANALYZER = None


def filter_copypasta(messages: List[ChatMsg], window_sec: float = 2.0) -> List[ChatMsg]:
    if not messages:
        return []

    filtered: List[ChatMsg] = []
    recent_texts: List[Tuple[float, str]] = []

    for msg in sorted(messages, key=lambda m: m.t):
        cleaned = re.sub(r"\s+", " ", msg.text.strip().lower())
        # Clean out older than window
        recent_texts = [item for item in recent_texts if (msg.t - item[0]) <= window_sec]

        # Count duplicates in window
        count = sum(1 for item in recent_texts if item[1] == cleaned)
        if count < 3:  # Allow max 3 identical repetitions in 2s
            filtered.append(msg)
            recent_texts.append((msg.t, cleaned))

    return filtered


def compute_chat_features(
    messages: List[ChatMsg],
    duration: float,
    audio_energy: Optional[np.ndarray] = None,
    keywords: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool]:
    length = int(np.ceil(duration))
    raw_features = {
        "velocity": np.zeros(length, dtype=np.float32),
        "unique_ratio": np.zeros(length, dtype=np.float32),
        "lexicon_density": np.zeros(length, dtype=np.float32),
        "clip_it": np.zeros(length, dtype=np.float32),
        "sentiment": np.zeros(length, dtype=np.float32),
        "keyword_hits": np.zeros(length, dtype=np.float32),
    }

    if not messages:
        return raw_features, np.zeros(length, dtype=np.float32), False

    cleaned_msgs = filter_copypasta(messages)
    if not cleaned_msgs:
        return raw_features, np.zeros(length, dtype=np.float32), False

    # 5-second sliding windows centered at each second
    half_bin = 2.5
    kw_set = {k.lower() for k in (keywords or [])}

    for sec in range(length):
        start = max(0.0, sec - half_bin)
        end = min(duration, sec + half_bin)
        bin_msgs = [m for m in cleaned_msgs if start <= m.t < end]

        if not bin_msgs:
            continue

        # 1. Message velocity
        raw_features["velocity"][sec] = float(len(bin_msgs))

        # 2. Unique chatter ratio
        unique_authors = len({m.author for m in bin_msgs})
        raw_features["unique_ratio"][sec] = float(unique_authors / len(bin_msgs))

        # 3. Lexicons (hype, laugh, shock)
        lex_count = 0
        clip_it_count = 0
        kw_count = 0
        sent_sum = 0.0

        for m in bin_msgs:
            txt_lower = m.text.lower()
            tokens = set(re.findall(r"\w+", txt_lower))

            # Check clip it phrases
            if any(p in txt_lower for p in CLIP_IT_PHRASES):
                clip_it_count += 1

            # Check hype / laugh / shock
            if tokens & HYPE_LEXICON or tokens & LAUGH_LEXICON or tokens & SHOCK_LEXICON:
                lex_count += 1

            # Keyword hits
            if kw_set and (tokens & kw_set):
                kw_count += 1

            if _VADER_ANALYZER is not None:
                sent_score = _VADER_ANALYZER.polarity_scores(m.text)["compound"]
                sent_sum += sent_score

        raw_features["lexicon_density"][sec] = float(lex_count / len(bin_msgs))
        raw_features["clip_it"][sec] = float(clip_it_count * 2.0)  # burst x2
        raw_features["keyword_hits"][sec] = float(kw_count)
        raw_features["sentiment"][sec] = float(sent_sum / len(bin_msgs))

    # Cross-correlate lag vs audio energy and shift chat if audio provided
    if audio_energy is not None and len(audio_energy) > 0:
        lag = cross_correlate_lag(raw_features["velocity"], audio_energy, max_lag=15)
        if lag > 0:
            logger.info("detected_chat_lag", lag_sec=lag)
            for k in raw_features:
                raw_features[k] = shift_chat_series(raw_features[k], lag)

    # Normalize velocity
    norm_vel = rolling_median_mad(raw_features["velocity"], window_size=120)
    norm_lex = rolling_median_mad(raw_features["lexicon_density"], window_size=120)
    norm_clip = rolling_median_mad(raw_features["clip_it"], window_size=120)

    # Formula: chat_score = .4 velocity_z + .2 unique_ratio + .2 lexicon + .2 clip_it
    chat_score = (
        0.4 * norm_vel
        + 0.2 * raw_features["unique_ratio"]
        + 0.2 * norm_lex
        + 0.2 * norm_clip
    )
    chat_score = np.clip(chat_score, -3.0, 3.0).astype(np.float32)

    return raw_features, chat_score, True
