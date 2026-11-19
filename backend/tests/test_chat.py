import numpy as np
from backend.app.models import ChatMsg
from backend.app.chat.lexicons import HYPE_LEXICON, LAUGH_LEXICON, CLIP_IT_PHRASES
from backend.app.chat.lag import cross_correlate_lag, shift_chat_series
from backend.app.chat.features import compute_chat_features, filter_copypasta


def test_injected_chat_burst_detected_within_3s():
    duration = 100.0
    messages = []

    # Background chat: 1 message every 2 seconds
    for t in range(0, int(duration), 2):
        messages.append(ChatMsg(t=float(t), author=f"user_{t}", text="hello stream"))

    # Injected burst at t=45..47: 50 enthusiastic messages with hype, laugh, and clip-it phrases
    target_time = 46.0
    for i in range(50):
        t_burst = 45.0 + (i % 3) * 0.8
        phrase = "clip it! LMAO POGGERS that was insane!"
        messages.append(ChatMsg(t=t_burst, author=f"hyped_user_{i}", text=phrase))

    raw_feats, chat_score, available = compute_chat_features(
        messages=messages,
        duration=duration,
        keywords=["insane"],
    )

    assert available is True
    assert len(chat_score) == int(duration)
    assert not np.isnan(chat_score).any()

    # Peak must be within +/- 3s of target_time (46.0)
    peak_idx = int(np.argmax(chat_score))
    assert abs(peak_idx - target_time) <= 3.0, f"Expected peak near {target_time}, found {peak_idx}"


def test_filter_copypasta():
    messages = [
        ChatMsg(t=1.0, author="bot1", text="BUY BITCOIN NOW"),
        ChatMsg(t=1.1, author="bot2", text="buy bitcoin now"),
        ChatMsg(t=1.2, author="bot3", text="BUY BITCOIN NOW"),
        ChatMsg(t=1.3, author="bot4", text="BUY BITCOIN NOW"),  # 4th duplicate in <2s
        ChatMsg(t=1.4, author="bot5", text="BUY BITCOIN NOW"),  # 5th duplicate
        ChatMsg(t=2.0, author="legit", text="cool stream bro"),
    ]
    filtered = filter_copypasta(messages, window_sec=2.0)
    assert len(filtered) < len(messages)
    assert any(m.author == "legit" for m in filtered)


def test_chat_lag_correlation():
    # Audio energy peaks at t=20
    audio_energy = np.zeros(60, dtype=np.float32)
    audio_energy[20] = 5.0

    # Chat velocity reacts with a 4-second delay at t=24
    chat_velocity = np.zeros(60, dtype=np.float32)
    chat_velocity[24] = 10.0

    lag = cross_correlate_lag(chat_velocity, audio_energy, max_lag=10)
    assert lag == 4

    shifted = shift_chat_series(chat_velocity, lag_sec=lag)
    # The peak should now align back at t=20
    assert np.argmax(shifted) == 20


def test_empty_chat_fallback():
    raw_feats, chat_score, available = compute_chat_features(
        messages=[],
        duration=30.0,
    )
    assert available is False
    assert len(chat_score) == 30
    assert (chat_score == 0.0).all()
