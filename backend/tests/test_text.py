import json
from pathlib import Path
from unittest.mock import MagicMock
import httpx
import numpy as np
import pytest
from backend.app.models import Segment
from backend.app.text.segment import segment_transcript_into_windows
from backend.app.text.llm import GroqProvider, MockProvider, strip_code_fences
from backend.app.text.scoring import (
    score_text_windows,
    compute_heuristic_text_score,
    BatchScoringSchema,
)
from backend.app.errors import LLMUnavailable


def test_segment_transcript_into_windows():
    # 10 segments of 5s each = 50s total
    segments = [
        Segment(start=i * 5.0, end=(i + 1) * 5.0, text=f"Sentence {i} about GTA 6 Vice City.")
        for i in range(10)
    ]
    windows = segment_transcript_into_windows(segments, min_window=20.0, max_window=40.0)
    assert len(windows) >= 1
    for win in windows:
        assert win["duration"] >= 15.0
        assert "start" in win and "end" in win
        assert "text" in win


def test_strip_code_fences():
    raw = "```json\n{\"foo\": \"bar\"}\n```"
    assert strip_code_fences(raw) == '{"foo": "bar"}'
    raw2 = "```\n{\"foo\": \"bar\"}\n```"
    assert strip_code_fences(raw2) == '{"foo": "bar"}'


def test_groq_provider_caching(tmp_path: Path):
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"windows": [{"id": 1, "hook": 8.0, "payoff": 7.0, "standalone": 9.0, "quotable": 8.5, "novelty": 7.5, "reason": "Great hook"}]})}}]
    }
    mock_client.post.return_value = mock_resp

    provider = GroqProvider(api_key="gsk_mock_test_key", client=mock_client)
    provider.cache.cache_dir = tmp_path

    # First call: makes HTTP request
    result1 = provider.complete("system prompt", "user query", schema=BatchScoringSchema)
    assert len(result1["windows"]) == 1
    assert mock_client.post.call_count == 1

    # Second call with identical payload: hits cache, 0 new HTTP calls
    result2 = provider.complete("system prompt", "user query", schema=BatchScoringSchema)
    assert len(result2["windows"]) == 1
    assert mock_client.post.call_count == 1


def test_groq_provider_429_retry_after(tmp_path: Path):
    mock_client = MagicMock(spec=httpx.Client)

    # 1st call returns 429 with Retry-After header, 2nd call succeeds
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.01"}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"windows": []})}}]
    }

    mock_client.post.side_effect = [resp_429, resp_200]

    provider = GroqProvider(api_key="gsk_mock_test_key", client=mock_client)
    provider.cache.cache_dir = tmp_path
    res = provider.complete("sys", "user", schema=BatchScoringSchema)
    assert res == {"windows": []}
    assert mock_client.post.call_count == 2


def test_malformed_json_retries_and_falls_back():
    mock_client = MagicMock(spec=httpx.Client)
    # Always return broken json
    bad_resp = MagicMock()
    bad_resp.status_code = 200
    bad_resp.json.return_value = {
        "choices": [{"message": {"content": "not valid json at all {"}}]
    }
    mock_client.post.return_value = bad_resp

    provider = GroqProvider(api_key="gsk_mock_test_key", client=mock_client)

    # Calling complete directly raises LLMUnavailable after retries
    with pytest.raises(LLMUnavailable):
        provider.complete("sys", "user", schema=BatchScoringSchema)

    # When used in score_text_windows, it catches LLMUnavailable and uses heuristic fallback
    windows = [
        {"id": 1, "start": 0.0, "end": 25.0, "duration": 25.0, "text": "GTA 6 trailer is out now with Lucia."}
    ]
    norm_scores, scored_wins, llm_avail = score_text_windows(
        windows=windows,
        duration=30.0,
        provider=provider,
        keywords=["GTA", "Lucia"],
    )

    assert llm_avail is False
    assert len(norm_scores) == 30
    assert not np.isnan(norm_scores).any()
    assert (norm_scores >= -3.0).all() and (norm_scores <= 3.0).all()
    # Window should receive heuristic score
    assert scored_wins[0]["reason"] == "Heuristic text score"
