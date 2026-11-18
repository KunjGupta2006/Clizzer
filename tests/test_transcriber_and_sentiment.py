"""
Tests for Transcriber and SentimentAnalyzer modules.
"""

import os
import tempfile
from unittest.mock import MagicMock
import pytest

from backend.core.transcriber import Transcriber
from backend.core.sentiment import SentimentAnalyzer


class TestTranscriber:
    def test_init_and_lazy_load(self):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        assert t._model is None
        assert t.model_size == "tiny"
        assert t.compute_type == "int8"
        assert t.device == "cpu"

    def test_init_with_config_dict(self):
        t = Transcriber({"model_size": "base", "compute_type": "float32", "device": "cuda"})
        assert t.model_size == "base"
        assert t.compute_type == "float32"
        assert t.device == "cuda"

    def test_transcribe_invalid_inputs(self):
        t = Transcriber()
        with pytest.raises(ValueError):
            t.transcribe("")
        with pytest.raises(FileNotFoundError):
            t.transcribe("non_existent_audio_path_xyz.wav")

    def test_transcribe_mock(self):
        t = Transcriber()
        mock_model = MagicMock()
        mock_w1 = MagicMock(word="Hello", start=0.0, end=1.0, probability=0.95)
        mock_w2 = MagicMock(word="World", start=1.0, end=2.0, probability=0.98)
        mock_segment = MagicMock(id=0, text="Hello World", start=0.0, end=2.0, words=[mock_w1, mock_w2])
        mock_info = MagicMock(duration=2.0, language="en", language_probability=0.99)

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        t._model = mock_model

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            progress_records = []
            result = t.transcribe(temp_path, progress_callback=lambda p: progress_records.append(p))

            assert "segments" in result
            assert "full_text" in result
            assert "language" in result
            assert "language_probability" in result

            assert result["full_text"] == "Hello World"
            assert result["language"] == "en"
            assert result["language_probability"] == 0.99
            assert len(result["segments"]) == 1
            assert len(result["segments"][0]["words"]) == 2
            assert result["segments"][0]["words"][0]["word"] == "Hello"
            assert result["segments"][0]["words"][1]["word"] == "World"
            assert progress_records[-1] == 1.0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_get_words_in_range(self):
        t = Transcriber()
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "First segment",
                "words": [
                    {"word": "First", "start": 0.2, "end": 0.8},
                    {"word": "segment", "start": 1.0, "end": 1.9},
                ],
            },
            {
                "id": 1,
                "start": 2.5,
                "end": 5.0,
                "text": "Second segment here",
                "words": [
                    {"word": "Second", "start": 2.6, "end": 3.1},
                    {"word": "segment", "start": 3.2, "end": 3.8},
                    {"word": "here", "start": 4.0, "end": 4.7},
                ],
            },
        ]

        # Slicing from 0.0 to 1.5
        words = t.get_words_in_range(segments, 0.0, 1.5)
        extracted = [w["word"] for w in words]
        assert "First" in extracted
        assert "segment" in extracted
        assert "Second" not in extracted

        # Slicing empty segments
        assert t.get_words_in_range([], 0.0, 1.0) == []

        # Inverted start and end
        words_inverted = t.get_words_in_range(segments, 1.5, 0.0)
        assert len(words_inverted) == len(words)


class TestSentimentAnalyzer:
    def test_init_and_lazy_load(self):
        sa = SentimentAnalyzer()
        assert sa._vader is None
        # Accessing vader lazy-loads
        vader = sa.vader
        assert vader is not None
        assert sa._vader is not None

    def test_compute_hype_score_formula(self):
        sa = SentimentAnalyzer()
        # "THIS IS CRAZY!" -> 3 words, 3 caps (ratio 1.0 -> 0.2), 1 exclamation (density 1/3 -> 0.0333), 1 hype word ("crazy" -> 0.1)
        score = sa._compute_hype_score("THIS IS CRAZY!", compound=0.5, booster_mode="formula")
        assert score == pytest.approx(0.8333, abs=1e-3)
        assert 0.0 <= score <= 1.0

    def test_compute_hype_score_discrete(self):
        sa = SentimentAnalyzer()
        # "THIS IS CRAZY!" -> 3 caps (+0.3), 1 exclamation (+0.05), 1 hype word (+0.1), compound=0.5 -> 0.95
        score = sa._compute_hype_score("THIS IS CRAZY!", compound=0.5, booster_mode="discrete")
        assert score == pytest.approx(0.95, abs=1e-3)

    def test_analyze_segments(self):
        sa = SentimentAnalyzer()
        segments = [
            {"text": "Just having a calm talk today.", "start": 0.0, "end": 2.5},
            {"text": "OH MY GOD THAT WAS INSANE!!!", "start": 3.0, "end": 6.0},
            {"text": "let's go no way unbelievable!", "start": 6.5, "end": 9.0},
            {"text": "", "start": 9.5, "end": 10.0},
        ]

        analyzed = sa.analyze_segments(segments)
        for s in analyzed:
            assert "sentiment_score" in s
            assert "hype_level" in s
            assert 0.0 <= s["sentiment_score"] <= 1.0
            assert s["hype_level"] in ("low", "medium", "high", "extreme")

        assert analyzed[1]["sentiment_score"] > analyzed[0]["sentiment_score"]
        assert analyzed[1]["hype_level"] in ("high", "extreme")
        assert analyzed[0]["hype_level"] == "low"

    def test_get_score_at_time(self):
        sa = SentimentAnalyzer()
        segments = [
            {"start": 0.0, "end": 3.0, "sentiment_score": 0.2},
            {"start": 3.0, "end": 6.0, "sentiment_score": 0.9},
            {"start": 6.0, "end": 9.0, "sentiment_score": 0.5},
        ]

        assert sa.get_score_at_time(segments, 1.5) == 0.2
        assert sa.get_score_at_time(segments, 4.0) == 0.9
        assert sa.get_score_at_time(segments, 15.0, default=0.0) == 0.0

    def test_pipeline_analyze_method(self):
        sa = SentimentAnalyzer()
        data = {
            "segments": [
                {"start": 1.0, "end": 3.0, "text": "normal text"},
                {"start": 4.0, "end": 6.0, "text": "INSANE clutch let's go!!"},
            ]
        }
        res = sa.analyze(data)
        assert "timestamps" in res
        assert "sentiment_scores" in res
        assert "scores" in res
        assert len(res["sentiment_scores"]) == 2
        assert res["sentiment_scores"][1] > res["sentiment_scores"][0]
