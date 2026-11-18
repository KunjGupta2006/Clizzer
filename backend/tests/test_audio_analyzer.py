"""Unit tests for Clizzer AudioAnalyzer."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf

from backend.core.audio_analyzer import AudioAnalyzer, AudioAnalysisError


def test_init_validation():
    """Verify initialization parameters and validation."""
    analyzer = AudioAnalyzer(sample_rate=22050, hop_length=512)
    assert analyzer.sample_rate == 22050
    assert analyzer.hop_length == 512

    with pytest.raises(ValueError):
        AudioAnalyzer(sample_rate=0)

    with pytest.raises(ValueError):
        AudioAnalyzer(hop_length=-1)


def test_normalize():
    """Verify min-max normalization and edge cases."""
    analyzer = AudioAnalyzer()

    # Empty array
    empty = analyzer._normalize(np.array([]))
    assert empty.size == 0

    # Constant array (zero range)
    const = analyzer._normalize(np.ones(10) * 42.0)
    assert np.all(const == 0.0)

    # Standard array
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    norm = analyzer._normalize(arr)
    assert np.isclose(norm[0], 0.0)
    assert np.isclose(norm[-1], 1.0)
    assert np.isclose(norm[2], 0.5)

    # Array with NaNs / Infs
    with_nan = np.array([np.nan, 2.0, np.inf, 4.0])
    norm_nan = analyzer._normalize(with_nan)
    assert not np.isnan(norm_nan).any()
    assert not np.isinf(norm_nan).any()


def test_smooth():
    """Verify moving average smoothing with np.convolve."""
    analyzer = AudioAnalyzer(sample_rate=1000, hop_length=10)
    # fps = 100
    # window_sec = 0.1 -> window_frames = 10
    arr = np.ones(50)
    smoothed = analyzer._smooth(arr, window_sec=0.1)
    assert len(smoothed) == len(arr)

    # Empty input
    assert analyzer._smooth(np.array([])).size == 0


def test_find_peaks():
    """Verify peak detection with minimum distance enforcement."""
    analyzer = AudioAnalyzer(sample_rate=1000, hop_length=10)
    # 40 seconds duration, dt = 0.1s
    times = np.linspace(0, 40, 401)
    timeline = np.zeros_like(times)

    # Place peaks
    timeline[50] = 0.95   # at 5.0s
    timeline[100] = 0.80  # at 10.0s (only 5s away, should be rejected by min_distance=15s)
    timeline[260] = 0.99  # at 26.0s (21s away from 5s, should be accepted)

    peak_times, peak_scores = analyzer._find_peaks(timeline, times, min_distance_sec=15.0)
    assert len(peak_times) == 2
    assert np.isclose(peak_times[0], 5.0, atol=0.2)
    assert np.isclose(peak_times[1], 26.0, atol=0.2)


def test_analyze_file_not_found():
    """Verify FileNotFoundError on non-existent file."""
    analyzer = AudioAnalyzer()
    with pytest.raises(FileNotFoundError):
        analyzer.analyze("non_existent_audio_file.wav")


def test_analyze_empty_path():
    """Verify ValueError on empty path."""
    analyzer = AudioAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze("")


def test_full_pipeline_with_audio():
    """Verify full end-to-end analysis on synthesized audio."""
    sr = 22050
    duration = 20.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 300 * t)

    # Add bursts
    audio[int(4 * sr):int(6 * sr)] += 0.8 * np.sin(2 * np.pi * 600 * t[:int(2 * sr)])
    audio[int(16 * sr):int(18 * sr)] += 0.9 * np.sin(2 * np.pi * 900 * t[:int(2 * sr)])

    progress_events = []

    def on_progress(pct: float, stage: str):
        progress_events.append((pct, stage))

    analyzer = AudioAnalyzer(sample_rate=sr, hop_length=512)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio, sr)

    try:
        results = analyzer.analyze(temp_path, progress_callback=on_progress)

        assert "timestamps" in results
        assert "features" in results
        assert "rms" in results
        assert "spectral_centroid" in results
        assert "percussive" in results
        assert "onsets" in results
        assert "hype_timeline" in results
        assert "peak_times" in results
        assert "peak_scores" in results
        assert "duration" in results
        assert np.isclose(results["duration"], duration, atol=0.1)

        # Check features structure
        feat = results["features"]
        assert "rms" in feat
        assert "spectral_centroid" in feat
        assert "percussive" in feat
        assert "onsets" in feat

        # Check shapes
        n_frames = len(results["timestamps"])
        assert len(results["hype_timeline"]) == n_frames
        assert len(results["rms"]) == n_frames
        assert len(results["spectral_centroid"]) == n_frames
        assert len(results["percussive"]) == n_frames

        # Check hype values are within valid range
        assert np.all(results["hype_timeline"] >= 0.0)

        # Progress callback was called
        assert len(progress_events) >= 5
        assert progress_events[-1][0] == 1.0

        # Peaks found
        assert len(results["peak_times"]) >= 1
    finally:
        Path(temp_path).unlink(missing_ok=True)
