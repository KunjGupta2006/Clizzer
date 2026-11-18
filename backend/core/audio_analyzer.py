"""
AUDIO ANALYSIS ENGINE for Clizzer.
Production-quality audio analysis using librosa, numpy, and scipy.signal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import librosa
import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


class AudioAnalysisError(Exception):
    """Raised when audio analysis fails during loading or feature extraction."""
    pass


class AudioAnalyzer:
    """
    Audio Analysis Engine for Clizzer.

    Extracts acoustic features (RMS energy, spectral centroid, percussive energy,
    and onsets), normalizes and combines them into a smoothed 'hype' timeline,
    and identifies peak excitement moments suitable for viral clip generation.
    """

    def __init__(self, sample_rate: int = 22050, hop_length: int = 512) -> None:
        """
        Initialize the AudioAnalyzer with sampling parameters.

        Args:
            sample_rate: Target sampling rate in Hz for audio loading and analysis.
            hop_length: Number of audio samples between successive analysis frames.

        Raises:
            ValueError: If sample_rate or hop_length are not positive integers.
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer, got {sample_rate}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be a positive integer, got {hop_length}")

        self.sample_rate: int = sample_rate
        self.hop_length: int = hop_length

    def analyze(
        self,
        audio_path: Union[str, Path],
        progress_callback: Union[Callable[[float, str], None], None] = None,
    ) -> Dict[str, Any]:
        """
        Run end-to-end audio analysis pipeline on the provided audio file.

        Steps:
            1. Load audio with librosa.load(audio_path, sr=self.sample_rate).
            2. Compute RMS energy, spectral centroid, percussive energy (via HPSS), and onsets.
            3. Normalize each feature to 0.0 - 1.0.
            4. Compute raw hype_timeline = 0.4*rms + 0.3*percussive + 0.3*centroid.
            5. Smooth with 2-second moving average (np.convolve).
            6. Find peaks with scipy.signal.find_peaks (minimum distance = 15 seconds).
            7. Return dictionary with timestamps, features, hype_timeline, peak_times, peak_scores.

        Args:
            audio_path: Path to the target audio or video file.
            progress_callback: Optional callable accepting (progress: float, stage: str),
                               where progress is between 0.0 and 1.0.

        Returns:
            Dictionary containing:
                - 'timestamps': np.ndarray of frame timestamps in seconds.
                - 'features': dict of individual normalized and raw features.
                - 'rms': np.ndarray of normalized RMS energy.
                - 'spectral_centroid': np.ndarray of normalized spectral centroid.
                - 'percussive': np.ndarray of normalized percussive energy.
                - 'onsets': np.ndarray of detected onset times in seconds.
                - 'hype_timeline': np.ndarray of smoothed hype scores.
                - 'raw_hype_timeline': np.ndarray of unsmoothed combined scores.
                - 'peak_times': np.ndarray of peak timestamps in seconds.
                - 'peak_scores': np.ndarray of hype scores at detected peaks.
                - 'duration': Total duration of audio in seconds.
                - 'sample_rate': Sampling rate used.
                - 'hop_length': Hop length used.

        Raises:
            FileNotFoundError: If audio_path does not exist on disk.
            ValueError: If audio is empty or audio_path is invalid.
            AudioAnalysisError: If librosa fails to decode or analyze the file.
        """
        def _report(progress: float, stage: str) -> None:
            if progress_callback is not None and callable(progress_callback):
                try:
                    progress_callback(progress, stage)
                except Exception as exc:
                    logger.warning("Error in progress_callback: %s", exc)

        if not audio_path:
            raise ValueError("audio_path cannot be empty.")

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

        # 1. Load audio
        _report(0.05, "Loading audio file")
        try:
            y, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        except Exception as exc:
            logger.error("Failed to load audio from %s: %s", path, exc, exc_info=True)
            raise AudioAnalysisError(f"Failed to load audio from '{path}': {exc}") from exc

        if y is None or y.size == 0:
            raise ValueError(f"Audio file '{path}' is empty or contains no valid samples.")

        duration = float(len(y) / sr)
        logger.info("Loaded audio %s (duration: %.2fs, sr: %d)", path.name, duration, sr)

        # 2. Compute features
        _report(0.20, "Computing RMS energy")
        try:
            rms_raw = self._compute_rms(y, self.hop_length)
        except Exception as exc:
            raise AudioAnalysisError(f"Error computing RMS: {exc}") from exc

        _report(0.40, "Computing spectral centroid")
        try:
            centroid_raw = self._compute_spectral_centroid(y, sr, self.hop_length)
        except Exception as exc:
            raise AudioAnalysisError(f"Error computing spectral centroid: {exc}") from exc

        _report(0.60, "Computing percussive energy via HPSS")
        try:
            percussive_raw = self._compute_percussive(y, self.hop_length)
        except Exception as exc:
            raise AudioAnalysisError(f"Error computing percussive energy: {exc}") from exc

        _report(0.75, "Detecting onsets")
        try:
            onsets = self._detect_onsets(y, sr)
        except Exception as exc:
            raise AudioAnalysisError(f"Error detecting onsets: {exc}") from exc

        # Align frame counts if any feature differs slightly by 1 frame
        n_frames = min(len(rms_raw), len(centroid_raw), len(percussive_raw))
        if n_frames == 0:
            raise AudioAnalysisError("Computed feature timelines contain zero frames.")

        rms_raw = rms_raw[:n_frames]
        centroid_raw = centroid_raw[:n_frames]
        percussive_raw = percussive_raw[:n_frames]

        timestamps = librosa.frames_to_time(
            np.arange(n_frames),
            sr=sr,
            hop_length=self.hop_length,
        )

        # 3. Normalize each to 0-1
        _report(0.85, "Normalizing features and constructing hype timeline")
        rms_norm = self._normalize(rms_raw)
        centroid_norm = self._normalize(centroid_raw)
        percussive_norm = self._normalize(percussive_raw)

        # 4. Create hype_timeline = 0.4*rms + 0.3*percussive + 0.3*centroid
        raw_hype = (
            0.4 * rms_norm
            + 0.3 * percussive_norm
            + 0.3 * centroid_norm
        )

        # 5. Smooth with 2-second moving average (np.convolve)
        _report(0.90, "Smoothing hype timeline")
        smoothed_hype = self._smooth(raw_hype, window_sec=2.0)

        # 6. Find peaks with scipy.signal.find_peaks, min distance = 15 seconds
        _report(0.95, "Finding hype peaks")
        peak_times, peak_scores = self._find_peaks(
            smoothed_hype,
            timestamps,
            min_distance_sec=15.0,
        )

        _report(1.0, "Audio analysis complete")

        features_dict: Dict[str, Any] = {
            "rms": rms_norm,
            "raw_rms": rms_raw,
            "spectral_centroid": centroid_norm,
            "raw_spectral_centroid": centroid_raw,
            "percussive": percussive_norm,
            "raw_percussive": percussive_raw,
            "onsets": onsets,
        }

        return {
            "timestamps": timestamps,
            "features": features_dict,
            "rms": rms_norm,
            "spectral_centroid": centroid_norm,
            "percussive": percussive_norm,
            "onsets": onsets,
            "hype_timeline": smoothed_hype,
            "raw_hype_timeline": raw_hype,
            "peak_times": peak_times,
            "peak_scores": peak_scores,
            "duration": duration,
            "sample_rate": sr,
            "hop_length": self.hop_length,
        }

    def _compute_rms(self, y: np.ndarray, hop_length: int | None = None) -> np.ndarray:
        """
        Compute root-mean-square (RMS) energy for each audio frame.

        Args:
            y: Audio time series (1D numpy array).
            hop_length: Number of samples between frames. Defaults to self.hop_length.

        Returns:
            1D numpy array of RMS energy per frame.
        """
        hl = hop_length if hop_length is not None else self.hop_length
        rms = librosa.feature.rms(y=y, hop_length=hl)
        return rms.squeeze(axis=0) if rms.ndim > 1 else rms

    def _compute_spectral_centroid(
        self,
        y: np.ndarray,
        sr: int | None = None,
        hop_length: int | None = None,
    ) -> np.ndarray:
        """
        Compute spectral centroid (center of mass of spectrum) for each frame.

        Args:
            y: Audio time series (1D numpy array).
            sr: Audio sampling rate. Defaults to self.sample_rate.
            hop_length: Number of samples between frames. Defaults to self.hop_length.

        Returns:
            1D numpy array of spectral centroid values per frame.
        """
        sr_val = sr if sr is not None else self.sample_rate
        hl = hop_length if hop_length is not None else self.hop_length
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr_val, hop_length=hl)
        return centroid.squeeze(axis=0) if centroid.ndim > 1 else centroid

    def _compute_percussive(
        self,
        y: np.ndarray,
        hop_length: int | None = None,
    ) -> np.ndarray:
        """
        Compute percussive energy via Harmonic-Percussive Source Separation (HPSS)
        followed by RMS energy of the separated percussive signal.

        Args:
            y: Audio time series (1D numpy array).
            hop_length: Number of samples between frames. Defaults to self.hop_length.

        Returns:
            1D numpy array of percussive RMS energy per frame.
        """
        hl = hop_length if hop_length is not None else self.hop_length
        _, y_percussive = librosa.effects.hpss(y)
        return self._compute_rms(y_percussive, hop_length=hl)

    def _detect_onsets(self, y: np.ndarray, sr: int | None = None) -> np.ndarray:
        """
        Detect note and transient onset events in seconds.

        Args:
            y: Audio time series (1D numpy array).
            sr: Audio sampling rate. Defaults to self.sample_rate.

        Returns:
            1D numpy array of detected onset timestamps in seconds.
        """
        sr_val = sr if sr is not None else self.sample_rate
        onset_times = librosa.onset.onset_detect(
            y=y,
            sr=sr_val,
            hop_length=self.hop_length,
            units="time",
        )
        return np.asarray(onset_times, dtype=np.float64)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization scaling values to [0.0, 1.0].
        Gracefully handles constant arrays, all-zero arrays, NaNs, and empty inputs.

        Args:
            arr: Input numpy array.

        Returns:
            1D numpy array with values scaled between 0.0 and 1.0.
        """
        arr_float = np.asarray(arr, dtype=np.float64)
        if arr_float.size == 0:
            return arr_float.copy()

        # Handle non-finite values if present
        arr_clean = np.nan_to_num(arr_float, nan=0.0, posinf=0.0, neginf=0.0)

        min_val = float(np.min(arr_clean))
        max_val = float(np.max(arr_clean))
        val_range = max_val - min_val

        # If constant or zero variance, return zeros of same shape
        if val_range <= 1e-12:
            return np.zeros_like(arr_clean, dtype=np.float64)

        return (arr_clean - min_val) / val_range

    def _find_peaks(
        self,
        hype_timeline: np.ndarray,
        timestamps: np.ndarray,
        min_distance_sec: float = 15.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify peak excitement points along the hype timeline using scipy.signal.find_peaks,
        enforcing a minimum temporal distance between successive peaks.

        Args:
            hype_timeline: 1D numpy array of smoothed hype scores.
            timestamps: 1D numpy array of corresponding frame timestamps.
            min_distance_sec: Minimum temporal distance in seconds between peaks.

        Returns:
            Tuple of (peak_times, peak_scores) as 1D numpy arrays.
        """
        timeline = np.asarray(hype_timeline, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64)

        if timeline.size == 0 or times.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # Calculate frame-to-time resolution
        if len(times) > 1:
            dt = float((times[-1] - times[0]) / max(1, len(times) - 1))
            distance_frames = max(1, int(round(min_distance_sec / dt)))
        else:
            fps = self.sample_rate / self.hop_length
            distance_frames = max(1, int(round(min_distance_sec * fps)))

        peak_indices, _ = scipy.signal.find_peaks(timeline, distance=distance_frames)

        if len(peak_indices) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        peak_times = times[peak_indices]
        peak_scores = timeline[peak_indices]
        return peak_times, peak_scores

    def _smooth(self, arr: np.ndarray, window_sec: float = 2.0) -> np.ndarray:
        """
        Smooth an array using a moving average computed via np.convolve.

        Args:
            arr: Input 1D numpy array.
            window_sec: Window length in seconds for the moving average.

        Returns:
            1D numpy array of smoothed values matching the length of arr.
        """
        arr_float = np.asarray(arr, dtype=np.float64)
        if arr_float.size == 0:
            return arr_float.copy()

        fps = self.sample_rate / self.hop_length
        window_frames = max(1, int(round(window_sec * fps)))

        if window_frames <= 1 or len(arr_float) <= 1:
            return arr_float.copy()

        kernel = np.ones(window_frames, dtype=np.float64) / float(window_frames)
        smoothed = np.convolve(arr_float, kernel, mode="same")
        return smoothed
