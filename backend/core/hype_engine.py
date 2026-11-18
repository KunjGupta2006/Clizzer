"""
Hype Engine Module for Clizzer.
Fuses multi-modal analysis signals (audio, motion, face emotion, sentiment, crowd events)
into a unified hype timeline and identifies the optimal highlight clip windows.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from scipy.signal import find_peaks


@dataclass
class ClipWindow:
    """Represents a candidate highlight clip window with constituent scores."""
    start_time: float
    end_time: float
    duration: float
    hype_score: float
    audio_score: float
    motion_score: float
    face_score: float
    sentiment_score: float
    peak_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


class HypeEngine:
    """
    Fuses 5 multi-modal analysis signals to find the best clip moments:
    - Audio energy / loudness
    - Visual motion dynamics
    - Facial emotion / intensity
    - Transcript sentiment
    - Crowd event / cheering reaction
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "audio": 0.30,
        "motion": 0.20,
        "face_emotion": 0.20,
        "sentiment": 0.15,
        "crowd_event": 0.15,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_clip_duration: float = 15.0,
        max_clip_duration: float = 60.0,
        num_clips: int = 4,
        time_resolution: float = 0.5,
    ):
        """
        Initialize HypeEngine with signal weights and clip boundary constraints.

        Args:
            weights: Dictionary of signal weights.
                     Default: audio=0.30, motion=0.20, face_emotion=0.20, sentiment=0.15, crowd_event=0.15
            min_clip_duration: Minimum duration of a highlight clip in seconds (default 15.0).
            max_clip_duration: Maximum duration of a highlight clip in seconds (default 60.0).
            num_clips: Default number of clips to extract (default 4).
            time_resolution: Time step interval for the unified timeline in seconds (default 0.5).
        """
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            # Map common aliases if provided
            alias_map = {
                "face": "face_emotion",
                "emotion": "face_emotion",
                "faces": "face_emotion",
                "crowd": "crowd_event",
                "events": "crowd_event",
                "cheer": "crowd_event",
            }
            for k, v in weights.items():
                target_k = alias_map.get(k, k)
                self.weights[target_k] = float(v)

        # Normalize weights so they sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.min_clip_duration = float(min_clip_duration)
        self.max_clip_duration = float(max_clip_duration)
        self.num_clips = int(num_clips)
        self.time_resolution = float(time_resolution)

        # Cache of the most recent timeline and interpolated signals
        self._last_time_points: Optional[np.ndarray] = None
        self._last_hype_scores: Optional[np.ndarray] = None
        self._last_signals: Dict[str, np.ndarray] = {}

    def compute_hype_timeline(
        self,
        audio_data: Any,
        visual_data: Any,
        face_data: Any,
        sentiment_data: Any,
        duration: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a unified hype timeline by interpolating each signal onto a common time grid,
        fusing them via weighted sum, and applying a 3-second moving average smoothing.

        Args:
            audio_data: Audio analysis data (dict, list, or array).
            visual_data: Visual motion analysis data (dict, list, or array).
            face_data: Face emotion analysis data (dict, list, or array).
            sentiment_data: Sentiment analysis data (dict, list, or array).
            duration: Total media duration in seconds.

        Returns:
            Tuple of (time_points array, hype_scores array).
        """
        if duration <= 0:
            empty = np.array([], dtype=float)
            return empty, empty

        # 1. Create unified timeline at time_resolution intervals
        # Ensure the timeline spans from 0.0 to duration inclusive
        num_points = max(1, int(np.ceil(duration / self.time_resolution)) + 1)
        time_points = np.linspace(0.0, duration, num_points)

        # 2. Extract and interpolate each of the 5 signals
        t_audio, v_audio = self._extract_times_and_values(
            audio_data, ("hype_timeline", "energy", "scores", "audio_scores", "rms", "volume", "loudness"), duration
        )
        t_motion, v_motion = self._extract_times_and_values(
            visual_data, ("motion_scores", "motion_timeline", "motion", "scores", "flow"), duration
        )
        t_face, v_face = self._extract_times_and_values(
            face_data, ("emotion_scores", "face_scores", "scores", "emotions"), duration
        )
        t_sentiment, v_sentiment = self._extract_times_and_values(
            sentiment_data, ("sentiment_scores", "scores", "polarity", "sentiment"), duration
        )

        # Extract crowd event signal (often present in audio_data or visual_data)
        t_crowd, v_crowd = self._extract_crowd_signal(audio_data, visual_data, duration, t_audio, v_audio)

        # Interpolate each signal to the target common time grid using numpy.interp
        audio_interp = self._interpolate_to_timeline(t_audio, v_audio, time_points)
        motion_interp = self._interpolate_to_timeline(t_motion, v_motion, time_points)
        face_interp = self._interpolate_to_timeline(t_face, v_face, time_points)
        sentiment_interp = self._interpolate_to_timeline(t_sentiment, v_sentiment, time_points)
        crowd_interp = self._interpolate_to_timeline(t_crowd, v_crowd, time_points)

        # Normalize individual signals to [0.0, 1.0]
        audio_norm = self._normalize_signal(audio_interp)
        motion_norm = self._normalize_signal(motion_interp)
        face_norm = self._normalize_signal(face_interp)
        sentiment_norm = self._normalize_signal(sentiment_interp)
        crowd_norm = self._normalize_signal(crowd_interp)

        # Cache individual normalized signals for clip window score computation
        self._last_time_points = time_points
        self._last_signals = {
            "audio": audio_norm,
            "motion": motion_norm,
            "face_emotion": face_norm,
            "sentiment": sentiment_norm,
            "crowd_event": crowd_norm,
        }

        # 3. Fuse: H(t) = weighted sum of all signals
        w_audio = self.weights.get("audio", 0.30)
        w_motion = self.weights.get("motion", 0.20)
        w_face = self.weights.get("face_emotion", 0.20)
        w_sentiment = self.weights.get("sentiment", 0.15)
        w_crowd = self.weights.get("crowd_event", 0.15)

        raw_hype = (
            w_audio * audio_norm
            + w_motion * motion_norm
            + w_face * face_norm
            + w_sentiment * sentiment_norm
            + w_crowd * crowd_norm
        )

        # 4. Smooth with 3-second moving average
        window_size = max(1, int(round(3.0 / self.time_resolution)))
        kernel = np.ones(window_size)
        smoothed_hype = np.convolve(raw_hype, kernel, mode="same")
        # Normalize edges to prevent zero-padding edge attenuation
        norm_factor = np.convolve(np.ones_like(raw_hype), kernel, mode="same")
        smoothed_hype = smoothed_hype / np.maximum(norm_factor, 1e-6)
        smoothed_hype = np.clip(smoothed_hype, 0.0, 1.0)

        self._last_hype_scores = smoothed_hype
        return time_points, smoothed_hype

    def find_best_clips(
        self,
        hype_timeline_tuple: Tuple[np.ndarray, np.ndarray],
        scene_boundaries: Optional[Union[List[Any], np.ndarray]] = None,
        num_clips: Optional[int] = None,
    ) -> List[ClipWindow]:
        """
        Find the best highlight clips from the hype timeline:
        1. Find peaks with scipy.signal.find_peaks, min distance = min_clip_duration.
        2. Sort peaks by score descending.
        3. For each peak: expand window, snap to scene boundaries, check no overlap.
        4. Return list of ClipWindow dataclass instances.

        Args:
            hype_timeline_tuple: Tuple of (time_points, hype_scores) from compute_hype_timeline.
            scene_boundaries: Optional list or array of scene cut timestamps (numbers or dicts).
            num_clips: Number of clips to return (defaults to self.num_clips).

        Returns:
            List of ClipWindow dataclass instances sorted by hype_score descending.
        """
        time_points, hype_scores = hype_timeline_tuple
        if time_points is None or hype_scores is None or len(time_points) == 0:
            return []

        k_clips = num_clips if num_clips is not None else self.num_clips
        if k_clips <= 0:
            return []

        total_duration = float(time_points[-1])
        dt = float(np.median(np.diff(time_points))) if len(time_points) > 1 else self.time_resolution
        if dt <= 0:
            dt = self.time_resolution

        # 1. Find peaks with scipy.signal.find_peaks, min distance = min_clip_duration
        min_distance_samples = max(1, int(round(self.min_clip_duration / dt)))
        detected_peaks, _ = find_peaks(hype_scores, distance=min_distance_samples)

        # Collect candidate peak indices
        candidate_peak_indices: List[int] = list(detected_peaks)

        # Fallback if find_peaks returned fewer peaks than requested
        if len(candidate_peak_indices) < k_clips:
            # Rank all time indices by score and greedily pick non-conflicting peaks
            sorted_indices = np.argsort(-hype_scores)
            for idx in sorted_indices:
                idx = int(idx)
                if not any(abs(idx - p) < min_distance_samples for p in candidate_peak_indices):
                    candidate_peak_indices.append(idx)
                    if len(candidate_peak_indices) >= k_clips * 2:
                        break

        # If still empty (e.g., flat or tiny timeline), use center/equidistant points
        if not candidate_peak_indices:
            candidate_peak_indices = [int(len(time_points) // 2)]

        # 2. Sort candidate peaks by score descending
        candidate_peak_indices.sort(key=lambda idx: float(hype_scores[idx]), reverse=True)

        # Parse scene boundaries (handles numbers, floats, or scene dicts with start_time/end_time)
        boundaries_arr: np.ndarray = self._parse_scene_boundaries(scene_boundaries)

        selected_clips: List[ClipWindow] = []

        # 3. For each peak: expand window, snap to scene boundaries, check no overlap
        for peak_idx in candidate_peak_indices:
            peak_time = float(time_points[peak_idx])

            # Expand window around peak
            start_time, end_time = self._expand_window(
                peak_time=peak_time,
                scores=hype_scores,
                times=time_points,
                min_dur=self.min_clip_duration,
                max_dur=self.max_clip_duration,
            )

            # Snap to scene boundaries
            start_time, end_time = self._snap_to_scene_boundaries(
                start_time=start_time,
                end_time=end_time,
                scene_boundaries=boundaries_arr,
                total_duration=total_duration,
            )

            # Check no overlap with previously selected clips
            if self._has_overlap(start_time, end_time, selected_clips):
                continue

            # Compute constituent scores across the clip window
            mask = (time_points >= start_time) & (time_points <= end_time)
            if np.any(mask):
                w_hype = float(np.mean(hype_scores[mask]))
                w_audio = self._get_window_score("audio", mask, w_hype)
                w_motion = self._get_window_score("motion", mask, w_hype)
                w_face = self._get_window_score("face_emotion", mask, w_hype)
                w_sentiment = self._get_window_score("sentiment", mask, w_hype)
            else:
                w_hype = float(hype_scores[peak_idx])
                w_audio = w_motion = w_face = w_sentiment = w_hype

            duration = round(end_time - start_time, 2)
            clip_window = ClipWindow(
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                duration=duration,
                hype_score=round(w_hype, 4),
                audio_score=round(w_audio, 4),
                motion_score=round(w_motion, 4),
                face_score=round(w_face, 4),
                sentiment_score=round(w_sentiment, 4),
                peak_timestamp=round(peak_time, 2),
            )
            selected_clips.append(clip_window)

            if len(selected_clips) >= k_clips:
                break

        return selected_clips

    def _interpolate_to_timeline(
        self,
        source_times: Any,
        source_values: Any,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate an analysis signal to the common target timeline using numpy.interp.

        Args:
            source_times: Array or list of timestamps for the source signal.
            source_values: Array or list of values for the source signal.
            target_times: 1D array of common target timeline timestamps.

        Returns:
            1D numpy array of interpolated values matching target_times.
        """
        if target_times is None or len(target_times) == 0:
            return np.array([], dtype=float)

        if source_times is None or source_values is None:
            return np.zeros_like(target_times, dtype=float)

        s_times = np.asarray(source_times, dtype=float)
        s_values = np.asarray(source_values, dtype=float)

        if len(s_times) == 0 or len(s_values) == 0:
            return np.zeros_like(target_times, dtype=float)

        if len(s_times) == 1:
            val = float(s_values[0])
            return np.full_like(target_times, val, dtype=float)

        # Sort source timestamps for np.interp
        sort_order = np.argsort(s_times)
        s_times_sorted = s_times[sort_order]
        s_values_sorted = s_values[sort_order]

        # Deduplicate identical timestamps by averaging their values
        unique_times, unique_indices = np.unique(s_times_sorted, return_index=True)
        if len(unique_times) < len(s_times_sorted):
            unique_values = np.array([
                np.mean(s_values_sorted[s_times_sorted == t])
                for t in unique_times
            ], dtype=float)
        else:
            unique_times = s_times_sorted
            unique_values = s_values_sorted

        # Interpolate using np.interp with boundary constant extrapolation
        interpolated = np.interp(
            target_times,
            unique_times,
            unique_values,
            left=float(unique_values[0]),
            right=float(unique_values[-1]),
        )
        return interpolated

    def _expand_window(
        self,
        peak_time: float,
        scores: np.ndarray,
        times: np.ndarray,
        min_dur: float,
        max_dur: float,
    ) -> Tuple[float, float]:
        """
        Expand a clip window around a hype peak:
        1. Greedily expand towards higher score side until min_dur is met.
        2. Continue expanding if adjacent scores remain elevated up to max_dur.
        3. Clamp within timeline boundaries.

        Args:
            peak_time: Timestamp of the peak moment.
            scores: Hype scores array.
            times: Target timeline timestamps array.
            min_dur: Minimum clip duration in seconds.
            max_dur: Maximum clip duration in seconds.

        Returns:
            Tuple of (start_time, end_time).
        """
        if len(times) == 0:
            return 0.0, 0.0

        total_span = float(times[-1] - times[0])
        effective_min = min(min_dur, total_span)
        effective_max = min(max_dur, total_span)

        # Locate peak index
        peak_idx = int(np.argmin(np.abs(times - peak_time)))
        peak_score = float(scores[peak_idx])

        left_idx = peak_idx
        right_idx = peak_idx
        n = len(times)

        # Phase 1: Expand until minimum duration is reached
        while (times[right_idx] - times[left_idx]) < effective_min:
            can_expand_left = left_idx > 0
            can_expand_right = right_idx < n - 1

            if not can_expand_left and not can_expand_right:
                break
            elif can_expand_left and not can_expand_right:
                left_idx -= 1
            elif can_expand_right and not can_expand_left:
                right_idx += 1
            else:
                # Both directions available: expand in the direction of higher hype score
                score_left = scores[left_idx - 1]
                score_right = scores[right_idx + 1]
                if score_left >= score_right:
                    left_idx -= 1
                else:
                    right_idx += 1

        # Phase 2: Optionally expand further up to max_dur if hype remains elevated
        threshold = max(0.25, 0.50 * peak_score)
        while (times[right_idx] - times[left_idx]) < effective_max:
            can_expand_left = left_idx > 0 and scores[left_idx - 1] >= threshold
            can_expand_right = right_idx < n - 1 and scores[right_idx + 1] >= threshold

            if not can_expand_left and not can_expand_right:
                break
            elif can_expand_left and not can_expand_right:
                left_idx -= 1
            elif can_expand_right and not can_expand_left:
                right_idx += 1
            else:
                score_left = scores[left_idx - 1]
                score_right = scores[right_idx + 1]
                if score_left >= score_right:
                    left_idx -= 1
                else:
                    right_idx += 1

        start_time = float(times[left_idx])
        end_time = float(times[right_idx])

        # Enforce minimum duration if bounded by edge
        if (end_time - start_time) < effective_min:
            if left_idx == 0:
                end_time = min(start_time + effective_min, float(times[-1]))
            elif right_idx == n - 1:
                start_time = max(0.0, end_time - effective_min)

        return start_time, end_time

    def _snap_to_scene_boundaries(
        self,
        start_time: float,
        end_time: float,
        scene_boundaries: np.ndarray,
        total_duration: float,
        snap_tolerance: float = 2.5,
    ) -> Tuple[float, float]:
        """
        Snap clip boundaries to nearby visual scene cuts if within snap_tolerance,
        while maintaining duration constraints [min_clip_duration, max_clip_duration].
        """
        if len(scene_boundaries) == 0:
            return start_time, end_time

        snapped_start = start_time
        snapped_end = end_time

        # Find closest boundary for start_time
        start_diffs = np.abs(scene_boundaries - start_time)
        min_start_idx = int(np.argmin(start_diffs))
        if start_diffs[min_start_idx] <= snap_tolerance:
            candidate_start = float(scene_boundaries[min_start_idx])
            if 0.0 <= candidate_start < end_time:
                snapped_start = candidate_start

        # Find closest boundary for end_time
        end_diffs = np.abs(scene_boundaries - end_time)
        min_end_idx = int(np.argmin(end_diffs))
        if end_diffs[min_end_idx] <= snap_tolerance:
            candidate_end = float(scene_boundaries[min_end_idx])
            if candidate_end > snapped_start and candidate_end <= total_duration:
                snapped_end = candidate_end

        # Verify duration constraints
        candidate_dur = snapped_end - snapped_start
        min_limit = min(self.min_clip_duration, total_duration)
        if candidate_dur < min_limit or candidate_dur > self.max_clip_duration:
            return start_time, end_time

        return snapped_start, snapped_end

    def _has_overlap(self, start: float, end: float, existing_clips: List[ClipWindow]) -> bool:
        """Check if candidate interval [start, end] overlaps with any selected clip."""
        for clip in existing_clips:
            if max(start, clip.start_time) < min(end, clip.end_time):
                return True
        return False

    def _get_window_score(self, signal_name: str, mask: np.ndarray, fallback: float) -> float:
        """Compute average score for a named signal within a masked window."""
        if signal_name in self._last_signals and len(self._last_signals[signal_name]) == len(mask):
            sig = self._last_signals[signal_name][mask]
            if len(sig) > 0:
                return float(np.mean(sig))
        return fallback

    def _parse_scene_boundaries(self, scene_boundaries: Any) -> np.ndarray:
        """Parse scene boundaries from numbers, arrays, or scene dicts."""
        if scene_boundaries is None or len(scene_boundaries) == 0:
            return np.array([], dtype=float)

        extracted: List[float] = []
        for b in scene_boundaries:
            if isinstance(b, dict):
                if "start_time" in b:
                    extracted.append(float(b["start_time"]))
                if "end_time" in b:
                    extracted.append(float(b["end_time"]))
            elif isinstance(b, (int, float)):
                extracted.append(float(b))

        if not extracted:
            return np.array([], dtype=float)

        return np.unique(np.sort(np.asarray(extracted, dtype=float)))

    def _extract_times_and_values(
        self,
        data: Any,
        candidate_keys: Tuple[str, ...],
        duration: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flexible extractor to extract timestamps and score values from multiple data shapes."""
        if data is None:
            return np.array([], dtype=float), np.array([], dtype=float)

        # Case 1: Dictionary format
        if isinstance(data, dict):
            # Check for visual_analyzer motion_timeline: [{"timestamp": t, "motion_magnitude": v}, ...]
            if "motion_timeline" in data and isinstance(data["motion_timeline"], list):
                mt_times = []
                mt_vals = []
                for item in data["motion_timeline"]:
                    if isinstance(item, dict):
                        mt_times.append(float(item.get("timestamp", 0.0)))
                        mt_vals.append(float(item.get("motion_magnitude", 0.0)))
                if mt_times:
                    return np.asarray(mt_times, dtype=float), np.asarray(mt_vals, dtype=float)

            # Check for face_analyzer emotion_timeline: [{"timestamp": t, "intensity": v}, ...]
            if "emotion_timeline" in data and isinstance(data["emotion_timeline"], list):
                em_times = []
                em_vals = []
                for item in data["emotion_timeline"]:
                    if isinstance(item, dict):
                        em_times.append(float(item.get("timestamp", 0.0)))
                        val = item.get("intensity", item.get("emotion_intensity", item.get("smile_ratio", item.get("score", 0.5))))
                        em_vals.append(float(val))
                if em_times:
                    return np.asarray(em_times, dtype=float), np.asarray(em_vals, dtype=float)

            # Check for face_frames or primary_face_track
            for track_k in ("primary_face_track", "face_frames"):
                if track_k in data and isinstance(data[track_k], list):
                    fc_times = []
                    fc_vals = []
                    for item in data[track_k]:
                        if hasattr(item, "timestamp"):
                            fc_times.append(float(item.timestamp))
                            conf = getattr(item, "confidence", getattr(item, "smile_ratio", 0.5))
                            fc_vals.append(float(conf))
                        elif isinstance(item, dict):
                            fc_times.append(float(item.get("timestamp", 0.0)))
                            fc_vals.append(float(item.get("confidence", item.get("smile_ratio", item.get("score", 0.5)))))
                    if fc_times:
                        return np.asarray(fc_times, dtype=float), np.asarray(fc_vals, dtype=float)

            times = None
            values = None

            # Look for time keys
            for tk in ("timestamps", "times", "time", "t", "timestamp"):
                if tk in data and data[tk] is not None:
                    try:
                        times = np.asarray(data[tk], dtype=float)
                        break
                    except (ValueError, TypeError):
                        pass

            # Look for value keys
            for vk in candidate_keys + ("scores", "values", "score", "value", "data"):
                if vk in data and data[vk] is not None:
                    try:
                        values = np.asarray(data[vk], dtype=float)
                        break
                    except (ValueError, TypeError):
                        pass

            # Handle segment lists inside dictionary
            if values is None and "segments" in data and isinstance(data["segments"], list):
                segs = data["segments"]
                seg_times = []
                seg_vals = []
                for s in segs:
                    if isinstance(s, dict):
                        st = s.get("start", 0.0)
                        et = s.get("end", st)
                        mid = (st + et) / 2.0
                        sc = s.get("score", s.get("sentiment_score", s.get("intensity", 0.5)))
                        seg_times.append(mid)
                        seg_vals.append(sc)
                if seg_times:
                    return np.asarray(seg_times, dtype=float), np.asarray(seg_vals, dtype=float)

            if values is not None and len(values) > 0:
                if times is None or len(times) != len(values):
                    times = np.linspace(0.0, max(duration, 1.0), len(values))
                return times, values

            # If dict was provided but no valid timeline extracted, return empty safely
            return np.array([], dtype=float), np.array([], dtype=float)

        # Case 2: List of segment dicts e.g. [{"start": 0, "end": 2, "score": 0.8}, ...]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            seg_times = []
            seg_vals = []
            for s in data:
                st = s.get("start", s.get("time", s.get("timestamp", 0.0)))
                et = s.get("end", st)
                mid = (st + et) / 2.0
                sc = s.get("score", s.get("sentiment_score", s.get("motion_magnitude", s.get("value", 0.5))))
                seg_times.append(mid)
                seg_vals.append(sc)
            return np.asarray(seg_times, dtype=float), np.asarray(seg_vals, dtype=float)

        # Case 3: 1D array or list of values
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1 and len(arr) > 0:
            times = np.linspace(0.0, max(duration, 1.0), len(arr))
            return times, arr
        elif arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) > 0:
            return arr[:, 0], arr[:, 1]

        return np.array([], dtype=float), np.array([], dtype=float)

    def _extract_crowd_signal(
        self,
        audio_data: Any,
        visual_data: Any,
        duration: float,
        t_audio: np.ndarray,
        v_audio: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract crowd event / cheer signal or synthesize from audio transients."""
        # 1. Check audio_data for crowd keys
        if isinstance(audio_data, dict):
            for k in ("crowd_event", "crowd_scores", "crowd_events", "cheer_scores", "cheer", "percussive"):
                if k in audio_data and audio_data[k] is not None:
                    t, v = self._extract_times_and_values(audio_data[k], ("scores", "values"), duration)
                    if len(v) > 0:
                        return t, v

        # 2. Check visual_data for crowd reaction keys
        if isinstance(visual_data, dict):
            for k in ("crowd_event", "crowd_scores", "crowd"):
                if k in visual_data and visual_data[k] is not None:
                    t, v = self._extract_times_and_values(visual_data[k], ("scores", "values"), duration)
                    if len(v) > 0:
                        return t, v

        # 3. Derive crowd reaction signal from audio energy surges (transients)
        if len(v_audio) > 1:
            diff = np.diff(v_audio, prepend=v_audio[0])
            transients = np.maximum(0.0, diff)
            return t_audio, transients

        return np.array([], dtype=float), np.array([], dtype=float)

    def _normalize_signal(self, values: np.ndarray) -> np.ndarray:
        """Min-max normalize a signal to [0.0, 1.0]."""
        if len(values) == 0:
            return values
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if v_max > v_min:
            return (values - v_min) / (v_max - v_min)
        elif v_max > 0:
            return np.clip(values / v_max, 0.0, 1.0)
        return np.zeros_like(values, dtype=float)
