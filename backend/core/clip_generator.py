"""Video clip generator module for Clizzer backend.

Cuts highlight clip segments using FFmpeg, reframes to 9:16 vertical video
centered on detected faces, burns subtitles, and extracts middle-frame thumbnails.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from backend.config import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)


class ClipGenerator:
    """Generates extracted highlight video clips with optional 9:16 reframing,

    subtitle burning, and middle-frame thumbnail generation.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        output_format: str = "mp4",
        ffmpeg_cmd: str | None = None,
    ) -> None:
        """Set up output directory, create if not exists.

        Args:
            output_dir: Target directory path for storing generated clips and thumbnails.
                        Defaults to configured OUTPUT_DIR or 'output'.
            output_format: Container format extension for output video (default 'mp4').
            ffmpeg_cmd: Optional explicit path to ffmpeg executable.
        """
        if output_dir is None:
            if config is not None and hasattr(config, "OUTPUT_DIR"):
                self.output_dir = Path(config.OUTPUT_DIR)
            else:
                self.output_dir = Path("output")
        else:
            self.output_dir = Path(output_dir)

        # Ensure target directory exists on disk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format.lstrip(".").lower()

        # Locate ffmpeg executable
        if ffmpeg_cmd:
            self.ffmpeg_cmd = ffmpeg_cmd
        else:
            resolved = shutil.which("ffmpeg")
            if resolved:
                self.ffmpeg_cmd = resolved
            else:
                try:
                    import imageio_ffmpeg
                    self.ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    self.ffmpeg_cmd = "ffmpeg"

    def generate_clips(
        self,
        video_path: str | Path,
        clip_windows: list[dict] | list[Any],
        face_track: Any = None,
        subtitle_paths: str | Path | list[str | Path] | dict[int, str | Path] | None = None,
        progress_callback: Callable[[float, str], None] | Callable[[float], None] | None = None,
        reframe: bool = False,
    ) -> list[dict]:
        """For each clip window: cut with FFmpeg, optionally reframe 9:16, optionally burn subtitles.

        Generate thumbnail from middle frame.
        Return list of dicts: clip_path, thumbnail_path, start, end, duration, hype_score.

        Args:
            video_path: Path to the source input video.
            clip_windows: Sequence of clip window dictionaries or objects with 'start',
                          'end', and optional 'hype_score' / 'score'.
            face_track: Optional face tracking information (coordinates, dict, sequence,
                        or lookup callback) used to center 9:16 reframing.
            subtitle_paths: Optional path(s) to ASS subtitle files to burn into the clips.
                            Can be a single path, a list per clip window, or a dict.
            progress_callback: Optional progress notification callable accepting (progress, message)
                               or (progress).
            reframe: Explicit flag to enable 9:16 crop reframing (also auto-enabled if face_track
                     is provided or specified per window).

        Returns:
            list[dict]: List of dictionaries containing:
                - clip_path (str)
                - thumbnail_path (str)
                - start (float)
                - end (float)
                - duration (float)
                - hype_score (float)
        """
        source_path = Path(video_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source video file not found: {source_path}")

        if not clip_windows:
            logger.warning("generate_clips received an empty clip_windows list.")
            return []

        results: list[dict] = []
        total = len(clip_windows)

        for idx, window in enumerate(clip_windows):
            start, end, hype_score, window_reframe = self._parse_window(window)
            duration = round(max(0.0, end - start), 2)
            if duration <= 0:
                logger.warning("Clip window index %d has non-positive duration (start=%s, end=%s). Skipping.", idx, start, end)
                continue

            should_reframe = window_reframe if window_reframe is not None else (reframe or (face_track is not None))
            face_center = self._resolve_face_center(face_track, idx, start, end)
            sub_path = self._resolve_subtitle_path(subtitle_paths, idx)

            clip_stem = f"clip_{idx + 1}_{int(start)}_{int(end)}"
            final_clip_path = self.output_dir / f"{clip_stem}.{self.output_format}"
            thumb_path = self.output_dir / f"thumb_{idx + 1}_{int(start)}_{int(end)}.jpg"

            self._notify_progress(progress_callback, idx / total, f"Cutting clip {idx + 1}/{total}...")

            if sub_path:
                # When burning subtitles, cut video to a temporary intermediate file first
                temp_cut_path = self.output_dir / f"{clip_stem}_temp.{self.output_format}"
                try:
                    self._cut_clip(
                        video_path=source_path,
                        start=start,
                        end=end,
                        output_path=temp_cut_path,
                        reframe=should_reframe,
                        face_center=face_center,
                    )

                    self._notify_progress(progress_callback, (idx + 0.5) / total, f"Burning subtitles on clip {idx + 1}/{total}...")
                    self._burn_subtitles(
                        video_path=temp_cut_path,
                        subtitle_path=sub_path,
                        output_path=final_clip_path,
                    )
                finally:
                    if temp_cut_path.exists():
                        try:
                            temp_cut_path.unlink()
                        except OSError as err:
                            logger.debug("Could not remove intermediate cut file %s: %s", temp_cut_path, err)
            else:
                self._cut_clip(
                    video_path=source_path,
                    start=start,
                    end=end,
                    output_path=final_clip_path,
                    reframe=should_reframe,
                    face_center=face_center,
                )

            # Generate thumbnail from middle frame
            mid_timestamp = round(start + (duration / 2.0), 3)
            try:
                self._generate_thumbnail(source_path, mid_timestamp, thumb_path)
            except Exception as thumb_err:
                logger.warning(
                    "Thumbnail generation from source at %.3fs failed (%s); trying fallback from cut clip.",
                    mid_timestamp, thumb_err
                )
                try:
                    self._generate_thumbnail(final_clip_path, round(duration / 2.0, 3), thumb_path)
                except Exception as fb_err:
                    logger.error("Fallback thumbnail generation failed: %s", fb_err)

            results.append({
                "clip_path": str(final_clip_path),
                "thumbnail_path": str(thumb_path),
                "start": start,
                "end": end,
                "duration": duration,
                "hype_score": hype_score,
            })

            self._notify_progress(progress_callback, (idx + 1) / total, f"Completed clip {idx + 1}/{total}")

        return results

    def generate_clip(
        self,
        video_path: str | Path,
        start: float,
        end: float,
        output_path: str | Path,
        reframe: bool = False,
        face_center: Any = None,
        **kwargs: Any,
    ) -> str:
        """Alias to _cut_clip for interface consistency."""
        return self._cut_clip(
            video_path=video_path,
            start=start,
            end=end,
            output_path=output_path,
            reframe=reframe,
            face_center=face_center,
        )

    def _cut_clip(
        self,
        video_path: str | Path,
        start: float,
        end: float,
        output_path: str | Path,
        reframe: bool = False,
        face_center: Any = None,
    ) -> str:
        """Cut a clip window from the video.

        Basic: ffmpeg -ss start -to end -i video -c copy output
        With reframe: add crop filter centered on face
        Use subprocess.run with stderr capture for error handling.

        Args:
            video_path: Source input video path.
            start: Start timestamp in seconds.
            end: End timestamp in seconds.
            output_path: Destination path for the extracted clip.
            reframe: Whether to crop into 9:16 vertical video format.
            face_center: Center coordinate of tracked face (normalized 0.0-1.0 or pixels).

        Returns:
            str: Path to the generated output clip file.

        Raises:
            RuntimeError: If FFmpeg execution returns a non-zero exit code.
        """
        video_path = str(Path(video_path).resolve())
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not reframe:
            # Basic: ffmpeg -ss start -to end -i video -c copy output
            cmd = [
                self.ffmpeg_cmd,
                "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-c", "copy",
                str(output_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    "FFmpeg stream copy cut failed (code %d): %s. Retrying with encode.",
                    result.returncode, result.stderr[-200:].strip()
                )
                # Stream copy can fail on non-keyframe boundaries or container quirks.
                # Fall back to high-speed re-encode to ensure reliability.
                fallback_cmd = [
                    self.ffmpeg_cmd,
                    "-y",
                    "-ss", str(start),
                    "-to", str(end),
                    "-i", video_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "22",
                    "-c:a", "aac",
                    str(output_path),
                ]
                fb_result = subprocess.run(
                    fallback_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if fb_result.returncode != 0:
                    raise RuntimeError(
                        f"FFmpeg cut clip failed with code {fb_result.returncode}.\n"
                        f"Command: {' '.join(fallback_cmd)}\n"
                        f"Error output:\n{fb_result.stderr}"
                    )
        else:
            # With reframe: add crop filter centered on face
            crop_filter = self._compute_9_16_crop_filter(video_path, face_center)
            cmd = [
                self.ffmpeg_cmd,
                "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-vf", crop_filter,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-c:a", "aac",
                str(output_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg reframing cut failed with code {result.returncode}.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Error output:\n{result.stderr}"
                )

        return str(output_path)

    def _burn_subtitles(
        self,
        video_path: str | Path,
        subtitle_path: str | Path,
        output_path: str | Path,
    ) -> str:
        """Burn subtitles into video using FFmpeg ass filter.

        ffmpeg -i video -vf ass=subtitle_path output

        Args:
            video_path: Path to the input video.
            subtitle_path: Path to the ASS subtitle file.
            output_path: Path for the burned subtitle video.

        Returns:
            str: Path to the burned subtitle video file.

        Raises:
            FileNotFoundError: If the subtitle file does not exist.
            RuntimeError: If FFmpeg execution returns a non-zero exit code.
        """
        video_path = str(Path(video_path).resolve())
        sub_file = Path(subtitle_path).resolve()
        if not sub_file.exists():
            raise FileNotFoundError(f"Subtitle file not found: {sub_file}")

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Escape subtitle path for FFmpeg filter parsing (Windows drive letters & slashes)
        escaped_sub = str(sub_file).replace("\\", "/").replace(":", r"\:")
        filter_arg = f"ass='{escaped_sub}'"

        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-i", video_path,
            "-vf", filter_arg,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "copy",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg burn subtitles failed with code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error output:\n{result.stderr}"
            )

        return str(output_path)

    def _generate_thumbnail(
        self,
        video_path: str | Path,
        timestamp: float,
        output_path: str | Path,
    ) -> str:
        """Extract a single frame thumbnail at the specified timestamp.

        ffmpeg -ss timestamp -i video -vframes 1 -q:v 2 output.jpg

        Args:
            video_path: Path to the source video.
            timestamp: Timestamp in seconds for thumbnail capture.
            output_path: Destination path for the saved JPEG thumbnail image.

        Returns:
            str: Path to the created thumbnail file.

        Raises:
            RuntimeError: If FFmpeg execution returns a non-zero exit code.
        """
        video_path = str(Path(video_path).resolve())
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg thumbnail extraction failed with code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error output:\n{result.stderr}"
            )

        return str(output_path)

    def _compute_9_16_crop_filter(self, video_path: str, face_center: Any = None) -> str:
        """Calculate the FFmpeg crop filter for 9:16 vertical framing centered on face coordinates."""
        w, h = self._get_video_dimensions(video_path)
        if w > 0 and h > 0:
            target_aspect = 9.0 / 16.0
            current_aspect = w / h

            if current_aspect >= target_aspect:
                # Video is wider than 9:16 (standard landscape)
                crop_h = h - (h % 2)
                crop_w = int(crop_h * target_aspect)
                crop_w -= (crop_w % 2)

                center_x = self._extract_center_coord(face_center, w, is_x=True)
                crop_x = int(round(center_x - crop_w / 2.0))
                crop_x = max(0, min(w - crop_w, crop_x))
                crop_x -= (crop_x % 2)
                crop_y = 0
            else:
                # Video is narrower than 9:16
                crop_w = w - (w % 2)
                crop_h = int(crop_w / target_aspect)
                crop_h -= (crop_h % 2)

                center_y = self._extract_center_coord(face_center, h, is_x=False)
                crop_y = int(round(center_y - crop_h / 2.0))
                crop_y = max(0, min(h - crop_h, crop_y))
                crop_y -= (crop_y % 2)
                crop_x = 0

            return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},setsar=1"

        # Fallback to dynamic FFmpeg expression when dimensions cannot be pre-probed
        return "crop=ih*9/16:ih:(iw-ow)/2:0,setsar=1"

    def _get_video_dimensions(self, video_path: str) -> tuple[int, int]:
        """Query video frame dimensions (width, height) using OpenCV or FFmpeg probing."""
        if cv2 is not None:
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    if w > 0 and h > 0:
                        return (w, h)
            except Exception as err:
                logger.debug("cv2.VideoCapture could not read video dimensions: %s", err)

        try:
            cmd = [self.ffmpeg_cmd, "-i", video_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            match = re.search(r",\s*(\d{2,5})x(\d{2,5})\s*[, ]", res.stderr)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        except Exception as err:
            logger.debug("FFmpeg dimension probe error: %s", err)

        return (0, 0)

    def _extract_center_coord(self, face_center: Any, total_dim: int, is_x: bool) -> float:
        """Extract pixel coordinate from face_center representation."""
        if face_center is None:
            return total_dim / 2.0

        val: float
        if isinstance(face_center, (int, float)):
            val = float(face_center)
        elif isinstance(face_center, (list, tuple)) and len(face_center) > 0:
            idx = 0 if is_x else (1 if len(face_center) > 1 else 0)
            val = float(face_center[idx])
        elif isinstance(face_center, dict):
            key = "x" if is_x else "y"
            alt_key = "center_x" if is_x else "center_y"
            val = float(face_center.get(key, face_center.get(alt_key, 0.5)))
        else:
            return total_dim / 2.0

        # Convert normalized 0.0-1.0 coordinate to absolute pixels
        if 0.0 <= val <= 1.0:
            return val * total_dim
        return val

    def _resolve_face_center(self, face_track: Any, clip_index: int, start: float, end: float) -> Any:
        """Resolve face center coordinate for a specific clip window."""
        if face_track is None:
            return None

        if callable(face_track):
            try:
                return face_track((start + end) / 2.0)
            except Exception:
                return None

        if isinstance(face_track, dict):
            if clip_index in face_track:
                return face_track[clip_index]
            mid = (start + end) / 2.0
            best_t, best_val = None, None
            for k, v in face_track.items():
                if isinstance(k, (int, float)):
                    if best_t is None or abs(k - mid) < abs(best_t - mid):
                        best_t, best_val = k, v
            return best_val

        if isinstance(face_track, (list, tuple)):
            if len(face_track) == 0:
                return None
            if clip_index < len(face_track) and not isinstance(face_track[0], dict):
                return face_track[clip_index]
            if isinstance(face_track[0], dict) and "timestamp" in face_track[0]:
                mid = (start + end) / 2.0
                in_window = [e for e in face_track if start <= e.get("timestamp", 0) <= end]
                if in_window:
                    xs = [e.get("center_x", e.get("x", 0.5)) for e in in_window]
                    ys = [e.get("center_y", e.get("y", 0.5)) for e in in_window]
                    return (sum(xs) / len(xs), sum(ys) / len(ys))
                closest = min(face_track, key=lambda e: abs(e.get("timestamp", 0) - mid))
                return (closest.get("center_x", closest.get("x", 0.5)), closest.get("center_y", closest.get("y", 0.5)))
            if clip_index < len(face_track):
                return face_track[clip_index]

        return face_track

    def _resolve_subtitle_path(
        self,
        subtitle_paths: str | Path | list[str | Path] | dict[int, str | Path] | None,
        clip_index: int,
    ) -> Optional[Path]:
        """Resolve the ASS subtitle path for a specific clip index."""
        if not subtitle_paths:
            return None

        sub_candidate: Any = None
        if isinstance(subtitle_paths, (str, Path)):
            sub_candidate = subtitle_paths
        elif isinstance(subtitle_paths, (list, tuple)):
            if clip_index < len(subtitle_paths):
                sub_candidate = subtitle_paths[clip_index]
        elif isinstance(subtitle_paths, dict):
            sub_candidate = subtitle_paths.get(clip_index) or subtitle_paths.get(f"clip_{clip_index}")

        if sub_candidate:
            resolved = Path(sub_candidate).resolve()
            if resolved.exists():
                return resolved
            logger.warning("Subtitle path specified but not found on disk: %s", resolved)

        return None

    def _parse_window(self, window: Any) -> tuple[float, float, float, Optional[bool]]:
        """Extract start, end, hype_score, and optional reframe override from a window item."""
        reframe_override: Optional[bool] = None

        if isinstance(window, dict):
            start = float(window.get("start", 0.0))
            end = float(window.get("end", 0.0))
            hype_score = float(window.get("hype_score", window.get("score", 0.0)))
            reframe_override = window.get("reframe")
        elif hasattr(window, "start") and hasattr(window, "end"):
            start = float(window.start)
            end = float(window.end)
            hype_score = float(getattr(window, "hype_score", getattr(window, "score", 0.0)))
            reframe_override = getattr(window, "reframe", None)
        elif isinstance(window, (list, tuple)):
            start = float(window[0])
            end = float(window[1])
            hype_score = float(window[2]) if len(window) > 2 else 0.0
        else:
            raise TypeError(f"Unsupported clip window type: {type(window)}")

        return start, end, hype_score, reframe_override

    def _notify_progress(
        self,
        callback: Callable[..., None] | None,
        fraction: float,
        message: str = "",
    ) -> None:
        """Safely trigger progress callback handling both single-arg and two-arg signatures."""
        if not callback or not callable(callback):
            return
        try:
            callback(min(1.0, max(0.0, fraction)), message)
        except TypeError:
            try:
                callback(min(1.0, max(0.0, fraction)))
            except Exception as err:
                logger.debug("Progress callback execution error: %s", err)
        except Exception as err:
            logger.debug("Progress callback execution error: %s", err)
