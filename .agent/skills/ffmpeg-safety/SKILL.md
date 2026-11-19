---
name: ffmpeg-safety
description: Guidelines for robust, secure, and performant ffmpeg/ffprobe execution.
---
# FFmpeg Safety

1. Subprocess execution with explicit argument lists only, never `shell=True`.
2. Timeout on every call to prevent hung processes.
3. Validate source files with `ffprobe` first; raise `SourceUnavailable` if invalid.
4. Extract raw video frames via `-f rawvideo -pix_fmt rgb24 pipe:1` in fixed chunks.
5. Place `-ss` after `-i` when re-encoding to guarantee frame accuracy.
6. Encoder selection: use `h264_nvenc` if listed and available, else `libx264 -crf 19 -preset veryfast`.
7. Two-pass or filter loudnorm to target -14 LUFS (+/- 1.5 LUFS).
8. Crop x-coordinate from EMA-smoothed bounding boxes, clamped to frame bounds, even dimensions.
9. ASS captions: properly escape curly braces `{}` and backslashes `\`, maintain 10% safe margins.
10. Fixtures: generate synthetic test video/audio using lavfi `testsrc2` + `sine`.
