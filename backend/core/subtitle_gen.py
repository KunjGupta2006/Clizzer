"""Subtitle generator module for Clizzer backend.

Generates ASS subtitle files with karaoke word highlights and GTA style formatting
using pysubs2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pysubs2
from pysubs2 import Alignment, Color, SSAEvent, SSAFile, SSAStyle

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """Generates ASS subtitle files from word-level timestamps.

    Supports:
    - 'karaoke' style: words highlight as spoken using ASS backslash-k tags.
    - 'gta' style: bold uppercase, yellow (#F5B800) with heavy black outline.
    Default styling uses GTA aesthetics: font Big Shoulders Display, white text,
    black outline width 3, shadow, bottom center alignment.
    """

    def __init__(
        self,
        font_name: str = "Big Shoulders Display",
        font_size: int = 40,
        outline_width: float = 3.0,
        shadow_width: float = 2.0,
        max_words_per_line: int = 6,
    ) -> None:
        """Initialize SubtitleGenerator with GTA style settings.

        Args:
            font_name: Typographic font family (default 'Big Shoulders Display').
            font_size: Font size in subtitle points.
            outline_width: Outline border thickness in pixels (default 3.0).
            shadow_width: Drop shadow depth in pixels (default 2.0).
            max_words_per_line: Maximum number of words grouped per subtitle line (default 6).
        """
        self.font_name = font_name
        self.font_size = font_size
        self.outline_width = outline_width
        self.shadow_width = shadow_width
        self.max_words_per_line = max_words_per_line

        # Color definitions
        # Base text color: pure white
        self.white_color = Color(255, 255, 255)
        # GTA signature highlight yellow: #F5B800 (R:245, G:184, B:0)
        self.gta_yellow = Color(245, 184, 0)
        # Deep black for heavy outline
        self.black_color = Color(0, 0, 0)
        # Drop shadow color: semi-transparent black
        self.shadow_color = Color(0, 0, 0, 128)
        # Subtitle positioning: bottom center
        self.alignment = Alignment.BOTTOM_CENTER

    def generate_subtitles(
        self,
        words: list[dict],
        output_path: str | Path,
        style: str = "karaoke",
    ) -> str:
        """Create ASS subtitle file from word-level timestamps using pysubs2.

        'karaoke' style: words highlight as spoken using backslash-k tags.
        'gta' style: bold uppercase, yellow (#F5B800) with heavy black outline.
        Group words into lines of max 6 words.

        Args:
            words: List of dicts with word timestamps, e.g.
                   [{'word': 'Hello', 'start': 0.5, 'end': 0.9}, ...]
            output_path: Destination path for the generated .ass subtitle file.
            style: Formatting preset ('karaoke' or 'gta', default 'karaoke').

        Returns:
            str: Absolute path to the saved .ass file.

        Raises:
            ValueError: If output_path is empty or invalid.
        """
        if not output_path:
            raise ValueError("output_path must be a valid file path string or Path object.")

        dest_path = Path(output_path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        subs = SSAFile()

        # Target 9:16 vertical canvas coordinate space by default (1080x1920)
        subs.info["PlayResX"] = 1080
        subs.info["PlayResY"] = 1920
        subs.info["WrapStyle"] = "0"
        subs.info["ScaledBorderAndShadow"] = "yes"

        # Register GTA style and Karaoke style
        self._create_gta_style(subs)
        self._create_karaoke_style(subs)

        if not words:
            logger.warning("Empty words list provided to generate_subtitles. Creating empty ASS file.")
            subs.save(str(dest_path))
            return str(dest_path)

        # Group words into lines of max 6 words
        lines = self._group_words_into_lines(words, max_per_line=self.max_words_per_line)

        style_choice = style.lower().strip()
        if style_choice == "karaoke":
            self._create_karaoke_events(subs, lines, style_name="karaoke")
        elif style_choice == "gta":
            self._create_gta_events(subs, lines, style_name="gta")
        else:
            logger.warning("Unknown subtitle style '%s'; defaulting to 'karaoke'.", style)
            self._create_karaoke_events(subs, lines, style_name="karaoke")

        subs.save(str(dest_path))
        return str(dest_path)

    def _create_gta_style(self, subs: SSAFile) -> SSAStyle:
        """Create and register the GTA subtitle style on the SSAFile.

        Style: bold uppercase, yellow (#F5B800) with heavy black outline, shadow,
        bottom center alignment.

        Args:
            subs: SSAFile instance to register the style into.

        Returns:
            SSAStyle: The created GTA style instance.
        """
        style = SSAStyle()
        style.fontname = self.font_name
        style.fontsize = self.font_size
        style.bold = True
        style.italic = False
        style.primarycolor = self.gta_yellow
        style.secondarycolor = self.white_color
        style.outlinecolor = self.black_color
        style.backcolor = self.shadow_color
        style.outline = self.outline_width
        style.shadow = self.shadow_width
        style.alignment = self.alignment
        style.marginv = 60
        style.marginl = 30
        style.marginr = 30

        subs.styles["gta"] = style
        if "Default" not in subs.styles:
            subs.styles["Default"] = style
        return style

    def _create_karaoke_style(self, subs: SSAFile) -> SSAStyle:
        """Create and register the Karaoke subtitle style on the SSAFile.

        In ASS \\k karaoke: text starts in secondarycolor (white) and progressively
        highlights to primarycolor (yellow #F5B800) as words are spoken.

        Args:
            subs: SSAFile instance to register the style into.

        Returns:
            SSAStyle: The created Karaoke style instance.
        """
        style = SSAStyle()
        style.fontname = self.font_name
        style.fontsize = self.font_size
        style.bold = True
        style.italic = False
        # Highlighted color: Yellow #F5B800
        style.primarycolor = self.gta_yellow
        # Unhighlighted base color: White
        style.secondarycolor = self.white_color
        style.outlinecolor = self.black_color
        style.backcolor = self.shadow_color
        style.outline = self.outline_width
        style.shadow = self.shadow_width
        style.alignment = self.alignment
        style.marginv = 60
        style.marginl = 30
        style.marginr = 30

        subs.styles["karaoke"] = style
        return style

    def _group_words_into_lines(self, words: list[dict], max_per_line: int = 6) -> list[list[dict]]:
        """Group word-level timestamp dictionaries into subtitle lines.

        Enforces a maximum of max_per_line words per line, while also breaking
        naturally on significant acoustic pauses (>= 1.5s) or sentence-ending punctuation.

        Args:
            words: List of dicts with keys 'word'/'text', 'start', and 'end'.
            max_per_line: Upper bound on words per line (default 6).

        Returns:
            list[list[dict]]: List of lines, where each line is a list of word dicts.
        """
        if not words:
            return []

        normalized_words: list[dict] = []
        for w in words:
            if not isinstance(w, dict):
                continue
            text = str(w.get("word", w.get("text", ""))).strip()
            if not text:
                continue
            try:
                start = float(w.get("start", 0.0))
                end = float(w.get("end", start + 0.1))
            except (ValueError, TypeError):
                continue
            if end < start:
                end = start + 0.1
            normalized_words.append({
                "word": text,
                "start": start,
                "end": end,
            })

        if not normalized_words:
            return []

        lines: list[list[dict]] = []
        current_line: list[dict] = []

        for w in normalized_words:
            if not current_line:
                current_line.append(w)
                continue

            gap = w["start"] - current_line[-1]["end"]
            prev_word = current_line[-1]["word"]
            is_sentence_boundary = prev_word.endswith((".", "!", "?"))

            # Break line if max_per_line reached, significant speech pause, or sentence ended with enough words
            if (
                len(current_line) >= max_per_line
                or gap >= 1.5
                or (is_sentence_boundary and len(current_line) >= 3)
            ):
                lines.append(current_line)
                current_line = [w]
            else:
                current_line.append(w)

        if current_line:
            lines.append(current_line)

        return lines

    def _create_karaoke_events(
        self,
        subs: SSAFile,
        lines: list[list[dict]],
        style_name: str = "karaoke",
    ) -> None:
        """Generate ASS events with karaoke timing tags (\\k<centiseconds>).

        Args:
            subs: SSAFile instance to append events into.
            lines: List of lines containing word dictionaries.
            style_name: Style name to assign to the events (default 'karaoke').
        """
        for line in lines:
            if not line:
                continue

            start_sec = line[0]["start"]
            end_sec = max(line[-1]["end"], start_sec + 0.2)
            start_ms = int(round(start_sec * 1000))
            end_ms = int(round(end_sec * 1000))

            karaoke_tags: list[str] = []
            curr_time = start_sec

            for idx, w in enumerate(line):
                w_start = w["start"]
                w_end = w["end"]
                w_text = w["word"].upper()

                # Account for acoustic silence / gap before this word
                if w_start > curr_time:
                    gap_cs = int(round((w_start - curr_time) * 100))
                    if gap_cs > 0:
                        karaoke_tags.append(f"{{\\k{gap_cs}}}")

                # Duration of current word in centiseconds (1 cs = 10 ms)
                dur_cs = max(1, int(round((w_end - w_start) * 100)))
                karaoke_tags.append(f"{{\\k{dur_cs}}}{w_text}")

                # Append space separator between words within line
                if idx < len(line) - 1:
                    karaoke_tags.append(" ")

                curr_time = w_end

            event_text = "".join(karaoke_tags)
            event = SSAEvent(start=start_ms, end=end_ms, text=event_text, style=style_name)
            subs.events.append(event)

    def _create_gta_events(
        self,
        subs: SSAFile,
        lines: list[list[dict]],
        style_name: str = "gta",
    ) -> None:
        """Generate static GTA style dialogue events with bold uppercase text.

        Args:
            subs: SSAFile instance to append events into.
            lines: List of lines containing word dictionaries.
            style_name: Style name to assign to the events (default 'gta').
        """
        for line in lines:
            if not line:
                continue

            start_sec = line[0]["start"]
            end_sec = max(line[-1]["end"], start_sec + 0.2)
            start_ms = int(round(start_sec * 1000))
            end_ms = int(round(end_sec * 1000))

            line_text = " ".join(w["word"].upper() for w in line)
            event = SSAEvent(start=start_ms, end=end_ms, text=line_text, style=style_name)
            subs.events.append(event)
