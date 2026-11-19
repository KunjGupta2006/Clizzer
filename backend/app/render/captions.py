from pathlib import Path
import re
from typing import List, Optional
from backend.app.models import Word

PROFANITY_LIST = {"fuck", "shit", "bitch", "asshole", "cunt", "dick", "pussy", "nigger", "faggot"}


def escape_ass(text: str) -> str:
    # Escape backslashes and curly braces in ASS
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return text


def mask_profanity(word: str) -> str:
    lower = re.sub(r"[^\w]", "", word.lower())
    if lower in PROFANITY_LIST:
        if len(word) <= 2:
            return "**"
        return word[0] + "*" * (len(word) - 2) + word[-1]
    return word


def format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milli = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milli:03d}"


def format_timestamp_ass(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def generate_srt(
    words: List[Word],
    clip_start: float,
    clip_end: float,
    out_srt_path: Path,
    mask_curse: bool = True,
) -> Path:
    clip_words = [w for w in words if clip_start <= w.start <= clip_end]
    lines = []

    # Group words into 3-5 word chunks
    chunk_size = 4
    idx = 1
    for i in range(0, len(clip_words), chunk_size):
        group = clip_words[i : i + chunk_size]
        start_rel = max(0.0, group[0].start - clip_start)
        end_rel = max(start_rel + 0.3, group[-1].end - clip_start)

        group_text = " ".join([
            mask_profanity(w.text) if mask_curse else w.text
            for w in group
        ])

        lines.append(f"{idx}")
        lines.append(f"{format_timestamp_srt(start_rel)} --> {format_timestamp_srt(end_rel)}")
        lines.append(group_text)
        lines.append("")
        idx += 1

    out_srt_path.parent.mkdir(parents=True, exist_ok=True)
    out_srt_path.write_text("\n".join(lines), encoding="utf-8")
    return out_srt_path


def generate_ass(
    words: List[Word],
    clip_start: float,
    clip_end: float,
    out_ass_path: Path,
    mask_curse: bool = True,
    creator_credit: Optional[str] = None,
    video_width: int = 1080,
    video_height: int = 1920,
) -> Path:
    clip_words = [w for w in words if clip_start <= w.start <= clip_end]

    # ASS Header with viral, high-retention TikTok/Reels styling
    # Placed safely in lower-third (MarginV=22% height), bold Arial Black font, high-contrast black outline
    margin_v = int(video_height * 0.22)
    margin_lr = int(video_width * 0.08)
    font_size = int(video_width * 0.065)  # ~70px on 1080p

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,{font_size},&H00FFFFFF,&H0000E6FF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,5,3,2,{margin_lr},{margin_lr},{margin_v},1
Style: Credit,Arial Black,{int(font_size * 0.50)},&H00CCCCCC,&H00000000,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,8,{margin_lr},{margin_lr},50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    # Optional creator credit overlay at top of screen
    if creator_credit:
        esc_credit = escape_ass(f"Source: {creator_credit}")
        duration_ass = format_timestamp_ass(clip_end - clip_start)
        events.append(f"Dialogue: 0,0:00:00.00,{duration_ass},Credit,,0,0,0,,{esc_credit}")

    # Generate word-highlight viral karaoke dialogue (punchy 2-3 word chunks)
    chunk_size = 3
    for i in range(0, len(clip_words), chunk_size):
        group = clip_words[i : i + chunk_size]
        g_start = max(0.0, group[0].start - clip_start)
        g_end = max(g_start + 0.3, group[-1].end - clip_start)

        # Build highlighted word dialogue for each active word with pop-scale emphasis
        for active_idx, target_word in enumerate(group):
            w_start = max(0.0, target_word.start - clip_start)
            w_end = max(w_start + 0.15, target_word.end - clip_start)

            words_rendered = []
            for j, w in enumerate(group):
                clean_txt = mask_profanity(w.text) if mask_curse else w.text
                # Convert to punchy uppercase for modern viral look
                upper_txt = clean_txt.upper()
                esc_txt = escape_ass(upper_txt)
                if j == active_idx:
                    # Electric yellow highlight with subtle punch pop
                    words_rendered.append(f"{{\\c&H0000E6FF&\\b1\\fscx110\\fscy110}}{esc_txt}{{\\r}}")
                else:
                    words_rendered.append(esc_txt)

            sentence = " ".join(words_rendered)
            events.append(
                f"Dialogue: 1,{format_timestamp_ass(w_start)},{format_timestamp_ass(w_end)},Default,,0,0,0,,{sentence}"
            )

    out_ass_path.parent.mkdir(parents=True, exist_ok=True)
    out_ass_path.write_text(header + "\n".join(events), encoding="utf-8")
    return out_ass_path
