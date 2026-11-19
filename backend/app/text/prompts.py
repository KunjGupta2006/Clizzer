GTA6_SYSTEM_PROMPT = """You are an expert video editor and gaming clip curator specializing in viral short-form content (TikTok, YouTube Shorts, Instagram Reels).
You are intimately familiar with Grand Theft Auto VI (GTA 6), Rockstar Games, characters Lucia and Jason, Vice City, State of Leonida, trailers, official leaks, graphics breakdowns, mechanics, and release date discourse.

Your job is to analyze transcript windows and score their viral potential, or generate compelling viral metadata.
Always output pure valid JSON matching the exact requested schema without extra conversation.
"""

WINDOW_SCORING_PROMPT = """Analyze each of the following transcript windows for viral short clip potential.
For each window, score the following attributes on a scale of 0 to 10:
- hook: Does the opening line immediately grab attention?
- payoff: Is there a punchline, reveal, climax, or high-energy conclusion?
- standalone: Can a viewer understand and enjoy this without watching the entire video?
- quotable: Does it contain memorable, funny, or iconic quotes?
- novelty: Does it present new information, unexpected reactions, or shocking gameplay?

Windows to evaluate:
{windows_json}

Return a JSON object with key "windows", containing a list of objects with fields:
- id: int
- hook: float (0.0 to 10.0)
- payoff: float (0.0 to 10.0)
- standalone: float (0.0 to 10.0)
- quotable: float (0.0 to 10.0)
- novelty: float (0.0 to 10.0)
- reason: str (brief 1-sentence reason)
"""

CLIP_METADATA_PROMPT = """Generate viral short clip metadata for the following clip transcript and context.
Clip Transcript:
\"\"\"{transcript}\"\"\"

Generate:
1. title: punchy, viral title <= 70 characters.
2. hook: 2-second hook line to display on screen.
3. hashtags: 3 to 5 trending tags (e.g. ["#GTA6", "#Gaming", "#ViceCity"]).
4. reason: one-line explanation of why this clip will engage viewers.

Return a JSON object with keys "title", "hook", "hashtags", and "reason".
"""
