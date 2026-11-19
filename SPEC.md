# CLIZZER SPECIFICATION & ARCHITECTURE

## System Goal
Given a YouTube link or uploaded video file, analyze multimodal signals (speech/ASR, speaker diarization, audio energy/events, video scene cuts, faces/expressions, OCR, live chat sentiment/velocity, LLM-evaluated semantic windows) to output top-N ranked short vertical clips (9:16 / 1:1 / 16:9) with titles, hooks, hashtags, and subtitles.

## Modality Weights
- Auto (Default): Audio 0.25, Chat 0.25, Text 0.25, Visual 0.15, Face 0.10
- Hype: Audio 0.30, Chat 0.35, Text 0.10, Visual 0.15, Face 0.10
- Funny: Audio 0.25, Chat 0.30, Text 0.25, Visual 0.10, Face 0.10
- Informational: Audio 0.15, Chat 0.10, Text 0.50, Visual 0.15, Face 0.10

## Fallback Chains
- LLM: Groq (llama-3.3-70b-versatile -> llama-3.1-8b-instant) -> Ollama -> Heuristic
- Diarization: PyAnnote 3.1 (with HF_TOKEN) -> SpeechBrain ECAPA / Clustering -> Single Speaker
- Vision/Faces: InsightFace -> MediaPipe -> No Face (0 weight, letterbox blurred bg)
- Chat: YouTube/Twitch live chat fetch -> No Chat (0 weight)
- Audio events: Transformers AST -> RMS/Spectral flux only
