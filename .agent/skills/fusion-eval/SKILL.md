---
name: fusion-eval
description: Rules for multimodal signal fusion, peak picking, MMR deduplication, and evaluation.
---
# Fusion & Evaluation

1. Z-score normalization using rolling median and Median Absolute Deviation (MAD) over a 120 s window.
2. Clip normalized values to `[-3.0, 3.0]`; use MAD epsilon `1e-6` to avoid division by zero.
3. Renormalize weights across all available modalities:
   `w_norm = w_i / sum(w_available)`
4. Modality weight presets must come only from `config.py` (auto, hype, funny, informational).
5. Planted-peak tests: generated peaks must be recovered within +/- 2 seconds.
6. Hit definition in evaluation: predicted start timestamp within +/- 5 seconds of ground truth label.
7. Ablation tests: zero out each modality series, run ranking, and record metric deltas.
8. Determinism test: `seed=42` must produce identical candidate ordering and scores on identical inputs.
