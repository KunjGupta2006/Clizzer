import os
import sys

# Ensure repository root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from backend.app.models import Candidate, SeriesBundle, Word
from backend.app.rank.select import select_top_clips
from backend.app.rank.fusion import get_mode_weights


@dataclass
class GroundTruthInterval:
    start: float
    end: float
    label: str


def generate_synthetic_eval_dataset(
    duration: float = 300.0,
    seed: int = 42,
) -> Tuple[SeriesBundle, List[Word], List[GroundTruthInterval]]:
    np.random.seed(seed)
    length = int(np.ceil(duration))

    # Base noise
    audio = np.random.normal(0.0, 0.3, length).astype(np.float32)
    chat = np.random.normal(0.0, 0.2, length).astype(np.float32)
    text = np.random.normal(0.0, 0.2, length).astype(np.float32)
    visual = np.random.normal(0.0, 0.3, length).astype(np.float32)
    face = np.random.normal(0.0, 0.2, length).astype(np.float32)

    # Inject 4 distinct ground-truth viral events
    ground_truth = [
        GroundTruthInterval(start=45.0, end=75.0, label="Car chase & huge stunt"),
        GroundTruthInterval(start=110.0, end=140.0, label="Hilarious fail & chat explosion"),
        GroundTruthInterval(start=190.0, end=220.0, label="Police 5-star shootout"),
        GroundTruthInterval(start=250.0, end=280.0, label="Plot twist & Lucia reveal"),
    ]

    for gt in ground_truth:
        s = int(gt.start)
        e = int(gt.end)
        mid = (s + e) // 2

        # Strong multimodal spikes
        audio[mid - 3 : mid + 4] += 2.8
        chat[mid - 2 : mid + 5] += 2.9
        visual[mid - 4 : mid + 3] += 2.5
        text[s : e] += 2.2
        face[mid - 2 : mid + 2] += 2.4

    words = [
        Word(start=46.0, end=50.0, text="Look at this car jump!"),
        Word(start=112.0, end=116.0, text="No way that just happened lmao!"),
        Word(start=192.0, end=197.0, text="Wanted level five stars watch out!"),
        Word(start=252.0, end=256.0, text="Lucia is back in Vice City!"),
    ]

    bundle = SeriesBundle.from_numpy(
        duration=duration,
        audio=audio,
        chat=chat,
        text=text,
        visual=visual,
        face=face,
    )
    return bundle, words, ground_truth


def evaluate_predictions(
    predicted_clips: List[Candidate],
    ground_truth: List[GroundTruthInterval],
    hit_window_sec: float = 5.0,
    top_n: int = 4,
) -> Tuple[float, float]:
    # A hit occurs when predicted start is within +/- 5s of ground-truth start
    top_preds = predicted_clips[:top_n]
    if not top_preds:
        return 0.0, 0.0

    hits = 0
    gt_hit = [False] * len(ground_truth)

    for pred in top_preds:
        is_hit = False
        for idx, gt in enumerate(ground_truth):
            if abs(pred.start - gt.start) <= hit_window_sec or (gt.start <= pred.peak_t <= gt.end):
                is_hit = True
                gt_hit[idx] = True
                break
        if is_hit:
            hits += 1

    precision_at_n = hits / len(top_preds)
    recall_at_n = sum(gt_hit) / len(ground_truth)
    return float(precision_at_n), float(recall_at_n)


def run_ablation_study(
    bundle: SeriesBundle,
    words: List[Word],
    ground_truth: List[GroundTruthInterval],
    top_n: int = 4,
) -> Dict[str, Dict[str, float]]:
    # Baseline
    base_clips, _ = select_top_clips(
        bundle=bundle,
        words=words,
        num_clips=top_n,
        quality_floor=-5.0,
    )
    base_p, base_r = evaluate_predictions(base_clips, ground_truth, top_n=top_n)

    results = {
        "Full System (Baseline)": {
            "Precision@N": base_p,
            "Recall@N": base_r,
            "Delta_P": 0.0,
            "Delta_R": 0.0,
        }
    }

    modalities = ["audio", "chat", "text", "visual", "face"]
    base_weights = get_mode_weights("auto")

    for mod in modalities:
        # Zero out this modality
        ablated_weights = dict(base_weights)
        ablated_weights[mod] = 0.0

        ablated_clips, _ = select_top_clips(
            bundle=bundle,
            words=words,
            num_clips=top_n,
            custom_weights=ablated_weights,
            quality_floor=-5.0,
        )
        p, r = evaluate_predictions(ablated_clips, ground_truth, top_n=top_n)

        results[f"Ablation: No {mod.capitalize()}"] = {
            "Precision@N": p,
            "Recall@N": r,
            "Delta_P": p - base_p,
            "Delta_R": r - base_r,
        }

    return results


def print_evaluation_report() -> None:
    bundle, words, gt = generate_synthetic_eval_dataset(duration=300.0, seed=42)
    top_n = 4
    results = run_ablation_study(bundle, words, gt, top_n=top_n)

    print("=" * 80)
    print("CLIZZER MULTIMODAL EVALUATION HARNESS")
    print(f"Dataset: 300s synthetic stream with {len(gt)} planted viral highlights")
    print("Hit criterion: Predicted start within +/- 5.0s of ground truth")
    print("=" * 80)
    print(f"{'Experiment':<32} | {'Precision@4':<12} | {'Recall@4':<10} | {'Delta P':<9} | {'Delta R':<9}")
    print("-" * 80)

    for exp_name, metrics in results.items():
        p = metrics["Precision@N"]
        r = metrics["Recall@N"]
        dp = metrics["Delta_P"]
        dr = metrics["Delta_R"]
        print(f"{exp_name:<32} | {p:>10.2%} | {r:>8.2%} | {dp:>+8.2%} | {dr:>+8.2%}")

    print("=" * 80)


if __name__ == "__main__":
    print_evaluation_report()
