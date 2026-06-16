from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from motion_boundary.config import load_config
from motion_boundary.data import MotionBoundaryDataset, build_datasets
from motion_boundary.evaluate import evaluate_record_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heuristic motion-boundary baselines.")
    parser.add_argument("--config", default="motion_boundary/configs/default.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    _, val_dataset = build_datasets(config)
    results = evaluate_heuristics(val_dataset, config)
    output_path = Path(args.output) if args.output else Path(config["outputs"]["run_dir"]) / "heuristic_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def evaluate_heuristics(dataset: MotionBoundaryDataset, config: dict[str, Any]) -> dict[str, Any]:
    raw_scores = {
        "low_recent_motion": [],
        "last_step_change": [],
        "jerk": [],
        "gripper_transition": [],
    }
    for record in dataset.records:
        features = np.asarray(record.features, dtype=np.float64)
        diffs = np.diff(features, axis=0)
        if len(diffs) == 0:
            recent_motion = 0.0
            last_step = 0.0
            jerk = 0.0
        else:
            recent = diffs[-min(4, len(diffs)) :]
            recent_motion = float(np.mean(np.linalg.norm(recent, axis=1)))
            last_step = float(np.linalg.norm(diffs[-1]))
            if len(diffs) >= 2:
                jerk = float(np.linalg.norm(diffs[-1] - diffs[-2]))
            else:
                jerk = 0.0
        gripper = float(abs(features[-1, -1])) if features.size else 0.0

        raw_scores["low_recent_motion"].append(-recent_motion)
        raw_scores["last_step_change"].append(last_step)
        raw_scores["jerk"].append(jerk)
        raw_scores["gripper_transition"].append(gripper)

    normalized = {name: robust_minmax(values) for name, values in raw_scores.items()}
    normalized["combined"] = robust_minmax(
        0.4 * normalized["low_recent_motion"]
        + 0.3 * normalized["jerk"]
        + 0.2 * normalized["last_step_change"]
        + 0.1 * normalized["gripper_transition"]
    )

    metrics = {
        name: evaluate_record_scores(dataset.records, values.astype(float).tolist(), config)
        for name, values in normalized.items()
    }
    best_name = max(metrics, key=lambda name: metrics[name]["thresholds"]["best_f1_metrics"]["f1"])
    return {
        "best_heuristic": best_name,
        "heuristics": metrics,
    }


def robust_minmax(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    lo, hi = np.quantile(array, [0.01, 0.99])
    if hi <= lo:
        return np.zeros_like(array)
    array = np.clip(array, lo, hi)
    return (array - lo) / (hi - lo)


if __name__ == "__main__":
    raise SystemExit(main())
