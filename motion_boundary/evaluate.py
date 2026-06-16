from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from motion_boundary.config import load_config
from motion_boundary.data import build_datasets
from motion_boundary.metrics import average_precision, evaluate_event_grid
from motion_boundary.model import MotionStateBoundaryHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained motion-state boundary detector.")
    parser.add_argument("--config", default="motion_boundary/configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = resolve_eval_config(checkpoint, args.config)
    _, val_dataset = build_datasets(config)
    model = MotionStateBoundaryHead(input_dim=int(checkpoint["input_dim"]), **config["model"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(val_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    metrics = evaluate_model(model, loader, config, device=args.device)
    output_path = Path(args.output) if args.output else Path(config["outputs"]["run_dir"]) / "eval_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def resolve_eval_config(checkpoint: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return load_config(config_path)
    if not config_path:
        return deepcopy(checkpoint_config)

    override = load_config(config_path)
    config = deepcopy(checkpoint_config)
    for section in ("data", "evaluation", "outputs"):
        if section in override:
            config[section] = deepcopy(override[section])
    for key in ("batch_size", "num_workers"):
        if key in override.get("training", {}):
            config.setdefault("training", {})[key] = deepcopy(override["training"][key])
    return config


@torch.no_grad()
def evaluate_model(
    model: MotionStateBoundaryHead,
    loader: DataLoader,
    config: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    model.eval()
    labels = []
    scores = []
    grouped_scores: dict[str, list[tuple[int, float]]] = defaultdict(list)
    grouped_events: dict[str, set[int]] = defaultdict(set)
    tolerance = int(config["evaluation"].get("event_tolerance", 3))
    for batch in loader:
        features = batch["features"].to(device)
        logits = model(features)
        batch_scores = torch.sigmoid(logits).reshape(-1).detach().cpu().tolist()
        batch_labels = batch["label"].reshape(-1).detach().cpu().tolist()
        batch_valid = batch["valid"].reshape(-1).detach().cpu().tolist()
        for score, label, valid, trajectory_id, frame_index, event_frame in zip(
            batch_scores,
            batch_labels,
            batch_valid,
            batch["trajectory_id"],
            batch["frame_index"],
            batch["event_frame"],
        ):
            frame = int(frame_index)
            grouped_scores[str(trajectory_id)].append((frame, float(score)))
            if label > 0:
                grouped_events[str(trajectory_id)].add(int(event_frame))
            if valid > 0:
                labels.append(float(label))
                scores.append(float(score))

    event_metrics = evaluate_event_grid(
        grouped_scores,
        {key: sorted(value) for key, value in grouped_events.items()},
        config["evaluation"].get("threshold_grid", []),
        tolerance=tolerance,
        cooldown=int(config["evaluation"].get("cooldown", 10)),
    )
    return {
        "auprc": average_precision(labels, scores),
        "event_metrics": [asdict(item) for item in event_metrics],
        "thresholds": select_thresholds(event_metrics, config),
    }


def select_thresholds(event_metrics: list[Any], config: dict[str, Any]) -> dict[str, float | None]:
    planner_target = float(config["evaluation"].get("planner_recall_target", 0.9))
    memory_target = float(config["evaluation"].get("memory_precision_target", 0.95))
    planner_candidates = [item for item in event_metrics if item.recall >= planner_target]
    memory_candidates = [item for item in event_metrics if item.precision >= memory_target]
    planner_threshold = max((item.threshold for item in planner_candidates), default=None)
    memory_threshold = min((item.threshold for item in memory_candidates), default=None)
    if planner_threshold is not None and memory_threshold is not None and memory_threshold <= planner_threshold:
        memory_threshold = None
    return {
        "replan_threshold": planner_threshold,
        "memory_write_threshold": memory_threshold,
    }


if __name__ == "__main__":
    raise SystemExit(main())
