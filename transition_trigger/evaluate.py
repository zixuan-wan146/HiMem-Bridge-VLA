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

from transition_trigger.config import load_config
from transition_trigger.data import build_datasets
from transition_trigger.data import WindowRecord
from transition_trigger.metrics import average_precision, evaluate_event_grid
from transition_trigger.model import TransitionTriggerHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained causal transition trigger.")
    parser.add_argument("--config", default="transition_trigger/configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = resolve_eval_config(checkpoint, args.config)
    val_dataset = build_evaluation_dataset(config)
    model = TransitionTriggerHead(input_dim=int(checkpoint["input_dim"]), **config["model"]).to(args.device)
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


def build_evaluation_dataset(config: dict[str, Any]):
    split_name = str(config.get("evaluation", {}).get("dataset_split", "eval"))
    eval_config = deepcopy(config)
    data_config = eval_config.setdefault("data", {})
    if split_name == "eval":
        data_config["eval_split"] = data_config.get("eval_split", "eval")
    elif split_name == "test":
        data_config["eval_split"] = data_config.get("test_split", "test")
    else:
        data_config["eval_split"] = split_name
    _, dataset = build_datasets(eval_config)
    return dataset


@torch.no_grad()
def evaluate_model(
    model: TransitionTriggerHead,
    loader: DataLoader,
    config: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    model.eval()
    records: list[WindowRecord] = []
    scores: list[float] = []
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
            records.append(
                WindowRecord(
                    trajectory_id=str(trajectory_id),
                    task_id=None,
                    frame_index=int(frame_index),
                    event_frame=int(event_frame),
                    features=torch.empty(0).numpy(),
                    label=float(label),
                    valid=float(valid),
                    group="",
                    distance_to_event=None,
                )
            )
            scores.append(float(score))
    return evaluate_record_scores(records, scores, config)


def evaluate_record_scores(records: list[WindowRecord], scores: list[float], config: dict[str, Any]) -> dict[str, Any]:
    if len(records) != len(scores):
        raise ValueError(f"records/scores length mismatch: {len(records)} != {len(scores)}")
    labels = []
    valid_scores = []
    grouped_scores: dict[str, list[tuple[int, float]]] = defaultdict(list)
    grouped_events: dict[str, set[int]] = defaultdict(set)
    tolerance = int(config["evaluation"].get("event_tolerance", 3))
    for record, score in zip(records, scores):
        grouped_scores[str(record.trajectory_id)].append((int(record.frame_index), float(score)))
        if record.label > 0:
            grouped_events[str(record.trajectory_id)].add(int(record.event_frame))
        if record.valid > 0:
            labels.append(float(record.label))
            valid_scores.append(float(score))
    min_delay = config["evaluation"].get("match_min_delay")
    max_delay = config["evaluation"].get("match_max_delay")
    early_tolerance = config["evaluation"].get("early_tolerance")
    event_metrics = evaluate_event_grid(
        grouped_scores,
        {key: sorted(value) for key, value in grouped_events.items()},
        config["evaluation"].get("threshold_grid", []),
        tolerance=tolerance,
        cooldown=int(config["evaluation"].get("cooldown", 10)),
        min_delay=None if min_delay is None else int(min_delay),
        max_delay=None if max_delay is None else int(max_delay),
        early_tolerance=None if early_tolerance is None else int(early_tolerance),
    )
    return {
        "auprc": average_precision(labels, valid_scores),
        "event_metrics": [asdict(item) for item in event_metrics],
        "thresholds": select_thresholds(event_metrics, config),
    }


def select_thresholds(event_metrics: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    planner_target = float(config["evaluation"].get("planner_recall_target", 0.9))
    planner_precision_floor = float(config["evaluation"].get("planner_precision_floor", 0.0))
    memory_target = float(config["evaluation"].get("memory_precision_target", 0.95))
    memory_recall_floor = float(config["evaluation"].get("memory_recall_floor", 0.0))
    best_f1 = max(event_metrics, key=lambda item: item.f1, default=None)

    fixed_memory_threshold = config["evaluation"].get("memory_write_fixed_threshold")
    planner_candidates = [
        item
        for item in event_metrics
        if item.recall >= planner_target and item.precision >= planner_precision_floor
    ]
    if fixed_memory_threshold is not None:
        planner_candidates = [item for item in planner_candidates if item.threshold < float(fixed_memory_threshold)]
    if not planner_candidates:
        planner_candidates = [item for item in event_metrics if item.recall >= planner_target]
        if fixed_memory_threshold is not None:
            planner_candidates = [item for item in planner_candidates if item.threshold < float(fixed_memory_threshold)]
    planner_item = max(planner_candidates, key=lambda item: item.threshold, default=None)

    fixed_memory_threshold = config["evaluation"].get("memory_write_fixed_threshold")
    memory_item = None
    if fixed_memory_threshold is not None:
        fixed_memory_threshold = float(fixed_memory_threshold)
        fixed_candidates = [item for item in event_metrics if item.threshold >= fixed_memory_threshold]
        memory_item = min(fixed_candidates, key=lambda item: item.threshold, default=None)
        if memory_item is None:
            memory_item = min(event_metrics, key=lambda item: abs(item.threshold - fixed_memory_threshold), default=None)
    else:
        memory_candidates = [
            item
            for item in event_metrics
            if item.precision >= memory_target and item.recall >= memory_recall_floor
        ]
        if planner_item is not None:
            memory_candidates = [item for item in memory_candidates if item.threshold > planner_item.threshold]
        memory_item = min(memory_candidates, key=lambda item: item.threshold, default=None)
    return {
        "best_f1_threshold": None if best_f1 is None else best_f1.threshold,
        "best_f1_metrics": None if best_f1 is None else asdict(best_f1),
        "replan_threshold": None if planner_item is None else planner_item.threshold,
        "replan_metrics": None if planner_item is None else asdict(planner_item),
        "memory_write_threshold": None if memory_item is None else memory_item.threshold,
        "memory_write_metrics": None if memory_item is None else asdict(memory_item),
        "targets": {
            "planner_recall_target": planner_target,
            "planner_precision_floor": planner_precision_floor,
            "memory_precision_target": memory_target,
            "memory_recall_floor": memory_recall_floor,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
