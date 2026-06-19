from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transition_trigger.evaluate import build_evaluation_dataset  # noqa: E402
from transition_trigger.metrics import average_precision, match_events  # noqa: E402
from transition_trigger.runtime import TransitionTriggerRuntime  # noqa: E402


DEFAULT_PACKAGE_DIR = (
    Path.home()
    / "autodl-tmp"
    / "runs"
    / "transition_trigger"
    / "selected"
    / "robomme_rmbench_w32_value_delta_transformer_d512"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a selected TransitionTrigger runtime policy.")
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE_DIR))
    parser.add_argument("--split", default="test", help="Dataset split to replay, usually eval or test.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = TransitionTriggerRuntime.from_package(args.package_dir, device=args.device)
    config = runtime.config
    config.setdefault("evaluation", {})["dataset_split"] = args.split
    dataset = build_evaluation_dataset(config)
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = score_dataset(runtime, loader)
    metrics = replay_policy(dataset.records, scores, runtime.config, package_dir=str(args.package_dir), split=args.split)
    output_path = Path(args.output) if args.output else Path(args.package_dir) / f"runtime_policy_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


@torch.no_grad()
def score_dataset(runtime: TransitionTriggerRuntime, loader: DataLoader) -> list[float]:
    scores: list[float] = []
    for batch in loader:
        batch_scores = runtime.score_window(batch["features"])
        scores.extend(float(score) for score in batch_scores.tolist())
    return scores


def replay_policy(records: list[Any], scores: list[float], config: dict[str, Any], *, package_dir: str, split: str) -> dict[str, Any]:
    if len(records) != len(scores):
        raise ValueError(f"records/scores length mismatch: {len(records)} != {len(scores)}")
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    grouped_events: dict[str, set[int]] = defaultdict(set)
    labels: list[float] = []
    valid_scores: list[float] = []
    for record, score in zip(records, scores):
        trajectory_id = str(record.trajectory_id)
        grouped[trajectory_id].append((int(record.frame_index), float(score)))
        if float(record.label) > 0:
            grouped_events[trajectory_id].add(int(record.event_frame))
        if float(record.valid) > 0:
            labels.append(float(record.label))
            valid_scores.append(float(score))

    predicted = {
        "soft_plan": defaultdict(list),
        "memory_write": defaultdict(list),
        "hard_plan": defaultdict(list),
        "all_plan": defaultdict(list),
    }
    violations = 0
    for trajectory_id, frame_scores in grouped.items():
        policy = runtime_policy_from_config(config)
        for frame, score in sorted(frame_scores):
            decision = policy.decide(score, frame_index=frame)
            if decision.memory_write and not decision.hard_plan:
                violations += 1
            if decision.soft_plan:
                predicted["soft_plan"][trajectory_id].append(frame)
            if decision.memory_write:
                predicted["memory_write"][trajectory_id].append(frame)
            if decision.hard_plan:
                predicted["hard_plan"][trajectory_id].append(frame)
            if decision.should_plan:
                predicted["all_plan"][trajectory_id].append(frame)

    evaluation = config["evaluation"]
    policy_config = config["trigger_policy"]
    event_metrics = {
        "soft_plan": evaluate_trigger_set(
            predicted["soft_plan"],
            grouped_events,
            grouped,
            evaluation,
            threshold=float(policy_config["planner_threshold"]),
        ),
        "memory_write": evaluate_trigger_set(
            predicted["memory_write"],
            grouped_events,
            grouped,
            evaluation,
            threshold=float(policy_config["memory_write_threshold"]),
        ),
        "all_plan": evaluate_trigger_set(
            predicted["all_plan"],
            grouped_events,
            grouped,
            evaluation,
            threshold=float(policy_config["planner_threshold"]),
        ),
    }
    counts = {name: sum(len(frames) for frames in by_traj.values()) for name, by_traj in predicted.items()}
    return {
        "package_dir": package_dir,
        "split": split,
        "policy": {
            "score_mode": str(policy_config.get("score_mode", "threshold")),
            "planner_threshold": float(policy_config["planner_threshold"]),
            "memory_write_threshold": float(policy_config["memory_write_threshold"]),
            "replan_cooldown_frames": int(policy_config.get("replan_cooldown_frames", 0)),
            "memory_write_cooldown_frames": int(policy_config.get("memory_write_cooldown_frames", 0)),
            "memory_write_implies_plan": bool(policy_config.get("memory_write_implies_plan", True)),
        },
        "scored_frames": len(scores),
        "valid_scored_frames": len(valid_scores),
        "trajectories": len(grouped),
        "events": sum(len(events) for events in grouped_events.values()),
        "frame_auprc": average_precision(labels, valid_scores),
        "trigger_counts": counts,
        "memory_write_implies_hard_plan_violations": violations,
        "event_metrics": event_metrics,
    }


def runtime_policy_from_config(config: dict[str, Any]):
    from transition_trigger.trigger_policy import build_transition_policy_from_config

    return build_transition_policy_from_config(config)


def evaluate_trigger_set(
    predicted_by_traj: dict[str, list[int]],
    events_by_traj: dict[str, set[int]],
    scored_by_traj: dict[str, list[tuple[int, float]]],
    evaluation: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    all_predicted: list[int] = []
    all_truth: list[int] = []
    total_scored_frames = 0
    offsets = 0
    tolerance = int(evaluation.get("event_tolerance", 3))
    min_delay = evaluation.get("match_min_delay")
    max_delay = evaluation.get("match_max_delay")
    early_tolerance = evaluation.get("early_tolerance")
    for trajectory_id, frame_scores in scored_by_traj.items():
        frames = [frame for frame, _ in frame_scores]
        events = sorted(events_by_traj.get(trajectory_id, set()))
        all_predicted.extend(offsets + frame for frame in predicted_by_traj.get(trajectory_id, []))
        all_truth.extend(offsets + event for event in events)
        total_scored_frames += len(frame_scores)
        max_frame = max(frames + events, default=0)
        offsets += max_frame + max(tolerance, abs(int(min_delay or 0)), abs(int(max_delay or 0))) + 100
    metrics = match_events(
        all_predicted,
        all_truth,
        tolerance,
        min_delay=None if min_delay is None else int(min_delay),
        max_delay=None if max_delay is None else int(max_delay),
        early_tolerance=None if early_tolerance is None else int(early_tolerance),
    )
    data = asdict(metrics)
    data["threshold"] = float(threshold)
    data["triggers_per_100_frames"] = len(all_predicted) / max(total_scored_frames, 1) * 100.0
    return data


if __name__ == "__main__":
    raise SystemExit(main())
