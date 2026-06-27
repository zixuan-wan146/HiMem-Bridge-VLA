from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from himem_bridge_vla.evaluation.run_metadata import build_run_metadata


@dataclass(frozen=True)
class EpisodeResult:
    task_suite: str
    task_id: int
    episode_id: int
    task_description: str
    success: bool
    decision_steps: int
    control_steps: int
    failure_reason: str = ""
    video_path: str = ""


def summarize_episode_results(results: Sequence[EpisodeResult | Mapping[str, Any]]) -> dict[str, Any]:
    episodes = [_episode_to_dict(result) for result in results]
    successful = [episode for episode in episodes if episode["success"]]

    suite_names = sorted({episode["task_suite"] for episode in episodes})
    suite_summaries = {
        suite_name: _summarize_subset(
            [episode for episode in episodes if episode["task_suite"] == suite_name]
        )
        for suite_name in suite_names
    }

    summary = _summarize_subset(episodes)
    summary["suites"] = suite_summaries
    summary["successful_episode_ids"] = [
        {
            "task_suite": episode["task_suite"],
            "task_id": episode["task_id"],
            "episode_id": episode["episode_id"],
        }
        for episode in successful
    ]
    return summary


def write_result_summary(
    path: str | Path,
    *,
    config: Any,
    results: Sequence[EpisodeResult | Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    result_path = Path(path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [_episode_to_dict(result) for result in results]
    payload = {
        "config": _serialize_config(config),
        "metadata": dict(metadata) if metadata is not None else build_run_metadata(),
        "summary": summarize_episode_results(episodes),
        "episodes": episodes,
    }
    with result_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return result_path


def _summarize_subset(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_episodes = len(episodes)
    successful_episodes = sum(1 for episode in episodes if episode["success"])
    success_decision_steps = [
        int(episode["decision_steps"]) for episode in episodes if episode["success"]
    ]
    all_decision_steps = [int(episode["decision_steps"]) for episode in episodes]
    all_control_steps = [int(episode["control_steps"]) for episode in episodes]

    return {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "failed_episodes": total_episodes - successful_episodes,
        "success_rate": successful_episodes / total_episodes if total_episodes else 0.0,
        "average_decision_steps": _mean(all_decision_steps),
        "average_control_steps": _mean(all_control_steps),
        "average_success_decision_steps": _mean(success_decision_steps),
    }


def _mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _episode_to_dict(result: EpisodeResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, EpisodeResult):
        payload = asdict(result)
    else:
        payload = dict(result)
    payload["success"] = bool(payload["success"])
    payload["task_id"] = int(payload["task_id"])
    payload["episode_id"] = int(payload["episode_id"])
    payload["decision_steps"] = int(payload["decision_steps"])
    payload["control_steps"] = int(payload["control_steps"])
    payload["failure_reason"] = str(payload.get("failure_reason") or "")
    payload["video_path"] = str(payload.get("video_path") or "")
    return payload


def _serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    return {"repr": repr(config)}
