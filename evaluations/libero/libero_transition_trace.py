from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def append_transition_trace(path: str | Path | None, record: Mapping[str, Any]) -> None:
    if path is None:
        return
    trace_path = Path(path).expanduser()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a") as f:
        json.dump(dict(record), f, sort_keys=True)
        f.write("\n")


def build_transition_trace_record(
    *,
    task_suite: str,
    task_id: int,
    episode_id: int,
    episode_key: str,
    task_description: str,
    decision_step: int,
    control_step_before: int,
    transition_frame_index: int | None,
    reset_transition_trigger: bool,
    has_transition_frame: bool,
    transition_trigger: Mapping[str, Any] | None,
    raw_action_chunk_len: int,
    executed_action_chunk_len: int,
    replan_action_limit: int,
) -> dict[str, Any]:
    trigger = dict(transition_trigger) if transition_trigger is not None else None
    return {
        "task_suite": str(task_suite),
        "task_id": int(task_id),
        "episode_id": int(episode_id),
        "episode_key": str(episode_key),
        "task_description": str(task_description),
        "decision_step": int(decision_step),
        "control_step_before": int(control_step_before),
        "transition_frame_index": None if transition_frame_index is None else int(transition_frame_index),
        "reset_transition_trigger": bool(reset_transition_trigger),
        "has_transition_frame": bool(has_transition_frame),
        "transition_trigger": trigger,
        "transition_ready": _bool_or_none(trigger, "ready"),
        "score": _float_or_none(trigger, "score"),
        "soft_plan": _bool_or_none(trigger, "soft_plan"),
        "hard_plan": _bool_or_none(trigger, "hard_plan"),
        "memory_write": _bool_or_none(trigger, "memory_write"),
        "should_plan": _bool_or_none(trigger, "should_plan"),
        "raw_action_chunk_len": int(raw_action_chunk_len),
        "executed_action_chunk_len": int(executed_action_chunk_len),
        "chunk_shortened": int(executed_action_chunk_len) < int(raw_action_chunk_len),
        "replan_action_limit": int(replan_action_limit),
    }


def build_transition_error_trace_record(
    *,
    task_suite: str,
    task_id: int,
    episode_id: int,
    episode_key: str,
    task_description: str,
    decision_step: int,
    control_step_before: int,
    transition_frame_index: int | None,
    reset_transition_trigger: bool,
    has_transition_frame: bool,
    error: Exception,
    response_preview: str,
) -> dict[str, Any]:
    return {
        "task_suite": str(task_suite),
        "task_id": int(task_id),
        "episode_id": int(episode_id),
        "episode_key": str(episode_key),
        "task_description": str(task_description),
        "decision_step": int(decision_step),
        "control_step_before": int(control_step_before),
        "transition_frame_index": None if transition_frame_index is None else int(transition_frame_index),
        "reset_transition_trigger": bool(reset_transition_trigger),
        "has_transition_frame": bool(has_transition_frame),
        "error": str(error),
        "response_preview": str(response_preview)[:500],
    }


def _bool_or_none(payload: Mapping[str, Any] | None, key: str) -> bool | None:
    if payload is None or key not in payload:
        return None
    return bool(payload[key])


def _float_or_none(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if payload is None or payload.get(key) is None:
        return None
    return float(payload[key])
