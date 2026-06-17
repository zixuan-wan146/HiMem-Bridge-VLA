from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


LIBERO_CONTROL_DIM = 7


@dataclass(frozen=True)
class ParsedActionResponse:
    actions: list[list[float]]
    transition_trigger: dict[str, Any] | None = None


def parse_action_response(
    message: str,
    horizon: int,
    min_action_dim: int = LIBERO_CONTROL_DIM,
) -> list[list[float]]:
    return parse_action_response_with_metadata(message, horizon, min_action_dim).actions


def parse_action_response_with_metadata(
    message: str,
    horizon: int,
    min_action_dim: int = LIBERO_CONTROL_DIM,
) -> ParsedActionResponse:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Action response is not valid JSON: {exc}") from exc

    if isinstance(payload, dict):
        if "error" in payload:
            raise RuntimeError(f"HiMem server returned error: {payload['error']}")
        if "actions" not in payload:
            raise ValueError(f"Action response object must contain 'actions', got keys: {sorted(payload.keys())}")
        transition_trigger = payload.get("transition_trigger")
        if transition_trigger is not None and not isinstance(transition_trigger, dict):
            raise ValueError("Action response transition_trigger must be an object")
        payload = payload["actions"]
    else:
        transition_trigger = None

    if not isinstance(payload, list):
        raise ValueError(f"Action response must be a list, got {type(payload).__name__}")

    if len(payload) < horizon:
        raise ValueError(f"Action response has {len(payload)} step(s), expected at least horizon {horizon}")

    actions: list[list[float]] = []
    for step, row in enumerate(payload[:horizon]):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"Action at step {step} must be a sequence, got {type(row).__name__}")
        if len(row) < min_action_dim:
            raise ValueError(
                f"Action at step {step} has dimension {len(row)}, expected at least {min_action_dim}"
            )
        actions.append([_to_float(value, step, dim) for dim, value in enumerate(row)])
    return ParsedActionResponse(actions=actions, transition_trigger=transition_trigger)


def to_libero_action(action: Sequence[float], control_dim: int = LIBERO_CONTROL_DIM) -> list[float]:
    if len(action) < control_dim:
        raise ValueError(f"Action dimension {len(action)} is smaller than LIBERO control dim {control_dim}")
    libero_action = [float(value) for value in action[:control_dim]]
    libero_action[6] = -1.0 if libero_action[6] > 0.5 else 1.0
    return libero_action


def select_actions_for_transition_policy(
    actions: list[list[float]],
    transition_trigger: dict[str, Any] | None,
    *,
    replan_action_limit: int = 0,
) -> list[list[float]]:
    """Optionally shorten an action chunk after a transition-trigger event."""

    if replan_action_limit <= 0 or transition_trigger is None:
        return actions
    triggered = any(
        bool(transition_trigger.get(key, False))
        for key in ("should_plan", "memory_write", "soft_plan", "hard_plan")
    )
    if not triggered:
        return actions
    return actions[: max(1, min(int(replan_action_limit), len(actions)))]


def _to_float(value: Any, step: int, dim: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action value at step {step}, dim {dim} is not numeric: {value!r}") from exc
