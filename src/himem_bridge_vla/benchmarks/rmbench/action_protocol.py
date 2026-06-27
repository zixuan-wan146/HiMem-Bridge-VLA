from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any


DEFAULT_ACTION_HORIZON = 32
DEFAULT_ACTION_DIM = 14


def parse_action_response(message: str, *, horizon: int, action_dim: int) -> list[list[float]]:
    if int(horizon) <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if int(action_dim) <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Action response is not valid JSON: {exc}") from exc

    if isinstance(payload, Mapping):
        if "error" in payload:
            raise RuntimeError(f"HiMem server returned error: {payload['error']}")
        if "actions" not in payload:
            raise ValueError(f"Action response object must contain 'actions', got keys: {sorted(payload.keys())}")
        payload = payload["actions"]
    if not isinstance(payload, list):
        raise ValueError(f"Action response must be a list, got {type(payload).__name__}")
    if len(payload) < horizon:
        raise ValueError(f"Action response has {len(payload)} step(s), expected at least horizon {horizon}")

    parsed: list[list[float]] = []
    for step, row in enumerate(payload[:horizon]):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"Action at step {step} must be a sequence, got {type(row).__name__}")
        if len(row) < action_dim:
            raise ValueError(f"Action at step {step} has dimension {len(row)}, expected at least {action_dim}")
        parsed.append([_to_float(value, step, dim) for dim, value in enumerate(row[:action_dim])])
    return parsed


def _to_float(value: Any, step: int, dim: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action value at step {step}, dim {dim} is not numeric: {value!r}") from exc
