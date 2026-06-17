from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


JOINT_STATE_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_qpos",
    "robot0_joint_positions",
    "robot0_joint_state",
)


def build_libero_transition_frame(
    obs: Mapping[str, Any],
    action: Sequence[float],
    *,
    dataset_name: str,
) -> dict[str, list[float]]:
    if dataset_name != "robomme_four_tasks":
        raise ValueError(
            "LIBERO transition-frame adapter currently supports dataset_name='robomme_four_tasks' only"
        )
    return build_libero_robomme_frame(obs, action)


def build_libero_robomme_frame(obs: Mapping[str, Any], action: Sequence[float]) -> dict[str, list[float]]:
    eef_state = np.concatenate(
        [
            _as_vector(obs["robot0_eef_pos"], 3, "robot0_eef_pos"),
            _as_vector(obs["robot0_eef_quat"], 4, "robot0_eef_quat"),
        ]
    )
    joint_state = _first_available_vector(obs, JOINT_STATE_KEYS, dim=7)
    gripper_state = _as_vector(obs["robot0_gripper_qpos"], 2, "robot0_gripper_qpos")
    return {
        "action": _as_vector(action, 7, "action").tolist(),
        "eef_state": eef_state.astype(float).tolist(),
        "joint_state": joint_state.astype(float).tolist(),
        "gripper_state": gripper_state.astype(float).tolist(),
    }


def _first_available_vector(obs: Mapping[str, Any], keys: Sequence[str], *, dim: int) -> np.ndarray:
    for key in keys:
        if key in obs:
            return _as_vector(obs[key], dim, key)
    return np.zeros(dim, dtype=np.float32)


def _as_vector(value: Any, dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if array.shape[0] == dim:
        return array
    if array.shape[0] > dim:
        return array[:dim]
    padded = np.zeros(dim, dtype=np.float32)
    padded[: array.shape[0]] = array
    return padded
