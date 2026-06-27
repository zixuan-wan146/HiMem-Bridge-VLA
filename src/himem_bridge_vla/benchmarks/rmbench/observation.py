from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


def build_rmbench_images_by_view(
    observation: Mapping[str, Any],
    *,
    camera_names=DEFAULT_CAMERA_NAMES,
) -> dict[str, np.ndarray]:
    return {str(camera_name): _extract_rgb(observation, str(camera_name)) for camera_name in camera_names}


def build_rmbench_state(observation: Mapping[str, Any], *, state_source: str) -> np.ndarray:
    if state_source == "qpos":
        return _build_qpos_state(observation)
    if state_source != "endpose":
        raise ValueError(f"state_source must be 'endpose' or 'qpos', got {state_source!r}")
    return _build_endpose_state(observation)


def _extract_rgb(observation: Mapping[str, Any], camera_name: str) -> np.ndarray:
    try:
        image = observation["observation"][camera_name]["rgb"]
    except KeyError as exc:
        raise KeyError(f"RMBench observation is missing camera {camera_name!r} rgb") from exc
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"camera {camera_name!r} rgb must have shape HxWx3, got {array.shape}")
    if array.size == 0:
        raise ValueError(f"camera {camera_name!r} rgb is empty")
    if array.min() < 0 or array.max() > 255:
        raise ValueError(f"camera {camera_name!r} rgb values must be in 0..255")
    return np.asarray(array, dtype=np.uint8)


def _build_qpos_state(observation: Mapping[str, Any]) -> np.ndarray:
    joint_action = observation["joint_action"]
    if "vector" in joint_action:
        return _flat_float_array(joint_action["vector"], "joint_action.vector")
    return _flat_float_array(
        list(joint_action["left_arm"])
        + [joint_action["left_gripper"]]
        + list(joint_action["right_arm"])
        + [joint_action["right_gripper"]],
        "joint_action split fields",
    )


def _build_endpose_state(observation: Mapping[str, Any]) -> np.ndarray:
    endpose = observation["endpose"]
    return _flat_float_array(
        list(np.asarray(endpose["left_endpose"]).reshape(-1))
        + list(np.asarray(endpose["left_gripper"]).reshape(-1))
        + list(np.asarray(endpose["right_endpose"]).reshape(-1))
        + list(np.asarray(endpose["right_gripper"]).reshape(-1)),
        "endpose state",
    )


def _flat_float_array(values: Any, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array
