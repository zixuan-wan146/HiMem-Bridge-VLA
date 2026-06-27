from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


LIBERO_ENV_VIEW_TO_CACHE_VIEW = {
    "agentview_image": "agentview_rgb",
    "robot0_eye_in_hand_image": "eye_in_hand_rgb",
}


def build_libero_images_by_view(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        cache_view: np.ascontiguousarray(obs[env_key])
        for env_key, cache_view in LIBERO_ENV_VIEW_TO_CACHE_VIEW.items()
    }


def build_libero_state(obs: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)


def quat2axisangle(quat: Sequence[float] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape[0] < 4:
        raise ValueError(f"quat must contain at least 4 values, got shape {quat.shape}")
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den
