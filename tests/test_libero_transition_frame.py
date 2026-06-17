from __future__ import annotations

import numpy as np
import pytest

from evaluations.libero.libero_transition_frame import build_libero_transition_frame


def test_build_libero_transition_frame_for_robomme_schema():
    obs = {
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_joint_pos": np.arange(7, dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }

    frame = build_libero_transition_frame(obs, [0, 1, 2, 3, 4, 5, -1], dataset_name="robomme_four_tasks")

    assert frame["action"] == pytest.approx([0, 1, 2, 3, 4, 5, -1])
    assert frame["eef_state"] == pytest.approx([1, 2, 3, 0, 0, 0, 1])
    assert frame["joint_state"] == pytest.approx([0, 1, 2, 3, 4, 5, 6])
    assert frame["gripper_state"] == pytest.approx([0.1, 0.2])


def test_build_libero_transition_frame_pads_missing_joint_state():
    obs = {
        "robot0_eef_pos": [0.0, 0.0, 0.0],
        "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
        "robot0_gripper_qpos": [0.0, 0.0],
    }

    frame = build_libero_transition_frame(obs, [0.0] * 7, dataset_name="robomme_four_tasks")

    assert frame["joint_state"] == pytest.approx([0.0] * 7)


def test_build_libero_transition_frame_rejects_unknown_schema():
    with pytest.raises(ValueError, match="robomme_four_tasks"):
        build_libero_transition_frame({}, [0.0] * 7, dataset_name="rmbench_9tasks")
