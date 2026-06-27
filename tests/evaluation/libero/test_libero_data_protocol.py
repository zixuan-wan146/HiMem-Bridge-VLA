from __future__ import annotations

import numpy as np

from himem_bridge_vla.benchmarks.libero.protocol import LIBERO_ACTION_DIM
from himem_bridge_vla.benchmarks.libero.protocol import LIBERO_SHORT_MEMORY_OFFSETS
from himem_bridge_vla.benchmarks.libero.protocol import LIBERO_STATE_DIM
from himem_bridge_vla.benchmarks.libero.protocol import LIBERO_VIEW_ORDER
from himem_bridge_vla.benchmarks.libero.data_protocol import build_request_from_observation
from himem_bridge_vla.benchmarks.libero.history import LiberoObservationHistory
from himem_bridge_vla.benchmarks.rmbench.protocol import RMBENCH_ACTION_DIM
from himem_bridge_vla.benchmarks.rmbench.protocol import RMBENCH_STATE_DIM
from himem_bridge_vla.benchmarks.rmbench.protocol import RMBENCH_VIEW_ORDER


def test_libero_protocol_is_not_rmbench_protocol():
    assert LIBERO_VIEW_ORDER == ("agentview_rgb", "eye_in_hand_rgb")
    assert RMBENCH_VIEW_ORDER == ("head_camera", "left_camera", "right_camera")
    assert LIBERO_ACTION_DIM == 7
    assert RMBENCH_ACTION_DIM == 14
    assert LIBERO_STATE_DIM == 8
    assert RMBENCH_STATE_DIM == 16
    assert LIBERO_SHORT_MEMORY_OFFSETS == (16, 8)


def test_build_request_from_observation_uses_two_raw_libero_views():
    agent = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    wrist = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
    obs = {
        "agentview_image": agent,
        "robot0_eye_in_hand_image": wrist,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }

    request = build_request_from_observation(obs, "put the mug away", reset_memory=True)

    assert request["prompt"] == "put the mug away"
    assert request["benchmark"] == "libero"
    assert tuple(request["images_by_view"]) == ("agentview_rgb", "eye_in_hand_rgb")
    assert request["images_by_view"]["agentview_rgb"] == agent.tolist()
    assert request["images_by_view"]["eye_in_hand_rgb"] == wrist.tolist()
    assert request["action_dim"] == 7
    assert request["robot_key"] == "libero"
    assert request["reset_memory"] is True
    assert len(request["state"]) == 8


def test_build_request_from_observation_includes_offset_short_memory_when_history_has_frames():
    obs0 = _obs_with_image_values(agent_value=1, wrist_value=2)
    obs8 = _obs_with_image_values(agent_value=8, wrist_value=9)
    obs16 = _obs_with_image_values(agent_value=16, wrist_value=17)
    history = LiberoObservationHistory(max_offset=16)
    history.record(0, obs0)
    history.record(8, obs8)
    history.record(16, obs16)

    request = build_request_from_observation(obs16, "pick", history=history, current_step=16)

    short_memory = request["short_memory_images_by_offset"]
    assert tuple(short_memory) == ("16", "8")
    assert short_memory["16"]["agentview_rgb"] == obs0["agentview_image"].tolist()
    assert short_memory["8"]["eye_in_hand_rgb"] == obs8["robot0_eye_in_hand_image"].tolist()


def test_build_request_from_observation_emits_empty_short_memory_object_for_warmup_steps():
    obs = _obs_with_image_values(agent_value=1, wrist_value=2)
    history = LiberoObservationHistory(max_offset=16)
    history.record(0, obs)

    request = build_request_from_observation(obs, "pick", history=history, current_step=0)

    assert request["short_memory_images_by_offset"] == {}


def test_build_request_from_observation_includes_executed_actions():
    obs = _obs_with_image_values(agent_value=1, wrist_value=2)

    request = build_request_from_observation(
        obs,
        "pick",
        executed_actions=[[0, 1, 2, 3, 4, 5, 1]],
        executed_action_mask=[True],
    )

    assert request["executed_actions"] == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0]]
    assert request["executed_action_mask"] == [1]


def _obs_with_image_values(*, agent_value: int, wrist_value: int) -> dict:
    return {
        "agentview_image": np.full((2, 2, 3), agent_value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((2, 2, 3), wrist_value, dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }
