from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = REPO_ROOT / "evaluations" / "rmbench" / "policy" / "HiMemBridgeVLA" / "deploy_policy.py"


def load_policy_module():
    spec = importlib.util.spec_from_file_location("rmbench_himem_policy", POLICY_FILE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_request_from_observation_uses_three_cameras_and_endpose_state():
    module = load_policy_module()
    request = module.build_request_from_observation(_observation(), prompt="press the button")

    assert request["prompt"] == "press the button"
    assert request["robot_key"] == "rmbench"
    assert request["image_mask"] == [1, 1, 1]
    assert request["action_mask"] == [1] * 14
    assert len(request["image"]) == 3
    assert len(request["state"]) == 16
    assert request["state"][0] == pytest.approx(1.0)
    assert request["state"][7] == pytest.approx(0.25)
    assert request["state"][-1] == pytest.approx(0.75)


def test_build_request_from_observation_can_use_qpos_state():
    module = load_policy_module()
    request = module.build_request_from_observation(_observation(), prompt="", state_source="qpos")

    assert len(request["state"]) == 14
    assert request["state"] == pytest.approx([float(index) for index in range(14)])


def test_parse_action_response_accepts_plain_list_and_object_payload():
    module = load_policy_module()
    actions = [[float(i + j) for j in range(16)] for i in range(4)]

    parsed_plain = module.parse_action_response(json.dumps(actions), horizon=2, action_dim=14)
    parsed_object = module.parse_action_response(json.dumps({"actions": actions}), horizon=2, action_dim=14)

    assert len(parsed_plain) == 2
    assert len(parsed_plain[0]) == 14
    assert parsed_plain == parsed_object


def test_parse_action_response_rejects_server_error():
    module = load_policy_module()

    with pytest.raises(RuntimeError, match="server returned error"):
        module.parse_action_response(json.dumps({"error": "bad request"}), horizon=1, action_dim=14)


def test_eval_updates_observation_and_executes_qpos_actions():
    module = load_policy_module()
    env = FakeTaskEnv([_observation(), _observation(value=8), _observation(value=9)])
    model = FakePolicy(module)

    module.eval(env, model, env.get_obs())

    assert len(env.actions) == 2
    assert env.actions[0][1] == "qpos"
    assert model.update_count == 3


def _observation(value: int = 1) -> dict:
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    return {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image + 1},
            "right_camera": {"rgb": image + 2},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
        "endpose": {
            "left_endpose": np.ones((7,), dtype=np.float32),
            "left_gripper": np.array([0.25], dtype=np.float32),
            "right_endpose": np.full((7,), 2.0, dtype=np.float32),
            "right_gripper": np.array([0.75], dtype=np.float32),
        },
    }


class FakePolicy:
    def __init__(self, module):
        self.module = module
        self.action_type = "qpos"
        self.stop_on_success = False
        self.update_count = 0
        self.obs_cache = []

    def encode_observation(self, observation, prompt):
        return self.module.build_request_from_observation(observation, prompt=prompt)

    def update_obs(self, obs):
        self.obs_cache[:] = [obs]
        self.update_count += 1

    def get_action(self):
        return [[0.0] * 14, [1.0] * 14]


class FakeTaskEnv:
    def __init__(self, observations):
        self.observations = list(observations)
        self.index = 0
        self.actions = []
        self.eval_success = False

    def get_instruction(self):
        return "test instruction"

    def get_obs(self):
        obs = self.observations[min(self.index, len(self.observations) - 1)]
        self.index += 1
        return obs

    def take_action(self, action, action_type):
        self.actions.append((action, action_type))

