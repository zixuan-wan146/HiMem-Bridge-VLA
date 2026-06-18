from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from himem_bridge_vla.transition_trigger_manager import TransitionTriggerServerResult


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO_ROOT / "scripts" / "himem_server.py"


def load_server_module():
    spec = importlib.util.spec_from_file_location("himem_server_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_rgb_image(value: int = 0):
    return [
        [[value, value, value], [value, value, value]],
        [[value, value, value], [value, value, value]],
    ]


def valid_payload() -> dict:
    return {
        "image": [tiny_rgb_image(1), tiny_rgb_image(2), tiny_rgb_image(3)],
        "state": [0.0] * 7,
        "prompt": "pick",
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7,
        "episode_id": "episode-a",
    }


class FakeModel:
    config = {"state_dim": 7, "per_action_dim": 7}
    per_action_dim = 7

    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.zeros(1))
        self.last_memory_write_gate = None
        self.last_coarse_plan_refresh = None

    def parameters(self):
        return iter([self._parameter])

    def run_inference(self, **kwargs):
        self.last_memory_write_gate = kwargs.get("memory_write_gate")
        self.last_coarse_plan_refresh = kwargs.get("coarse_plan_refresh")
        return torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6]], dtype=torch.float32)


class FakeNormalizer:
    def normalize_state(self, state, robot_key=None):
        return state

    def denormalize_action(self, action, robot_key=None):
        return action


class FakeTransitionManager:
    def __init__(self, result: TransitionTriggerServerResult) -> None:
        self.result = result
        self.calls = []
        self.resets = []

    def update(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def reset(self, episode_key=None):
        self.resets.append(episode_key)


def test_transition_episode_key_combines_session_and_episode():
    server = load_server_module()

    assert server.transition_episode_key("episode", "client") == "client:episode"
    assert server.transition_episode_key("episode", None) == "episode"
    assert server.transition_episode_key(None, "client") == "client"
    assert server.transition_episode_key(None, None) is None


def test_update_transition_trigger_maps_memory_write_to_gate():
    server = load_server_module()
    manager = FakeTransitionManager(
        TransitionTriggerServerResult(
            ready=True,
            dataset_name="robomme_four_tasks",
            score=0.91,
            memory_write=True,
            soft_plan=False,
            hard_plan=True,
            should_plan=True,
        )
    )
    request = {
        "episode_id": "episode-a",
        "session_id": "client",
        "transition_frame": {"action": [0.0]},
        "transition_dataset_name": "robomme_four_tasks",
        "transition_frame_index": 12,
        "reset_transition_trigger": False,
    }

    result, gate = server.update_transition_trigger(request, manager)

    assert gate == 1.0
    assert result.hard_plan is True
    assert manager.calls[0]["episode_key"] == "client:episode-a"
    assert manager.calls[0]["dataset_name"] == "robomme_four_tasks"
    assert manager.calls[0]["frame_index"] == 12


def test_infer_from_json_dict_returns_transition_debug_and_gate(monkeypatch):
    server = load_server_module()
    monkeypatch.setattr(server, "decode_image_from_list", lambda img, device: torch.zeros(3, 448, 448, device=device))
    model = FakeModel()
    manager = FakeTransitionManager(
        TransitionTriggerServerResult(
            ready=True,
            dataset_name="robomme_four_tasks",
            score=0.91,
            memory_write=True,
            soft_plan=False,
            hard_plan=True,
            should_plan=True,
        )
    )
    payload = valid_payload()
    payload["return_debug"] = True
    payload["transition_dataset_name"] = "robomme_four_tasks"
    payload["transition_frame"] = {
        "action": [0.0] * 7,
        "eef_state": [0.0] * 7,
        "joint_state": [0.0] * 7,
        "gripper_state": [0.0, 0.0],
    }

    response = server.infer_from_json_dict(payload, model, FakeNormalizer(), transition_manager=manager)

    assert response["actions"] == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6000000238418579]]
    assert response["transition_trigger"]["hard_plan"] is True
    assert response["transition_trigger"]["memory_write"] is True
    assert model.last_memory_write_gate == 1.0
    assert model.last_coarse_plan_refresh is True


def test_infer_from_json_dict_keeps_legacy_response_without_debug(monkeypatch):
    server = load_server_module()
    monkeypatch.setattr(server, "decode_image_from_list", lambda img, device: torch.zeros(3, 448, 448, device=device))
    model = FakeModel()

    response = server.infer_from_json_dict(valid_payload(), model, FakeNormalizer(), transition_manager=None)

    assert response == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6000000238418579]]
    assert model.last_memory_write_gate is None
    assert model.last_coarse_plan_refresh is False


def test_infer_from_json_dict_uses_reset_transition_as_plan_refresh(monkeypatch):
    server = load_server_module()
    monkeypatch.setattr(server, "decode_image_from_list", lambda img, device: torch.zeros(3, 448, 448, device=device))
    model = FakeModel()
    payload = valid_payload()
    payload["reset_transition_trigger"] = True

    server.infer_from_json_dict(payload, model, FakeNormalizer(), transition_manager=None)

    assert model.last_coarse_plan_refresh is True
