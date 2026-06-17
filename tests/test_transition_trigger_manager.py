from __future__ import annotations

import torch

from himem_bridge_vla.transition_trigger_manager import ServerTransitionTriggerManager
from transition_trigger.trigger_policy import build_transition_policy_from_config


class FakeRuntime:
    input_dim = 1
    config = {
        "data": {"window_size": 2},
        "features": {
            "mode": "canonical_blocks",
            "expected_input_dim": 1,
            "include_deltas": False,
            "include_value_mask": False,
            "blocks": [{"name": "action", "key": "action", "dim": 1, "valid": 1.0}],
        },
        "trigger_policy": {
            "score_mode": "threshold",
            "planner_threshold": 0.5,
            "memory_write_threshold": 0.8,
            "replan_cooldown_frames": 0,
            "memory_write_cooldown_frames": 0,
            "memory_write_implies_plan": True,
        },
    }

    def new_policy(self):
        return build_transition_policy_from_config(self.config)

    def new_online_session(self, *, dataset_name=None):
        from transition_trigger.runtime import TransitionTriggerOnlineSession

        return TransitionTriggerOnlineSession(self, dataset_name=dataset_name)

    def score_window(self, features):
        assert tuple(features.shape) == (2, 1)
        return torch.tensor([0.9])


def test_server_transition_trigger_manager_buffers_until_ready():
    manager = ServerTransitionTriggerManager(FakeRuntime())

    first = manager.update(episode_key="episode-a", frame={"action": [0.0]}, frame_index=1)
    second = manager.update(episode_key="episode-a", frame={"action": [1.0]}, frame_index=2)

    assert first.ready is False
    assert first.should_plan is False
    assert second.ready is True
    assert second.memory_write is True
    assert second.hard_plan is True
    assert second.should_plan is True


def test_server_transition_trigger_manager_reset_clears_session():
    manager = ServerTransitionTriggerManager(FakeRuntime())
    manager.update(episode_key="episode-a", frame={"action": [0.0]}, frame_index=1)

    after_reset = manager.update(episode_key="episode-a", frame={"action": [1.0]}, frame_index=2, reset=True)

    assert after_reset.ready is False


def test_server_transition_trigger_manager_requires_episode_key():
    manager = ServerTransitionTriggerManager(FakeRuntime())

    try:
        manager.update(episode_key=None, frame={"action": [0.0]})
    except ValueError as exc:
        assert "episode_id or session_id" in str(exc)
    else:
        raise AssertionError("expected missing episode key to fail")
