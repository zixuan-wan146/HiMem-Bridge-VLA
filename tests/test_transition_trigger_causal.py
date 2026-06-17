from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from transition_trigger.config import load_config
from transition_trigger.data import (
    WindowRecord,
    _build_records_for_trajectory,
    build_canonical_block_features,
    resolve_label_window,
    split_by_manifest,
)
from transition_trigger.metrics import match_events
from transition_trigger.model import TransitionTriggerHead
from transition_trigger.online_features import CanonicalFeatureBuilder, OnlineTransitionFeatureBuffer
from transition_trigger.runtime import TransitionTriggerOnlineSession, TransitionTriggerRuntime, TransitionTriggerSession
from transition_trigger.trigger_policy import (
    CausalPeakTransitionPolicy,
    StatefulTransitionPolicy,
    build_transition_policy_from_config,
    decide_transition_actions,
    decide_transition_actions_from_config,
)


def test_causal_post_label_window_ignores_boundary_frame():
    window = resolve_label_window(
        {
            "label_mode": "causal_post",
            "positive_min_delay": 1,
            "positive_max_delay": 3,
            "ignore_min_delay": -2,
            "ignore_max_delay": 0,
        }
    )

    assert window["positive_min"] == 1
    assert window["positive_max"] == 3
    assert window["ignore_min"] == -2
    assert window["ignore_max"] == 0


def test_causal_post_records_do_not_label_event_frame_positive():
    frames = np.arange(8, dtype=np.int64)
    df = pd.DataFrame(
        {
            "action": [np.zeros(2, dtype=np.float32) for _ in range(8)],
            "state": [np.zeros(2, dtype=np.float32) for _ in range(8)],
        }
    )
    records = _build_records_for_trajectory(
        "traj",
        None,
        frames,
        [3],
        {
            "window_size": 1,
            "label_mode": "causal_post",
            "positive_min_delay": 1,
            "positive_max_delay": 3,
            "ignore_min_delay": -2,
            "ignore_max_delay": 0,
            "hard_negative_radius": 4,
            "label_sigma": 2.0,
            "soft_labels": False,
        },
        {
            "use_action": True,
            "use_state": False,
            "use_delta_action": False,
            "use_delta_state": False,
            "use_gripper_transition": False,
            "normalize": False,
        },
        df=df,
    )
    by_frame = {record.frame_index: record for record in records}

    assert by_frame[3].group == "ignore"
    assert by_frame[3].valid == 0.0
    assert by_frame[4].group == "positive"
    assert by_frame[6].group == "positive"
    assert by_frame[7].group == "hard_negative"


def test_post_boundary_matching_counts_event_frame_as_early():
    metrics = match_events([10, 11], [10], min_delay=1, max_delay=3, early_tolerance=3)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.early_triggers == 1
    assert metrics.mean_trigger_delay == 1.0


def test_memory_write_always_triggers_hard_plan():
    decision = decide_transition_actions(0.85, planner_threshold=0.4, memory_write_threshold=0.8)

    assert decision.memory_write is True
    assert decision.hard_plan is True
    assert decision.soft_plan is False
    assert decision.should_plan is True


def test_soft_plan_can_fire_without_memory_write():
    decision = decide_transition_actions(0.5, planner_threshold=0.4, memory_write_threshold=0.8)

    assert decision.memory_write is False
    assert decision.hard_plan is False
    assert decision.soft_plan is True
    assert decision.should_plan is True


def test_trigger_policy_can_read_thresholds_from_config():
    decision = decide_transition_actions_from_config(
        0.85,
        {
            "trigger_policy": {
                "planner_threshold": 0.5,
                "memory_write_threshold": 0.8,
                "memory_write_implies_plan": True,
            }
        },
    )

    assert decision.memory_write is True
    assert decision.hard_plan is True
    assert decision.should_plan is True


def test_stateful_policy_replan_cooldown_does_not_block_memory_write():
    policy = StatefulTransitionPolicy(
        planner_threshold=0.5,
        memory_write_threshold=0.8,
        replan_cooldown_frames=10,
        memory_write_cooldown_frames=10,
    )

    soft = policy.decide(0.6, frame_index=100)
    memory = policy.decide(0.9, frame_index=102)

    assert soft.soft_plan is True
    assert memory.memory_write is True
    assert memory.hard_plan is True
    assert memory.should_plan is True


def test_stateful_policy_suppresses_repeated_soft_plans():
    policy = StatefulTransitionPolicy(
        planner_threshold=0.5,
        memory_write_threshold=0.8,
        replan_cooldown_frames=10,
        memory_write_cooldown_frames=10,
    )

    first = policy.decide(0.6, frame_index=100)
    suppressed = policy.decide(0.7, frame_index=105)
    later = policy.decide(0.7, frame_index=111)

    assert first.soft_plan is True
    assert suppressed.should_plan is False
    assert later.soft_plan is True


def test_causal_peak_policy_waits_for_score_drop():
    policy = CausalPeakTransitionPolicy(
        StatefulTransitionPolicy(
            planner_threshold=0.5,
            memory_write_threshold=0.8,
            replan_cooldown_frames=10,
            memory_write_cooldown_frames=10,
        )
    )

    first = policy.decide(0.6, frame_index=10)
    rising = policy.decide(0.7, frame_index=11)
    confirmed_peak = policy.decide(0.4, frame_index=12)

    assert first.should_plan is False
    assert rising.should_plan is False
    assert confirmed_peak.soft_plan is True
    assert confirmed_peak.score == 0.7


def test_canonical_block_features_include_value_delta_mask_and_source():
    df = pd.DataFrame(
        {
            "action": [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 5.0], dtype=np.float32)],
            "state": [np.array([0.5], dtype=np.float32), np.array([1.5], dtype=np.float32)],
        }
    )

    features = build_canonical_block_features(
        df,
        {
            "dataset_name": "demo",
            "include_deltas": True,
            "include_value_mask": True,
            "source_one_hot": {"enabled": True, "names": ["demo", "other"]},
            "blocks": [
                {"name": "action", "key": "action", "dim": 3, "valid": 1.0},
                {"name": "missing", "dim": 2, "valid": 0.0},
                {"name": "constant", "constant": [1.0], "dim": 1, "valid": 1.0},
            ],
        },
    )

    assert features.shape == (2, 20)
    np.testing.assert_allclose(features[0, :6], [1.0, 2.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(features[1, 6:12], [2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(features[0, 12:18], [1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(features[0, 18:], [1.0, 0.0])


def test_online_canonical_feature_builder_matches_dataframe_builder():
    config = {
        "data": {"window_size": 2},
        "features": {
            "mode": "canonical_blocks",
            "dataset_name": "demo",
            "expected_input_dim": 20,
            "include_deltas": True,
            "include_value_mask": True,
            "source_one_hot": {"enabled": True, "names": ["demo", "other"]},
            "blocks": [
                {"name": "action", "key": "action", "dim": 3, "valid": 1.0},
                {"name": "missing", "dim": 2, "valid": 0.0},
                {"name": "constant", "constant": [1.0], "dim": 1, "valid": 1.0},
            ],
        },
    }
    frames = [
        {"action": np.array([1.0, 2.0], dtype=np.float32)},
        {"action": np.array([3.0, 5.0], dtype=np.float32)},
    ]
    df = pd.DataFrame(frames)

    expected = build_canonical_block_features(df, config["features"])
    actual = CanonicalFeatureBuilder(config).build_window(frames)

    np.testing.assert_allclose(actual, expected)


def test_selected_online_feature_builders_match_runtime_input_dim():
    config = load_config(_selected_config_path())
    robomme = CanonicalFeatureBuilder(config, dataset_name="robomme_four_tasks")
    rmbench = CanonicalFeatureBuilder(config, dataset_name="rmbench_9tasks")

    robomme_features = robomme.build_window(
        [
            {
                "action": np.full(7, index, dtype=np.float32),
                "eef_state": np.full(7, index + 0.1, dtype=np.float32),
                "joint_state": np.full(7, index + 0.2, dtype=np.float32),
                "gripper_state": np.full(2, index + 0.3, dtype=np.float32),
            }
            for index in range(32)
        ]
    )
    rmbench_features = rmbench.build_window(
        [
            {
                "left_joint_action": np.full(6, index, dtype=np.float32),
                "left_gripper_action": np.array([index + 0.1], dtype=np.float32),
                "right_joint_action": np.full(6, index + 0.2, dtype=np.float32),
                "right_gripper_action": np.array([index + 0.3], dtype=np.float32),
                "left_endpose": np.full(7, index + 0.4, dtype=np.float32),
                "right_endpose": np.full(7, index + 0.5, dtype=np.float32),
                "left_gripper": np.full(2, index + 0.6, dtype=np.float32),
                "right_gripper": np.full(2, index + 0.7, dtype=np.float32),
            }
            for index in range(32)
        ]
    )

    assert robomme.spec.input_dim == 144
    assert rmbench.spec.input_dim == 144
    assert robomme_features.shape == (32, 144)
    assert rmbench_features.shape == (32, 144)


def test_online_feature_buffer_waits_for_full_window():
    config = load_config(_selected_config_path())
    buffer = OnlineTransitionFeatureBuffer(config, dataset_name="robomme_four_tasks")
    frame = {
        "action": np.zeros(7, dtype=np.float32),
        "eef_state": np.zeros(7, dtype=np.float32),
        "joint_state": np.zeros(7, dtype=np.float32),
        "gripper_state": np.zeros(2, dtype=np.float32),
    }

    for _ in range(31):
        assert buffer.append_and_build(frame) is None

    window = buffer.append_and_build(frame)

    assert window is not None
    assert tuple(window.shape) == (32, 144)


def test_manifest_split_supports_dataset_scoped_ids(tmp_path):
    records = [
        WindowRecord("dataset_a:task/episode_0", None, 0, 0, np.zeros((1, 1), dtype=np.float32), 0, 1, "", None),
        WindowRecord("dataset_a:task/episode_1", None, 0, 0, np.zeros((1, 1), dtype=np.float32), 0, 1, "", None),
    ]
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        """
        {
          "splits": {
            "train": {"dataset_a": ["task/episode_0"]},
            "eval": {"dataset_a": ["task/episode_1"]}
          }
        }
        """
    )

    train_dataset, eval_dataset = split_by_manifest(records, manifest_path, "train", "eval")

    assert [record.trajectory_id for record in train_dataset.records] == ["dataset_a:task/episode_0"]
    assert [record.trajectory_id for record in eval_dataset.records] == ["dataset_a:task/episode_1"]


def test_ssm_transition_trigger_forward_shape():
    model = TransitionTriggerHead(
        input_dim=96,
        type="ssm",
        d_model=256,
        num_layers=2,
        state_dim=32,
        num_heads=8,
        mlp_ratio=4.0,
        max_seq_len=64,
        pooling="last",
        dropout=0.0,
        head_hidden_dim=256,
    )

    logits = model(torch.zeros(3, 24, 96))

    assert tuple(logits.shape) == (3, 1)


def test_transformer_transition_trigger_forward_shape():
    model = TransitionTriggerHead(
        input_dim=98,
        type="transformer",
        d_model=256,
        num_layers=2,
        state_dim=32,
        num_heads=8,
        mlp_ratio=4.0,
        max_seq_len=64,
        pooling="last",
        dropout=0.0,
        head_hidden_dim=256,
    )

    logits = model(torch.zeros(3, 24, 98))

    assert tuple(logits.shape) == (3, 1)


def test_transition_trigger_runtime_loads_package_and_scores(tmp_path):
    config = _runtime_test_config(tmp_path)
    model = TransitionTriggerHead(input_dim=4, **config["model"])
    package_dir = tmp_path / "selected"
    package_dir.mkdir()
    (package_dir / "config.yaml").write_text(
        """
seed: 7
data:
  window_size: 3
features:
  expected_input_dim: 4
model:
  type: transformer
  d_model: 8
  num_layers: 1
  state_dim: 4
  num_heads: 2
  mlp_ratio: 2.0
  max_seq_len: 8
  pooling: last
  head_hidden_dim: 8
  dropout: 0.0
trigger_policy:
  planner_threshold: 0.4
  memory_write_threshold: 0.8
  memory_write_implies_plan: true
        """
    )
    torch.save({"model": model.state_dict(), "input_dim": 4, "config": config}, package_dir / "checkpoint.pt")

    runtime = TransitionTriggerRuntime.from_package(package_dir)
    scores = runtime.score_window(torch.zeros(2, 3, 4))
    decision = runtime.decide_window(torch.zeros(3, 4))

    assert tuple(scores.shape) == (2,)
    assert 0.0 <= decision.score <= 1.0
    assert decision.decision.planner_threshold == 0.4
    assert decision.decision.memory_write_threshold == 0.8


def test_runtime_stateless_decision_rejects_causal_peak_config(tmp_path):
    config = _runtime_test_config(tmp_path)
    config["trigger_policy"]["score_mode"] = "causal_peak"
    model = TransitionTriggerHead(input_dim=4, **config["model"])
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "input_dim": 4, "config": config}, checkpoint)
    runtime = TransitionTriggerRuntime(config=config, checkpoint_path=checkpoint)

    with pytest.raises(ValueError, match="new_session"):
        runtime.decide_window(torch.zeros(3, 4))


def test_transition_trigger_session_uses_causal_peak_policy():
    class FakeRuntime:
        config = {
            "trigger_policy": {
                "score_mode": "causal_peak",
                "planner_threshold": 0.5,
                "memory_write_threshold": 0.8,
                "replan_cooldown_frames": 10,
                "memory_write_cooldown_frames": 10,
                "memory_write_implies_plan": True,
            }
        }

        def new_policy(self):
            return build_transition_policy_from_config(self.config)

        def score_window(self, features):
            return torch.tensor([float(features.item())])

    session = TransitionTriggerSession(FakeRuntime())

    first = session.decide_window(torch.tensor([[0.6]]), frame_index=10)
    rising = session.decide_window(torch.tensor([[0.7]]), frame_index=11)
    confirmed_peak = session.decide_window(torch.tensor([[0.4]]), frame_index=12)

    assert first.decision.should_plan is False
    assert rising.decision.should_plan is False
    assert confirmed_peak.decision.soft_plan is True
    assert confirmed_peak.decision.score == pytest.approx(0.7)


def test_online_session_buffers_features_before_scoring():
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
                "replan_cooldown_frames": 10,
                "memory_write_cooldown_frames": 10,
                "memory_write_implies_plan": True,
            },
        }

        def new_policy(self):
            return build_transition_policy_from_config(self.config)

        def score_window(self, features):
            assert tuple(features.shape) == (2, 1)
            return torch.tensor([0.9])

    online = TransitionTriggerOnlineSession(FakeRuntime(), dataset_name=None)

    assert online.append({"action": [0.0]}, frame_index=10) is None
    output = online.append({"action": [1.0]}, frame_index=11)

    assert output is not None
    assert output.decision.memory_write is True
    assert output.decision.hard_plan is True


def test_transition_trigger_runtime_rejects_wrong_window_shape(tmp_path):
    config = _runtime_test_config(tmp_path)
    model = TransitionTriggerHead(input_dim=4, **config["model"])
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "input_dim": 4, "config": config}, checkpoint)
    runtime = TransitionTriggerRuntime(config=config, checkpoint_path=checkpoint)

    try:
        runtime.score_window(torch.zeros(4, 4))
    except ValueError as exc:
        assert "window_size=3" in str(exc)
    else:
        raise AssertionError("expected wrong window size to fail")


def _runtime_test_config(tmp_path):
    return {
        "data": {"window_size": 3},
        "features": {"expected_input_dim": 4},
        "model": {
            "type": "transformer",
            "d_model": 8,
            "num_layers": 1,
            "state_dim": 4,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "max_seq_len": 8,
            "pooling": "last",
            "head_hidden_dim": 8,
            "dropout": 0.0,
        },
        "trigger_policy": {
            "score_mode": "threshold",
            "planner_threshold": 0.4,
            "memory_write_threshold": 0.8,
            "memory_write_implies_plan": True,
        },
        "outputs": {"run_dir": str(tmp_path)},
    }


def _selected_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "transition_trigger"
        / "configs"
        / "selected"
        / "robomme_rmbench_w32_value_delta_transformer_d512.yaml"
    )
