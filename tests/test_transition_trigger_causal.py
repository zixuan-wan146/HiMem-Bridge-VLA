from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from transition_trigger.data import (
    WindowRecord,
    _build_records_for_trajectory,
    build_canonical_block_features,
    resolve_label_window,
    split_by_manifest,
)
from transition_trigger.metrics import match_events
from transition_trigger.model import TransitionTriggerHead
from transition_trigger.trigger_policy import decide_transition_actions, decide_transition_actions_from_config


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
