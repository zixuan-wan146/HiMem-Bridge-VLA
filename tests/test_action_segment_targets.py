import numpy as np
import pytest

from himem_bridge_vla.dataset.action_segments import (
    build_action_segment_target,
    build_plan_active_mask,
    plan_consumption_from_steps,
    plan_suffix_token_offsets,
    token_span_steps,
)


def test_action_segment_target_keeps_full_chunk_trajectory():
    actions = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    segments, mask = build_action_segment_target(actions, num_plan_steps=2, planning_horizon=4)

    np.testing.assert_allclose(segments[0], actions[:2])
    np.testing.assert_allclose(segments[1], actions[2:4])
    np.testing.assert_array_equal(mask, np.array([True, True]))


def test_action_segment_target_masks_tail_chunks():
    actions = np.ones((6, 3), dtype=np.float32)

    segments, mask = build_action_segment_target(
        actions,
        num_plan_steps=3,
        planning_horizon=6,
        valid_action_count=4,
    )

    np.testing.assert_array_equal(mask, np.array([True, True, False]))
    np.testing.assert_allclose(segments[2], np.zeros((2, 3), dtype=np.float32))


def test_action_segment_target_rejects_nondivisible_horizon():
    with pytest.raises(ValueError, match="divisible"):
        build_action_segment_target(np.ones((5, 3), dtype=np.float32), num_plan_steps=2, planning_horizon=5)


def test_plan_consumption_uses_cumulative_steps():
    span = token_span_steps(planning_horizon=64, num_plan_steps=8)

    assert span == 8
    assert plan_consumption_from_steps(4, span_steps=span) == (0, 4)
    assert plan_consumption_from_steps(8, span_steps=span) == (1, 0)


def test_plan_active_mask_keeps_suffix_from_consumed_token():
    mask = build_plan_active_mask(num_plan_steps=8, consumed_tokens=2)

    np.testing.assert_array_equal(mask, np.array([False, False, True, True, True, True, True, True]))


def test_plan_suffix_offsets_follow_execution_horizon():
    offsets = plan_suffix_token_offsets(
        num_plan_steps=8,
        planning_horizon=64,
        execution_horizon=16,
    )

    assert offsets == [0, 2, 4, 6]
