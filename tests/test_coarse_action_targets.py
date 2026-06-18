import numpy as np
import pytest

from himem_bridge_vla.dataset.coarse_actions import build_coarse_action_target


def test_relative_coarse_action_target_sums_motion_and_keeps_last_gripper():
    actions = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    coarse, mask = build_coarse_action_target(
        actions,
        num_plan_steps=2,
        planning_horizon=4,
        action_convention="relative",
        gripper_indices=[-1],
    )

    np.testing.assert_allclose(coarse, np.array([[3.0, 0.0, 1.0], [7.0, 2.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(mask, np.array([True, True]))


def test_coarse_action_target_masks_tail_chunks():
    actions = np.ones((6, 3), dtype=np.float32)

    coarse, mask = build_coarse_action_target(
        actions,
        num_plan_steps=3,
        planning_horizon=6,
        valid_action_count=4,
    )

    np.testing.assert_array_equal(mask, np.array([True, True, False]))
    np.testing.assert_allclose(coarse[2], np.zeros(3, dtype=np.float32))


def test_absolute_delta_uses_chunk_endpoint_delta_for_motion():
    actions = np.array(
        [
            [1.0, 1.0, 0.0],
            [3.0, 4.0, 1.0],
            [10.0, 2.0, 1.0],
            [12.0, 7.0, 0.0],
        ],
        dtype=np.float32,
    )

    coarse, _ = build_coarse_action_target(
        actions,
        num_plan_steps=2,
        planning_horizon=4,
        action_convention="absolute_delta",
        gripper_indices=[2],
    )

    np.testing.assert_allclose(coarse, np.array([[2.0, 3.0, 1.0], [2.0, 5.0, 0.0]], dtype=np.float32))


def test_coarse_action_target_rejects_nondivisible_horizon():
    with pytest.raises(ValueError, match="divisible"):
        build_coarse_action_target(np.ones((5, 3), dtype=np.float32), num_plan_steps=2, planning_horizon=5)
