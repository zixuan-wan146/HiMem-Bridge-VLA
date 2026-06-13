import pytest


def test_normalization_stats_supports_short_model_action_dim():
    torch = pytest.importorskip("torch")
    normalization = pytest.importorskip("himem_bridge_vla.utils.normalization")

    stats = {
        "robot": {
            "observation.state": {"min": [0.0] * 7, "max": [1.0] * 7},
            "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
        }
    }
    normalizer = normalization.NormalizationStats(stats)

    state = normalizer.normalize_state(torch.full((1, 7), 0.5))
    action = normalizer.denormalize_action(torch.zeros(2, 7))

    assert state.shape == (1, 7)
    assert action.shape == (2, 7)


def test_normalization_stats_selects_robot_key_for_multi_robot_stats():
    torch = pytest.importorskip("torch")
    normalization = pytest.importorskip("himem_bridge_vla.utils.normalization")

    stats = {
        "robot_a": {
            "observation.state": {"min": [0.0], "max": [1.0]},
            "action": {"min": [0.0], "max": [10.0]},
        },
        "robot_b": {
            "observation.state": {"min": [10.0], "max": [20.0]},
            "action": {"min": [-2.0], "max": [2.0]},
        },
    }
    normalizer = normalization.NormalizationStats(stats, target_dim=1, robot_key="robot_b")

    state = normalizer.normalize_state(torch.tensor([[15.0]]))
    action = normalizer.denormalize_action(torch.tensor([[0.0]]), robot_key="robot_a")

    assert state.item() == pytest.approx(0.0)
    assert action.item() == pytest.approx(5.0)


def test_normalization_stats_requires_robot_key_at_use_for_multi_robot_stats():
    torch = pytest.importorskip("torch")
    normalization = pytest.importorskip("himem_bridge_vla.utils.normalization")

    stats = {
        "robot_a": {
            "observation.state": {"min": [0.0], "max": [1.0]},
            "action": {"min": [0.0], "max": [1.0]},
        },
        "robot_b": {
            "observation.state": {"min": [0.0], "max": [1.0]},
            "action": {"min": [0.0], "max": [1.0]},
        },
    }

    normalizer = normalization.NormalizationStats(stats, target_dim=1)

    with pytest.raises(ValueError, match="robot_key is required"):
        normalizer.normalize_state(torch.tensor([[0.5]]))
