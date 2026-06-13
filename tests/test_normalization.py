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
