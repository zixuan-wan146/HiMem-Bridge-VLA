from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def test_train_help_does_not_require_training_runtime():
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--dataset_config_path" in result.stdout


def test_build_training_config_uses_defaults_and_cli_overrides():
    train_script = _load_train_script()
    args = train_script.build_arg_parser().parse_args(
        [
            "--batch_size",
            "3",
            "--dataset_config_path",
            "configs/datasets/simulation.yaml",
            "--save_dir",
            "run_outputs/training/from_cli",
        ]
    )

    config = train_script.build_training_config(args)

    assert config["batch_size"] == 3
    assert config["dataset_config_path"] == "configs/datasets/simulation.yaml"
    assert config["save_dir"] == "run_outputs/training/from_cli"


def test_build_param_groups_rejects_model_with_no_trainable_parameters():
    train_script = _load_train_script()

    with pytest.raises(ValueError, match="No trainable parameters"):
        train_script.build_param_groups(_FrozenModel(), wd=0.01)


def test_unwrap_training_model_uses_accelerator_when_available(monkeypatch):
    train_script = _load_train_script()
    wrapper = object()
    unwrapped = object()

    class FakeAccelerator:
        def unwrap_model(self, model):
            assert model is wrapper
            return unwrapped

    monkeypatch.setattr(train_script, "accelerator", FakeAccelerator())

    assert train_script.unwrap_training_model(wrapper) is unwrapped


def test_validate_batch_image_masks_rejects_empty_sample():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()
    train_script.torch = torch

    image_masks = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="no active image"):
        train_script.validate_batch_image_masks(image_masks, step=3)


def test_compute_bridge_auxiliary_loss_uses_boundary_and_progress_labels():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()

    model = _ModelWithBridgeOutput(
        boundary_logits=torch.tensor([[0.0], [2.0]]),
        progress_logits=torch.tensor([[0.0], [2.0]]),
    )
    batch = {
        "boundary": torch.tensor([0.0, 1.0]),
        "progress": torch.tensor([0.5, 1.0]),
    }

    loss, metrics = train_script.compute_bridge_auxiliary_loss(
        model,
        batch,
        {"boundary_loss_weight": 1.0, "progress_loss_weight": 0.2},
    )

    assert loss is not None
    assert loss.item() > 0
    assert "boundary_loss" in metrics
    assert "progress_loss" in metrics
    assert "bridge_aux_loss" not in metrics


def test_custom_collate_includes_action_segment_targets():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()
    train_script.torch = torch

    item = {
        "prompt": "pick",
        "images": torch.zeros(1, 3, 4, 4),
        "state": torch.zeros(2),
        "action": torch.zeros(2, 2),
        "action_mask": torch.ones(2, 2, dtype=torch.bool),
        "image_mask": torch.ones(1, dtype=torch.bool),
        "embodiment_id": torch.tensor(0),
        "planner_prompt": "pick",
        "planner_images": torch.zeros(1, 3, 4, 4),
        "planner_image_mask": torch.ones(1, dtype=torch.bool),
        "planner_state": torch.zeros(2),
        "action_segments": torch.zeros(3, 4, 2),
        "action_segment_mask": torch.tensor([True, True, False]),
        "plan_active_mask": torch.tensor([False, True, True]),
        "plan_consumed_steps": torch.tensor(4),
        "plan_consumed_tokens": torch.tensor(1),
        "plan_residual_steps": torch.tensor(0),
    }

    batch = train_script.custom_collate_fn([item, item])

    assert tuple(batch["action_segments"].shape) == (2, 3, 4, 2)
    assert tuple(batch["action_segment_mask"].shape) == (2, 3)
    assert tuple(batch["planner_states"].shape) == (2, 2)
    assert tuple(batch["plan_active_mask"].shape) == (2, 3)


def test_compute_coarse_planner_loss_uses_cached_model_output():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()

    model = _ModelWithPlannerOutput(predicted_latents=torch.zeros(2, 3, 5))
    batch = {
        "action_segments": torch.ones(2, 3, 4, 2),
        "action_segment_mask": torch.ones(2, 3, dtype=torch.bool),
    }

    loss, metrics = train_script.compute_coarse_planner_loss(
        model,
        batch,
        {
            "coarse_planner_loss_weight": 0.5,
            "coarse_planner_gripper_indices": [],
        },
        segment_autoencoder=_FakeSegmentAutoencoder(latent_dim=5),
    )

    assert loss is not None
    assert loss.item() > 0
    assert "coarse_planner_loss" in metrics
    assert "coarse_planner_loss_weighted" in metrics


def test_build_action_segment_dataset_config_returns_none_when_disabled():
    train_script = _load_train_script()

    assert train_script.build_action_segment_dataset_config({"coarse_planner_enabled": False}) is None


def test_build_action_segment_dataset_config_maps_planner_fields():
    train_script = _load_train_script()

    config = train_script.build_action_segment_dataset_config(
        {
            "coarse_planner_enabled": True,
            "coarse_planner_num_plan_steps": 4,
            "coarse_planner_planning_horizon": 16,
            "coarse_planner_action_dim": 7,
            "coarse_planner_execution_horizon": 8,
            "coarse_planner_suffix_stride_tokens": 2,
        }
    )

    assert config == {
        "enabled": True,
        "num_plan_steps": 4,
        "planning_horizon": 16,
        "action_dim": 7,
        "execution_horizon": 8,
        "suffix_stride_tokens": 2,
    }


class _FrozenParam:
    requires_grad = False

    def dim(self) -> int:
        return 2


class _FrozenModel:
    def named_parameters(self):
        return [("weight", _FrozenParam())]


class _BridgeOutput:
    def __init__(self, *, boundary_logits, progress_logits):
        self.boundary_logits = boundary_logits
        self.progress_logits = progress_logits


class _ModelWithBridgeOutput:
    def __init__(self, *, boundary_logits, progress_logits):
        self.last_bridge_output = _BridgeOutput(
            boundary_logits=boundary_logits,
            progress_logits=progress_logits,
        )


class _PlannerOutput:
    def __init__(self, *, predicted_latents):
        self.predicted_latents = predicted_latents


class _ModelWithPlannerOutput:
    def __init__(self, *, predicted_latents):
        self.last_coarse_planner_output = _PlannerOutput(predicted_latents=predicted_latents)


class _FakeSegmentAutoencoder:
    def __init__(self, *, latent_dim):
        self.latent_dim = latent_dim

    def encode(self, action_segments):
        return action_segments.new_ones((*action_segments.shape[:2], self.latent_dim))

    def decode(self, predicted_latents):
        return predicted_latents.new_zeros((*predicted_latents.shape[:2], 4, 2))


def _load_train_script():
    spec = importlib.util.spec_from_file_location("himem_train_script_for_tests", TRAIN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
