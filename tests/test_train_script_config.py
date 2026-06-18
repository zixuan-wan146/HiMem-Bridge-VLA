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


def test_custom_collate_includes_coarse_planner_targets():
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
        "coarse_actions": torch.zeros(3, 2),
        "coarse_action_mask": torch.tensor([True, True, False]),
    }

    batch = train_script.custom_collate_fn([item, item])

    assert tuple(batch["coarse_actions"].shape) == (2, 3, 2)
    assert tuple(batch["coarse_action_mask"].shape) == (2, 3)


def test_compute_coarse_planner_loss_uses_cached_model_output():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()

    model = _ModelWithPlannerOutput(coarse_actions=torch.zeros(2, 3, 2))
    batch = {
        "coarse_actions": torch.ones(2, 3, 2),
        "coarse_action_mask": torch.ones(2, 3, dtype=torch.bool),
    }

    loss, metrics = train_script.compute_coarse_planner_loss(
        model,
        batch,
        {
            "coarse_planner_loss_weight": 0.5,
            "coarse_planner_gripper_indices": [],
            "coarse_planner_smoothness_weight": 0.0,
        },
    )

    assert loss is not None
    assert loss.item() > 0
    assert "coarse_planner_loss" in metrics
    assert "coarse_planner_loss_weighted" in metrics


def test_build_coarse_action_dataset_config_returns_none_when_disabled():
    train_script = _load_train_script()

    assert train_script.build_coarse_action_dataset_config({"coarse_planner_enabled": False}) is None


def test_build_coarse_action_dataset_config_maps_planner_fields():
    train_script = _load_train_script()

    config = train_script.build_coarse_action_dataset_config(
        {
            "coarse_planner_enabled": True,
            "coarse_planner_num_plan_steps": 4,
            "coarse_planner_planning_horizon": 16,
            "coarse_planner_action_dim": 7,
            "coarse_planner_action_convention": "relative",
            "coarse_planner_motion_indices": [0, 1],
            "coarse_planner_gripper_indices": [6],
        }
    )

    assert config == {
        "enabled": True,
        "num_plan_steps": 4,
        "planning_horizon": 16,
        "action_dim": 7,
        "action_convention": "relative",
        "motion_indices": [0, 1],
        "gripper_indices": [6],
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
    def __init__(self, *, coarse_actions):
        self.coarse_actions = coarse_actions


class _ModelWithPlannerOutput:
    def __init__(self, *, coarse_actions):
        self.last_coarse_planner_output = _PlannerOutput(coarse_actions=coarse_actions)


def _load_train_script():
    spec = importlib.util.spec_from_file_location("himem_train_script_for_tests", TRAIN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
