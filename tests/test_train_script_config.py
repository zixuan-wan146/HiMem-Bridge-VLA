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


def test_build_training_config_merges_yaml_and_cli_overrides():
    train_script = _load_train_script()
    config_path = "configs/training/calvin_stage1.yaml"
    args = train_script.build_arg_parser().parse_args(
        [
            "--config",
            config_path,
            "--batch_size",
            "3",
            "--save_dir",
            "run_outputs/training/from_cli",
        ]
    )

    config = train_script.build_training_config(args)

    assert config["batch_size"] == 3
    assert config["dataset_config_path"] == "configs/datasets/calvin.yaml"
    assert config["save_dir"] == "run_outputs/training/from_cli"
    assert config["training_config_path"] == config_path


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


def _load_train_script():
    spec = importlib.util.spec_from_file_location("himem_train_script_for_tests", TRAIN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
