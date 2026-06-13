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


class _FrozenParam:
    requires_grad = False

    def dim(self) -> int:
        return 2


class _FrozenModel:
    def named_parameters(self):
        return [("weight", _FrozenParam())]


def _load_train_script():
    spec = importlib.util.spec_from_file_location("himem_train_script_for_tests", TRAIN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
