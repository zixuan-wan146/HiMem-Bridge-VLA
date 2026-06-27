from __future__ import annotations

from pathlib import Path

import pytest

from himem_bridge_vla.training_config import validate_training_config
from himem_bridge_vla.training_config import (
    default_training_config,
    load_training_config,
    merge_training_config,
    resolve_training_config_paths,
)


def valid_config() -> dict:
    return {
        "dataset_config_path": "configs/datasets/simulation.yaml",
        "max_steps": 10,
        "log_interval": 1,
        "ckpt_interval": 5,
        "batch_size": 2,
        "horizon": 2,
        "per_action_dim": 3,
        "state_dim": 3,
        "num_workers": 0,
        "warmup_steps": 0,
        "lr": 1e-5,
        "grad_clip_norm": 1.0,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "device": "cpu",
        "resume": False,
        "resume_path": None,
    }


def existing_paths(*paths: str):
    existing = {Path(path) for path in paths}
    return lambda path: path in existing


def test_validate_training_config_accepts_valid_cpu_config():
    validate_training_config(
        valid_config(),
        cuda_available=False,
        path_exists=existing_paths("configs/datasets/simulation.yaml"),
    )


def test_validate_training_config_requires_dataset_config_path():
    config = valid_config()
    config["dataset_config_path"] = ""

    with pytest.raises(ValueError, match="dataset_config_path"):
        validate_training_config(config, cuda_available=False, path_exists=lambda path: True)


def test_validate_training_config_rejects_missing_dataset_config():
    with pytest.raises(FileNotFoundError, match="Dataset config file not found"):
        validate_training_config(valid_config(), cuda_available=False, path_exists=lambda path: False)


def test_validate_training_config_can_skip_external_path_checks_for_profile_validation():
    validate_training_config(
        valid_config(),
        cuda_available=False,
        path_exists=lambda path: False,
        validate_external_paths=False,
    )


def test_validate_training_config_rejects_missing_dataset_config_base_dir():
    config = valid_config()
    config["dataset_config_base_dir"] = "himem_bridge_vla"

    with pytest.raises(FileNotFoundError, match="Dataset config base directory not found"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_validate_training_config_rejects_non_positive_ints():
    config = valid_config()
    config["batch_size"] = 0

    with pytest.raises(ValueError, match="batch_size"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_validate_training_config_rejects_dropout_above_one():
    config = valid_config()
    config["dropout"] = 1.5

    with pytest.raises(ValueError, match="dropout"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_validate_training_config_rejects_warmup_longer_than_training():
    config = valid_config()
    config["warmup_steps"] = 11
    config["max_steps"] = 10

    with pytest.raises(ValueError, match="warmup_steps"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_validate_training_config_rejects_cuda_when_unavailable():
    config = valid_config()
    config["device"] = "cuda:0"

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_validate_training_config_requires_resume_pair():
    config = valid_config()
    config["resume"] = True
    config["resume_path"] = None

    with pytest.raises(ValueError, match="resume"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_validate_training_config_rejects_missing_resume_path():
    config = valid_config()
    config["resume"] = True
    config["resume_path"] = "checkpoints/step_1"

    with pytest.raises(FileNotFoundError, match="Resume checkpoint path not found"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_validate_training_config_requires_resume_for_resume_pretrain():
    config = valid_config()
    config["resume_pretrain"] = True

    with pytest.raises(ValueError, match="resume_pretrain"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_validate_training_config_rejects_random_frozen_progress_planner():
    config = valid_config()
    config.update(
        {
            "progress_planner_enabled": True,
            "progress_planner_checkpoint": None,
            "finetune_progress_planner": False,
        }
    )

    with pytest.raises(ValueError, match="random frozen progress planner"):
        validate_training_config(
            config,
            cuda_available=False,
            path_exists=existing_paths("configs/datasets/simulation.yaml"),
        )


def test_training_yaml_config_merges_with_cli_overrides(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_config_path: configs/datasets/simulation.yaml",
                "batch_size: 8",
                "disable_swanlab: true",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_training_config(config_path)
    merged = merge_training_config(
        default_training_config(tmp_path),
        file_config=loaded,
        cli_overrides={"batch_size": 4},
    )

    assert merged["dataset_config_path"] == "configs/datasets/simulation.yaml"
    assert merged["batch_size"] == 4
    assert merged["disable_swanlab"] is True


def test_resolve_training_config_paths_keeps_dataset_paths_project_relative(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    workdir = tmp_path / "outside"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    resolved = resolve_training_config_paths(
        {
            "dataset_config_path": "configs/datasets/simulation.yaml",
            "dataset_config_base_dir": ".",
        },
        repo_root,
    )

    assert resolved["dataset_config_path"] == "configs/datasets/simulation.yaml"
    assert resolved["dataset_config_base_dir"] == "."


def test_resolve_training_config_paths_keeps_outputs_project_relative(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    workdir = tmp_path / "outside"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    resolved = resolve_training_config_paths(
        {
            "save_dir": "run_outputs/training/stage1",
            "cache_dir": "checkpoints/cache",
        },
        repo_root,
    )

    assert resolved["save_dir"] == "run_outputs/training/stage1"
    assert resolved["cache_dir"] == "checkpoints/cache"


def test_resolve_training_config_paths_rejects_absolute_paths(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError, match="project-relative"):
        resolve_training_config_paths(
            {
                "save_dir": str(tmp_path / "outside"),
            },
            repo_root,
        )
