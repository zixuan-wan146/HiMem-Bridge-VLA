from __future__ import annotations

from pathlib import Path

import pytest

from himem_bridge_vla.training_config import validate_training_config


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
