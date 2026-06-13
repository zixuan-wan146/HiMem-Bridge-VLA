from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Any


POSITIVE_INT_KEYS = (
    "max_steps",
    "log_interval",
    "ckpt_interval",
    "batch_size",
    "horizon",
    "per_action_dim",
    "state_dim",
)

NON_NEGATIVE_INT_KEYS = (
    "num_workers",
    "warmup_steps",
)

POSITIVE_FLOAT_KEYS = (
    "lr",
    "grad_clip_norm",
)

NON_NEGATIVE_FLOAT_KEYS = (
    "weight_decay",
    "dropout",
)


def validate_training_config(
    config: Mapping[str, Any],
    *,
    cuda_available: bool | None = None,
    path_exists: Callable[[Path], bool] | None = None,
) -> None:
    path_exists = _default_path_exists if path_exists is None else path_exists

    dataset_config_path = config.get("dataset_config_path")
    if not dataset_config_path:
        raise ValueError("--dataset_config_path is required")
    dataset_config = Path(str(dataset_config_path))
    if not path_exists(dataset_config):
        raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")

    dataset_config_base_dir = config.get("dataset_config_base_dir")
    if dataset_config_base_dir and not path_exists(Path(str(dataset_config_base_dir))):
        raise FileNotFoundError(f"Dataset config base directory not found: {dataset_config_base_dir}")

    bridge_himem_config = config.get("bridge_himem_config")
    if bridge_himem_config and not path_exists(Path(str(bridge_himem_config))):
        raise FileNotFoundError(f"Bridge-HiMem config file not found: {bridge_himem_config}")

    for key in POSITIVE_INT_KEYS:
        value = _as_int(config.get(key, 0), f"--{key}")
        if value <= 0:
            raise ValueError(f"--{key} must be positive, got {value}")

    for key in NON_NEGATIVE_INT_KEYS:
        if key in config:
            value = _as_int(config[key], f"--{key}")
            if value < 0:
                raise ValueError(f"--{key} must be non-negative, got {value}")

    for key in POSITIVE_FLOAT_KEYS:
        if key in config:
            value = _as_float(config[key], f"--{key}")
            if value <= 0:
                raise ValueError(f"--{key} must be positive, got {value}")

    for key in NON_NEGATIVE_FLOAT_KEYS:
        if key in config:
            value = _as_float(config[key], f"--{key}")
            if value < 0:
                raise ValueError(f"--{key} must be non-negative, got {value}")

    device = str(config.get("device", "cuda"))
    if device.startswith("cuda"):
        if cuda_available is None:
            cuda_available = _torch_cuda_available()
        if not cuda_available:
            raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")

    resume = bool(config.get("resume", False))
    resume_path = config.get("resume_path")
    if resume != bool(resume_path):
        raise ValueError("Inconsistent resume configuration: --resume and --resume_path must be set together.")
    if bool(config.get("resume_pretrain", False)) and not resume:
        raise ValueError("--resume_pretrain requires --resume and --resume_path.")
    if resume and not path_exists(Path(str(resume_path))):
        raise FileNotFoundError(f"Resume checkpoint path not found: {resume_path}")


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got {value!r}") from exc


def _default_path_exists(path: Path) -> bool:
    return path.exists()


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())
