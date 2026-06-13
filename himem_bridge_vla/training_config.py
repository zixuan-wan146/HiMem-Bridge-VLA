from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Any

from .path_utils import normalize_project_relative_path, project_path


TRAINING_DEFAULTS: dict[str, Any] = {
    "device": "cuda",
    "run_name": "default_run",
    "vlm_name": "OpenGVLab/InternVL3-1B",
    "action_head": "flowmatching",
    "bridge_himem_config": None,
    "seed": None,
    "deterministic": False,
    "return_cls_only": False,
    "disable_wandb": False,
    "disable_swanlab": False,
    "dataset_type": "simulation",
    "dataset_config_path": None,
    "dataset_config_base_dir": ".",
    "cache_dir": "run_outputs/training_data_cache",
    "image_size": 448,
    "binarize_gripper": False,
    "use_augmentation": False,
    "lr": 1e-5,
    "batch_size": 16,
    "max_steps": 600,
    "warmup_steps": 300,
    "grad_clip_norm": 1.0,
    "weight_decay": 1e-5,
    "log_interval": 10,
    "ckpt_interval": 10,
    "save_dir": "checkpoints",
    "resume": False,
    "resume_path": None,
    "resume_pretrain": False,
    "finetune_vlm": False,
    "finetune_action_head": False,
    "per_action_dim": 7,
    "state_dim": 7,
    "horizon": 16,
    "num_layers": 8,
    "num_workers": 4,
    "dropout": 0.0,
    "boundary_loss_weight": 1.0,
    "progress_loss_weight": 0.2,
}


INPUT_PATH_KEYS = (
    "dataset_config_path",
    "dataset_config_base_dir",
    "bridge_himem_config",
    "resume_path",
)

OUTPUT_PATH_KEYS = (
    "save_dir",
    "cache_dir",
)

METADATA_PATH_KEYS = (
    "repo_root",
    "training_config_path",
    "bridge_himem_config_path",
)


def default_training_config(repo_root: str | Path | None = None) -> dict[str, Any]:
    config = dict(TRAINING_DEFAULTS)
    return config


def load_training_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to load training YAML configs") from exc

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    return dict(loaded)


def merge_training_config(
    defaults: Mapping[str, Any],
    file_config: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(defaults)
    if file_config:
        merged.update({key: value for key, value in file_config.items() if value is not None})
    if cli_overrides:
        merged.update({key: value for key, value in cli_overrides.items() if value is not None})
    return merged


def resolve_training_config_paths(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    resolved = dict(config)
    for key in (*INPUT_PATH_KEYS, *OUTPUT_PATH_KEYS, *METADATA_PATH_KEYS):
        value = resolved.get(key)
        if value in (None, ""):
            continue
        resolved[key] = normalize_project_relative_path(value, repo_root, label=f"--{key}")
    return resolved


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
    "boundary_loss_weight",
    "progress_loss_weight",
)


def validate_training_config(
    config: Mapping[str, Any],
    *,
    cuda_available: bool | None = None,
    path_exists: Callable[[Path], bool] | None = None,
    repo_root: str | Path | None = None,
) -> None:
    path_exists = _default_path_exists if path_exists is None else path_exists

    dataset_config_path = config.get("dataset_config_path")
    if not dataset_config_path:
        raise ValueError("--dataset_config_path is required")
    dataset_config = _validation_path(dataset_config_path, repo_root, label="--dataset_config_path")
    if not path_exists(dataset_config):
        raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")

    dataset_config_base_dir = config.get("dataset_config_base_dir")
    if dataset_config_base_dir and not path_exists(
        _validation_path(dataset_config_base_dir, repo_root, label="--dataset_config_base_dir")
    ):
        raise FileNotFoundError(f"Dataset config base directory not found: {dataset_config_base_dir}")

    bridge_himem_config = config.get("bridge_himem_config")
    if bridge_himem_config and not path_exists(
        _validation_path(bridge_himem_config, repo_root, label="--bridge_himem_config")
    ):
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

    dropout = _as_float(config.get("dropout", 0.0), "--dropout")
    if dropout > 1:
        raise ValueError(f"--dropout must be <= 1, got {dropout}")

    warmup_steps = _as_int(config.get("warmup_steps", 0), "--warmup_steps")
    max_steps = _as_int(config.get("max_steps", 0), "--max_steps")
    if warmup_steps > max_steps:
        raise ValueError(f"--warmup_steps must be <= --max_steps, got {warmup_steps} > {max_steps}")

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
    if resume and not path_exists(_validation_path(resume_path, repo_root, label="--resume_path")):
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


def _validation_path(value: Any, repo_root: str | Path | None, *, label: str) -> Path:
    if repo_root is None:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            raise ValueError(f"{label} must be project-relative, got {value!r}")
        return path
    return project_path(str(value), repo_root, label=label)
