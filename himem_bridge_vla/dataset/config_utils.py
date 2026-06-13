from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


REQUIRED_POSITIVE_INT_KEYS = (
    "max_action_dim",
    "max_state_dim",
    "max_views",
)


def resolve_dataset_path(raw_path: str | Path, base_dir: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir).expanduser() / path).resolve()


def iter_dataset_entries(config: Mapping[str, Any]) -> Iterator[tuple[Any, Any, Mapping[str, Any]]]:
    data_groups = config.get("data_groups")
    if not isinstance(data_groups, Mapping) or not data_groups:
        raise ValueError("data_groups must be a non-empty mapping")

    for group_name, group_config in data_groups.items():
        if not isinstance(group_config, Mapping) or not group_config:
            raise ValueError(f"data group {group_name!r} must contain datasets")
        for dataset_name, dataset_config in group_config.items():
            if not isinstance(dataset_config, Mapping):
                raise ValueError(f"dataset {group_name}/{dataset_name} must be a mapping")
            yield group_name, dataset_name, dataset_config


def validate_dataset_config_structure(config: Mapping[str, Any]) -> int:
    if not isinstance(config, Mapping):
        raise TypeError("dataset config must be a mapping")

    for key in REQUIRED_POSITIVE_INT_KEYS:
        value = config.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")

    dataset_count = 0
    for group_name, dataset_name, dataset_config in iter_dataset_entries(config):
        dataset_count += 1
        raw_path = dataset_config.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"dataset {group_name}/{dataset_name} has no path")

    return dataset_count


def resolve_dataset_config_paths(config: Mapping[str, Any], base_dir: str | Path) -> dict[str, Any]:
    validate_dataset_config_structure(config)

    resolved_config = deepcopy(dict(config))
    for group_name, dataset_name, _dataset_config in iter_dataset_entries(resolved_config):
        dataset_config = resolved_config["data_groups"][group_name][dataset_name]
        dataset_config["path"] = str(resolve_dataset_path(dataset_config["path"], base_dir))

    return resolved_config
