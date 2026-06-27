#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.training_config import (  # noqa: E402
    default_training_config,
    load_training_config,
    merge_training_config,
    resolve_training_config_paths,
    validate_training_config,
)
from himem_bridge_vla.experiment_config import resolve_experiment_config  # noqa: E402
from himem_bridge_vla.path_utils import display_project_path  # noqa: E402

PROFILE_PATH_KEYS = (
    "dataset_config_path",
    "dataset_config_base_dir",
    "bridge_himem_config",
    "progress_planner_checkpoint",
    "resume_path",
    "save_dir",
    "cache_dir",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate HiMem-Bridge-VLA training YAML profiles.")
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help=(
            "Specific training YAML profiles to validate. Defaults to configs/stage1/*.yaml, "
            "configs/training/*.yaml, and configs/training_templates/*.yaml."
        ),
    )
    args = parser.parse_args()

    paths = args.configs or (
        sorted((REPO_ROOT / "configs" / "stage1").glob("*.yaml"))
        + sorted((REPO_ROOT / "configs" / "training").glob("*.yaml"))
        + sorted((REPO_ROOT / "configs" / "training_templates").glob("*.yaml"))
    )
    if not paths:
        print("validated 0 training config(s); configs/stage1, configs/training, and configs/training_templates are empty")
        return 0

    for path in paths:
        config_path = _resolve_config_path(path)
        file_config = load_training_config(config_path)
        validate_profile_paths_are_relative(file_config, config_path)
        config = merge_training_config(default_training_config(REPO_ROOT), file_config=file_config)
        config = resolve_training_config_paths(config, REPO_ROOT)
        config = resolve_experiment_config(config)
        validate_training_config(config, cuda_available=True, repo_root=REPO_ROOT)
        print(
            f"{display_project_path(config_path, REPO_ROOT)}: run={config['run_name']} dataset={config['dataset_config_path']} "
            f"bridge={config.get('bridge_himem_config') or 'none'} max_steps={config['max_steps']}"
        )

    return 0


def _resolve_config_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPO_ROOT / candidate


def validate_profile_paths_are_relative(config: dict, config_path: Path) -> None:
    for key in PROFILE_PATH_KEYS:
        value = config.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            continue
        if Path(value).expanduser().is_absolute():
            raise ValueError(
                f"{display_project_path(config_path, REPO_ROOT)}: {key} must be project-relative, got {value!r}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
