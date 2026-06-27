#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from himem_bridge_vla.path_utils import find_repo_root


REPO_ROOT_FOR_IMPORTS = find_repo_root(__file__)
SRC_ROOT_FOR_IMPORTS = REPO_ROOT_FOR_IMPORTS / "src"
for import_root in (REPO_ROOT_FOR_IMPORTS, SRC_ROOT_FOR_IMPORTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from himem_bridge_vla.dataset.config_utils import resolve_dataset_config_paths, validate_dataset_config_structure  # noqa: E402
from himem_bridge_vla.dataset.validation import validate_configured_datasets  # noqa: E402
from himem_bridge_vla.path_utils import display_project_path, project_path  # noqa: E402


def load_dataset_config(path: Path) -> dict:
    spec = importlib.util.find_spec("yaml")
    if spec is None:
        raise RuntimeError("PyYAML is required to load dataset config YAML")

    import yaml  # type: ignore[import-not-found]

    with path.open("r") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"dataset config must contain a mapping: {path}")
    validate_dataset_config_structure(payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HiMem-Bridge-VLA simulation training dataset structure.")
    parser.add_argument("--dataset-config", default="configs/datasets/simulation.yaml", help="Dataset YAML config to validate.")
    parser.add_argument(
        "--dataset-base-dir",
        default=".",
        help="Base directory for relative dataset paths in the dataset config.",
    )
    parser.add_argument(
        "--no-require-videos",
        action="store_true",
        help="Do not fail when expected video files are missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = find_repo_root(__file__)
    config_path = project_path(args.dataset_config, repo_root, label="--dataset-config")
    base_dir = project_path(args.dataset_base_dir, repo_root, label="--dataset-base-dir")

    try:
        config = load_dataset_config(config_path)
        resolved_config = resolve_dataset_config_paths(config, base_dir)
        issues = validate_configured_datasets(
            resolved_config,
            base_dir,
            require_videos=not args.no_require_videos,
        )
    except Exception as exc:
        print(f"[FAIL] dataset: {exc}", file=sys.stderr)
        return 1

    if not issues:
        print(f"[OK] dataset: {display_project_path(config_path, repo_root)} training dataset structure is valid")
        return 0

    for issue in issues:
        print(f"[{issue.level}] dataset: {display_project_path(issue.path, repo_root)}: {issue.message}")
    return 1 if any(issue.level == "FAIL" for issue in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
