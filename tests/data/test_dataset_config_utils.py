from __future__ import annotations

from pathlib import Path

import pytest

from himem_bridge_vla.dataset.config_utils import (
    resolve_dataset_config_paths,
    resolve_dataset_path,
    validate_dataset_config_structure,
)


def minimal_dataset_config(path: str = "datasets/custom_sim/lerobot") -> dict:
    return {
        "max_action_dim": 24,
        "max_state_dim": 24,
        "max_views": 3,
        "data_groups": {
            "custom_sim": {
                "CustomSimulation": {
                    "path": path,
                    "view_map": {"image_1": "observation.images.image"},
                }
            }
        },
    }


def test_resolve_dataset_path_uses_explicit_base_dir(tmp_path):
    resolved = resolve_dataset_path("data/example", tmp_path / "himem_bridge_vla")

    assert resolved == (tmp_path / "himem_bridge_vla" / "data" / "example").resolve()


def test_resolve_dataset_config_paths_does_not_mutate_input(tmp_path):
    config = minimal_dataset_config()

    resolved = resolve_dataset_config_paths(config, tmp_path / "himem_bridge_vla")

    original_path = config["data_groups"]["custom_sim"]["CustomSimulation"]["path"]
    resolved_path = resolved["data_groups"]["custom_sim"]["CustomSimulation"]["path"]
    assert original_path == "datasets/custom_sim/lerobot"
    assert Path(resolved_path) == (tmp_path / "himem_bridge_vla" / "datasets" / "custom_sim" / "lerobot").resolve()


def test_validate_dataset_config_structure_reports_dataset_count():
    assert validate_dataset_config_structure(minimal_dataset_config()) == 1


def test_validate_dataset_config_structure_rejects_missing_path():
    config = minimal_dataset_config()
    del config["data_groups"]["custom_sim"]["CustomSimulation"]["path"]

    with pytest.raises(ValueError, match="has no path"):
        validate_dataset_config_structure(config)


def test_validate_dataset_config_structure_rejects_absolute_paths():
    config = minimal_dataset_config(str(Path("/").joinpath("datasets", "custom_sim", "lerobot")))

    with pytest.raises(ValueError, match="project-relative"):
        validate_dataset_config_structure(config)


def test_validate_dataset_config_structure_rejects_parent_relative_paths():
    config = minimal_dataset_config("../datasets/custom_sim/lerobot")

    with pytest.raises(ValueError, match="inside the project"):
        validate_dataset_config_structure(config)


def test_resolve_dataset_config_paths_resolves_optional_boundary_path(tmp_path):
    config = minimal_dataset_config()
    config["data_groups"]["custom_sim"]["CustomSimulation"]["boundary_path"] = "datasets/custom_sim/boundaries.jsonl"

    resolved = resolve_dataset_config_paths(config, tmp_path / "repo")

    boundary_path = resolved["data_groups"]["custom_sim"]["CustomSimulation"]["boundary_path"]
    assert Path(boundary_path) == (tmp_path / "repo" / "datasets" / "custom_sim" / "boundaries.jsonl").resolve()
