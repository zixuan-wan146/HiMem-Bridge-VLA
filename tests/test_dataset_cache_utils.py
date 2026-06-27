from pathlib import Path

from himem_bridge_vla.dataset.cache_utils import dataset_cache_namespace
from himem_bridge_vla.dataset.cache_utils import default_dataset_cache_dir


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_dataset_cache_dir_uses_repo_root_by_default(tmp_path: Path, monkeypatch):
    outside_workdir = tmp_path / "outside"
    outside_workdir.mkdir()
    monkeypatch.chdir(outside_workdir)

    assert default_dataset_cache_dir() == REPO_ROOT / "run_outputs" / "training_data_cache"


def test_default_dataset_cache_dir_accepts_explicit_repo_root(tmp_path: Path):
    repo_root = tmp_path / "repo"

    assert default_dataset_cache_dir(repo_root) == repo_root / "run_outputs" / "training_data_cache"


def test_dataset_cache_namespace_is_stable_for_mapping_order(tmp_path: Path):
    dataset_path = tmp_path / "dataset"
    config_a = {"path": str(dataset_path), "view_map": {"image_1": ["rgb", "wrist"]}, "adapter": "generic"}
    config_b = {"adapter": "generic", "view_map": {"image_1": ["rgb", "wrist"]}, "path": str(dataset_path)}

    assert dataset_cache_namespace(config_a, dataset_path, action_horizon=14, max_samples_per_file=None) == (
        dataset_cache_namespace(config_b, dataset_path, action_horizon=14, max_samples_per_file=None)
    )


def test_dataset_cache_namespace_changes_when_data_view_or_horizon_changes(tmp_path: Path):
    dataset_path = tmp_path / "dataset"
    config = {"path": str(dataset_path), "view_map": {"image_1": "rgb"}}

    base = dataset_cache_namespace(config, dataset_path, action_horizon=14, max_samples_per_file=None)
    different_horizon = dataset_cache_namespace(config, dataset_path, action_horizon=16, max_samples_per_file=None)
    different_view = dataset_cache_namespace(
        {"path": str(dataset_path), "view_map": {"image_1": "wrist"}},
        dataset_path,
        action_horizon=14,
        max_samples_per_file=None,
    )

    assert base != different_horizon
    assert base != different_view


def test_dataset_cache_namespace_changes_when_action_segment_config_changes(tmp_path: Path):
    dataset_path = tmp_path / "dataset"
    config = {"path": str(dataset_path), "view_map": {"image_1": "rgb"}}

    base = dataset_cache_namespace(config, dataset_path, action_horizon=14, max_samples_per_file=None)
    explicit_none = dataset_cache_namespace(
        config,
        dataset_path,
        action_horizon=14,
        max_samples_per_file=None,
        action_segment_config=None,
    )
    with_segments = dataset_cache_namespace(
        config,
        dataset_path,
        action_horizon=14,
        max_samples_per_file=None,
        action_segment_config={"enabled": True, "num_plan_steps": 4, "planning_horizon": 16},
    )

    assert base == explicit_none
    assert base != with_segments
