from __future__ import annotations

import json
from pathlib import Path

from himem_bridge_vla.dataset.validation import validate_configured_datasets


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def minimal_config(dataset_path: Path) -> dict:
    return {
        "max_action_dim": 4,
        "max_state_dim": 4,
        "max_views": 1,
        "data_groups": {
            "test_arm": {
                "tiny_dataset": {
                    "path": dataset_path.name,
                    "view_map": {"image_1": "observation.images.image"},
                }
            }
        },
    }


def create_minimal_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "dataset"
    write_jsonl(dataset_path / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "open drawer"}])
    write_jsonl(dataset_path / "meta" / "episodes.jsonl", [{"episode_index": 0}])
    (dataset_path / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"min": [0.0, -1.0], "max": [1.0, 1.0]},
                "action": {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
            }
        )
    )
    parquet_path = dataset_path / "data" / "episode_000" / "trajectory.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"placeholder")
    video_path = dataset_path / "videos" / "episode_000" / "observation.images.image" / "trajectory.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    return dataset_path


def test_validate_configured_datasets_accepts_minimal_structure(tmp_path: Path):
    dataset_path = create_minimal_dataset(tmp_path)

    issues = validate_configured_datasets(minimal_config(dataset_path), tmp_path, require_videos=True)

    assert issues == []


def test_validate_configured_datasets_rejects_missing_video(tmp_path: Path):
    dataset_path = create_minimal_dataset(tmp_path)
    (dataset_path / "videos" / "episode_000" / "observation.images.image" / "trajectory.mp4").unlink()

    issues = validate_configured_datasets(minimal_config(dataset_path), tmp_path, require_videos=True)

    assert any(issue.level == "FAIL" and "missing video" in issue.message for issue in issues)


def test_validate_configured_datasets_can_skip_video_requirement(tmp_path: Path):
    dataset_path = create_minimal_dataset(tmp_path)
    (dataset_path / "videos" / "episode_000" / "observation.images.image" / "trajectory.mp4").unlink()

    issues = validate_configured_datasets(minimal_config(dataset_path), tmp_path, require_videos=False)

    assert issues == []


def test_validate_configured_datasets_accepts_video_alias_candidates(tmp_path: Path):
    dataset_path = create_minimal_dataset(tmp_path)
    config = minimal_config(dataset_path)
    config["data_groups"]["test_arm"]["tiny_dataset"]["view_map"] = {
        "image_1": ["missing.alias", "observation.images.image"]
    }

    issues = validate_configured_datasets(config, tmp_path, require_videos=True)

    assert issues == []


def test_validate_configured_datasets_rejects_invalid_stats(tmp_path: Path):
    dataset_path = create_minimal_dataset(tmp_path)
    (dataset_path / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"min": [0.0, 2.0], "max": [1.0]},
                "action": {"min": [-1.0], "max": [1.0]},
            }
        )
    )

    issues = validate_configured_datasets(minimal_config(dataset_path), tmp_path, require_videos=True)

    assert any(issue.level == "FAIL" and "same length" in issue.message for issue in issues)


def test_validate_configured_datasets_accepts_configured_stat_aliases(tmp_path: Path):
    dataset_path = create_minimal_dataset(tmp_path)
    (dataset_path / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "state": {"min": [0.0, -1.0], "max": [1.0, 1.0]},
                "actions": {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
            }
        )
    )
    config = minimal_config(dataset_path)
    config["data_groups"]["test_arm"]["tiny_dataset"]["state_stat_keys"] = ["observation.state", "state"]
    config["data_groups"]["test_arm"]["tiny_dataset"]["action_stat_keys"] = ["action", "actions"]

    issues = validate_configured_datasets(config, tmp_path, require_videos=True)

    assert issues == []
