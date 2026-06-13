from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from himem_bridge_vla.dataset.calvin_adapter import CalvinBoundaryIndex, CalvinInputAdapter


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_calvin_boundary_index_labels_global_segments(tmp_path: Path):
    sidecar = tmp_path / "boundaries.jsonl"
    write_jsonl(
        sidecar,
        [
            {"segment_id": 0, "start": 10, "end": 20, "task": "open_drawer", "skill_id": 3, "language": "open drawer"},
            {"segment_id": 1, "start": 30, "end": 40, "task": "close_drawer", "skill_id": 4, "language": "close drawer"},
        ],
    )

    index = CalvinBoundaryIndex.from_jsonl(sidecar)
    assert index is not None

    middle = index.label_for(global_frame_index=15, frame_index=None, episode_id=None)
    boundary = index.label_for(global_frame_index=20, frame_index=None, episode_id=None)
    missing = index.label_for(global_frame_index=25, frame_index=None, episode_id=None)

    assert middle is not None
    assert middle.boundary == 0
    assert middle.progress == 0.5
    assert middle.language == "open drawer"
    assert boundary is not None
    assert boundary.boundary == 1
    assert missing is None


def test_calvin_adapter_maps_aliases_and_segment_metadata(tmp_path: Path):
    sidecar = tmp_path / "annotations" / "boundaries.jsonl"
    write_jsonl(
        sidecar,
        [
            {
                "segment_id": 7,
                "start": 100,
                "end": 110,
                "task": "move_slider_left",
                "skill_id": 2,
                "language": "move the slider left",
            }
        ],
    )
    dataset_config = {
        "adapter": "calvin",
        "boundary_path": str(sidecar),
        "fps": 20,
    }
    adapter = CalvinInputAdapter(dataset_config, tmp_path)
    row = pd.Series(
        {
            "state": [0.1] * 8,
            "actions": [0.2] * 7,
            "index": 105,
            "task_index": 99,
        },
        name=5,
    )

    metadata = adapter.sample_metadata(row, Path("data/episode_000/trajectory.parquet"), row_index=5)

    assert adapter.state(row) == [0.1] * 8
    assert adapter.action(row) == [0.2] * 7
    assert adapter.timestamp(row, row_index=5) == 5 / 20
    assert adapter.prompt(row, {99: "fallback task"}, metadata) == "move the slider left"
    assert metadata["boundary"] == 0
    assert metadata["progress"] == 0.5
    assert metadata["skill_id"] == 2
    assert metadata["segment_id"] == 7


def test_calvin_adapter_resolves_first_existing_video_alias(tmp_path: Path):
    adapter = CalvinInputAdapter({"adapter": "calvin"}, tmp_path)
    base_video_path = tmp_path / "videos" / "episode_000"
    video_path = base_video_path / "observation.images.image_0" / "trajectory.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")

    paths = adapter.resolve_video_paths(base_video_path, Path("data/episode_000/trajectory.parquet"))

    assert paths["image_1"] == str(video_path)
