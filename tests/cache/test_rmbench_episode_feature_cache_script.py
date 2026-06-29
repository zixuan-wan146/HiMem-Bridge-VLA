from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from himem_bridge_vla.dataset.memory_token_cache import EpisodeFeatureCacheTrajectoryDataset
from himem_bridge_vla.dataset.memory_token_cache import collate_direct_bridge_token_cache_windows


h5py = pytest.importorskip("h5py")
REPO_ROOT = find_repo_root(__file__)
INDEX_SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_rmbench_episode_replay_index.py"
FEATURE_SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_rmbench_episode_feature_cache.py"


def test_build_rmbench_episode_feature_cache_writes_stage1_episode_cache(tmp_path):
    rmbench_root = tmp_path / "RMBench"
    hdf5_path = rmbench_root / "data" / "swap_blocks" / "demo_clean" / "data" / "episode0.hdf5"
    instruction_path = rmbench_root / "data" / "swap_blocks" / "demo_clean" / "instructions" / "episode0.json"
    _write_rmbench_episode(hdf5_path, length=6)
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text(json.dumps({"seen": ["swap the blocks"]}), encoding="utf-8")
    episode_index = tmp_path / "episode_index.json"
    output_root = tmp_path / "features"

    subprocess.run(
        [
            sys.executable,
            str(INDEX_SCRIPT),
            "--rmbench-root",
            str(rmbench_root),
            "--tasks",
            "swap_blocks",
            "--output",
            str(episode_index),
            "--action-horizon",
            "2",
            "--stride",
            "2",
            "--short-offsets",
            "2",
            "1",
            "--executed-action-stride",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(FEATURE_SCRIPT),
            "--episode-index",
            str(episode_index),
            "--output-root",
            str(output_root),
            "--encoder",
            "image_stats",
            "--image-stats-hidden-dim",
            "8",
            "--image-stats-tokens-per-view",
            "2",
            "--include-vlm-hidden-states",
            "--hidden-state-layers",
            "3",
            "6",
            "9",
            "12",
            "--storage-dtype",
            "float32",
            "--max-episodes-per-shard",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    summary = json.loads(completed.stdout)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    payload = torch.load(output_root / "shards" / "shard_000000000_000000001.pt", map_location="cpu", weights_only=False)
    episode = payload["episodes"][0]

    assert summary["format"] == "libero_episode_feature_cache"
    assert summary["episode_count"] == 1
    assert manifest["benchmark"] == "RMBench"
    assert manifest["source_action_start_offset"] == 1
    assert manifest["source_short_offsets"] == [2, 1]
    assert manifest["state_dim"] == 16
    assert manifest["action_dim"] == 14
    assert len(manifest["normalization"]["stats"]["rmbench"]["action"]["min"]) == 14
    assert len(manifest["normalization"]["stats"]["rmbench"]["observation.state"]["min"]) == 16

    assert episode["episode_id"] == "swap_blocks:episode0"
    assert tuple(episode["actions"].shape) == (6, 14)
    assert sorted(episode["visual_tokens_by_step"]) == [0, 1, 2]
    assert sorted(episode["state_by_step"]) == [0, 1, 2]
    assert sorted(episode["current_features_by_step"]) == [0, 2]
    assert episode["nodes"][0]["future_action_range"] == [1, 3]
    assert episode["nodes"][1]["short_visual_steps"] == [0, 1]

    dataset = EpisodeFeatureCacheTrajectoryDataset(output_root, action_horizon=2)
    window = dataset[0]
    batch = collate_direct_bridge_token_cache_windows([window], memory_entry_tokens=4, action_horizon=2)

    assert len(dataset) == 1
    assert window["loss_mask"] == [True, True]
    assert tuple(window["samples"][0]["future_actions"].shape) == (2, 14)
    assert window["samples"][0]["future_actions"][0, 0].item() == pytest.approx(-1.0)
    assert window["samples"][1]["short_steps"].tolist() == [0, 1]
    assert tuple(window["samples"][1]["current_hidden_states"][0].shape) == (6, 8)
    assert tuple(window["samples"][1]["planner_vl_summary"].shape) == (8,)
    assert len(batch["trajectory_steps"]) == 2
    assert tuple(batch["trajectory_steps"][1]["actions"].shape) == (1, 2, 14)
    assert tuple(batch["trajectory_steps"][1]["states"].shape) == (1, 16)


def _write_rmbench_episode(path: Path, *, length: int) -> None:
    path.parent.mkdir(parents=True)
    images = np.zeros((length, 2, 3, 3), dtype=np.uint8)
    images[:, :, :, 1] = np.arange(length, dtype=np.uint8).reshape(length, 1, 1)
    with h5py.File(path, "w") as handle:
        for camera_name in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(f"observation/{camera_name}/rgb", data=images)
        handle.create_dataset("joint_action/vector", data=np.arange(length * 14, dtype=np.float32).reshape(length, 14))
        handle.create_dataset("endpose/left_endpose", data=np.ones((length, 7), dtype=np.float32))
        handle.create_dataset("endpose/right_endpose", data=np.full((length, 7), 2.0, dtype=np.float32))
        handle.create_dataset("endpose/left_gripper", data=np.full((length, 1), 0.25, dtype=np.float32))
        handle.create_dataset("endpose/right_gripper", data=np.full((length, 1), 0.75, dtype=np.float32))
