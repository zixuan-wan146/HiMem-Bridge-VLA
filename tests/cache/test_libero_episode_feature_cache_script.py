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
INDEX_SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_libero_episode_replay_index.py"
FEATURE_SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_libero_episode_feature_cache.py"


def test_build_libero_episode_feature_cache_writes_processed_episode_shard(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    hdf5_path = libero_root / "libero_10" / "turn_on_stove_demo.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    _write_libero_task(hdf5_path)
    episode_index = tmp_path / "episode_index.json"
    output_root = tmp_path / "features"

    subprocess.run(
        [
            sys.executable,
            str(INDEX_SCRIPT),
            "--libero-root",
            str(libero_root),
            "--suites",
            "libero_10",
            "--output",
            str(episode_index),
            "--action-horizon",
            "4",
            "--stride",
            "4",
            "--short-offsets",
            "4",
            "2",
            "--executed-action-stride",
            "4",
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
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    summary = json.loads(completed.stdout)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    payload = torch.load(output_root / "shards" / "shard_000000000_000000002.pt", map_location="cpu", weights_only=False)
    episode = payload["episodes"][0]

    assert summary["format"] == "libero_episode_feature_cache"
    assert summary["episode_count"] == 2
    assert manifest["format"] == "libero_episode_feature_cache"
    assert manifest["episode_count"] == 2
    assert manifest["node_count"] == 5
    assert manifest["required_visual_frame_count"] == 8
    assert manifest["source_action_start_offset"] == 0
    assert manifest["hidden_state_layers"] == [3, 6, 9, 12]
    assert manifest["planner_vl_summary"]["enabled"] is True
    assert manifest["state_dim"] == 8
    assert manifest["action_dim"] == 7

    assert episode["episode_id"] == "libero_10:turn_on_stove_demo:demo_0"
    assert "images_by_view" not in episode
    assert tuple(episode["actions"].shape) == (8, 7)
    assert sorted(episode["visual_tokens_by_step"]) == [0, 2, 4]
    assert sorted(episode["state_by_step"]) == [0, 2, 4]
    assert sorted(episode["current_features_by_step"]) == [0, 4]
    assert tuple(episode["visual_tokens_by_step"][2]["agentview_rgb"].shape) == (2, 8)
    assert tuple(episode["state_by_step"][2].shape) == (8,)
    assert len(episode["current_features_by_step"][4]["hidden_states"]) == 4
    assert tuple(episode["current_features_by_step"][4]["planner_vl_summary"].shape) == (8,)
    assert episode["nodes"][1]["short_visual_steps"] == [0, 2]

    dataset = EpisodeFeatureCacheTrajectoryDataset(output_root, action_horizon=4)
    window = dataset[0]
    batch = collate_direct_bridge_token_cache_windows([window], memory_entry_tokens=4, action_horizon=4)

    assert len(dataset) == 2
    assert window["loss_mask"] == [True, True]
    assert len(window["samples"]) == 2
    assert window["samples"][1]["short_steps"].tolist() == [0, 2]
    assert tuple(window["samples"][1]["short_tokens_by_view"][1]["eye_in_hand_rgb"].shape) == (2, 8)
    assert tuple(window["samples"][1]["current_hidden_states"][0].shape) == (4, 8)
    assert tuple(window["samples"][1]["planner_vl_summary"].shape) == (8,)
    assert window["samples"][1]["executed_action_mask"].sum().item() == 4
    assert len(batch["trajectory_steps"]) == 2
    assert batch["trajectory_steps"][1]["loss_mask"].tolist() == [True]
    assert tuple(batch["trajectory_steps"][1]["memory_context"].shape) == (1, 8, 8)
    assert tuple(batch["trajectory_steps"][1]["vlm_hidden_states"][0].shape) == (1, 4, 8)


def _write_libero_task(path: Path) -> None:
    images = np.zeros((12, 2, 3, 3), dtype=np.uint8)
    images[:, :, :, 0] = np.arange(12, dtype=np.uint8).reshape(12, 1, 1)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "turn on the stove"})
        demo_0 = data.create_group("demo_0")
        demo_0.create_dataset("actions", data=np.arange(56, dtype=np.float32).reshape(8, 7))
        demo_0.create_dataset("obs/agentview_rgb", data=images[:8])
        demo_0.create_dataset("obs/eye_in_hand_rgb", data=images[:8] + 1)
        demo_0.create_dataset("obs/ee_states", data=np.ones((8, 7), dtype=np.float32))
        demo_0.create_dataset("obs/gripper_states", data=np.full((8, 1), 0.5, dtype=np.float32))
        demo_1 = data.create_group("demo_1")
        demo_1.create_dataset("actions", data=np.arange(84, dtype=np.float32).reshape(12, 7))
        demo_1.create_dataset("obs/agentview_rgb", data=images)
        demo_1.create_dataset("obs/eye_in_hand_rgb", data=images + 1)
        demo_1.create_dataset("obs/ee_states", data=np.ones((12, 7), dtype=np.float32))
        demo_1.create_dataset("obs/gripper_states", data=np.full((12, 1), 0.5, dtype=np.float32))
