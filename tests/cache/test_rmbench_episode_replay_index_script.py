from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


h5py = pytest.importorskip("h5py")
REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_rmbench_episode_replay_index.py"


def test_build_rmbench_episode_replay_index_uses_next_qpos_targets(tmp_path):
    rmbench_root = tmp_path / "RMBench"
    hdf5_path = rmbench_root / "data" / "swap_blocks" / "demo_clean" / "data" / "episode0.hdf5"
    instruction_path = rmbench_root / "data" / "swap_blocks" / "demo_clean" / "instructions" / "episode0.json"
    _write_rmbench_episode(hdf5_path, length=6)
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text(json.dumps({"seen": ["swap the blocks"]}), encoding="utf-8")
    output = tmp_path / "rmbench_episode_replay.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--rmbench-root",
            str(rmbench_root),
            "--tasks",
            "swap_blocks",
            "--output",
            str(output),
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
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(result.stdout)

    assert payload["format"] == "rmbench_episode_replay_index"
    assert payload["benchmark"] == "RMBench"
    assert payload["action_start_offset"] == 1
    assert payload["short_offsets"] == [2, 1]
    assert payload["episode_count"] == 1
    assert payload["node_count"] == 2
    assert summary["node_count"] == 2

    episode = payload["episodes"][0]
    assert episode["episode_id"] == "swap_blocks:episode0"
    assert episode["prompt"] == "swap the blocks"
    assert episode["required_visual_steps"] == [0, 1, 2]
    assert episode["nodes"] == [
        {
            "action_valid_count": 2,
            "current_step": 0,
            "current_visual_step": 0,
            "executed_action_range": [0, 0],
            "executed_action_valid_count": 0,
            "future_action_range": [1, 3],
            "required_visual_steps": [0],
            "short_mask": [False, False],
            "short_visual_steps": [None, None],
        },
        {
            "action_valid_count": 2,
            "current_step": 2,
            "current_visual_step": 2,
            "executed_action_range": [0, 2],
            "executed_action_valid_count": 2,
            "future_action_range": [3, 5],
            "required_visual_steps": [0, 1, 2],
            "short_mask": [True, True],
            "short_visual_steps": [0, 1],
        },
    ]


def _write_rmbench_episode(path: Path, *, length: int) -> None:
    path.parent.mkdir(parents=True)
    images = np.zeros((length, 2, 3, 3), dtype=np.uint8)
    images[:, :, :, 0] = np.arange(length, dtype=np.uint8).reshape(length, 1, 1)
    with h5py.File(path, "w") as handle:
        for camera_name in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(f"observation/{camera_name}/rgb", data=images)
        handle.create_dataset("joint_action/vector", data=np.arange(length * 14, dtype=np.float32).reshape(length, 14))
        handle.create_dataset("endpose/left_endpose", data=np.ones((length, 7), dtype=np.float32))
        handle.create_dataset("endpose/right_endpose", data=np.full((length, 7), 2.0, dtype=np.float32))
        handle.create_dataset("endpose/left_gripper", data=np.full((length, 1), 0.25, dtype=np.float32))
        handle.create_dataset("endpose/right_gripper", data=np.full((length, 1), 0.75, dtype=np.float32))
