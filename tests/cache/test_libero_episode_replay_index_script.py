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
SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_libero_episode_replay_index.py"


def test_build_libero_episode_replay_index_writes_episode_first_json(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    hdf5_path = libero_root / "libero_10" / "turn_on_stove_demo.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    _write_libero_task(hdf5_path)
    output = tmp_path / "libero_10_episode_replay.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--libero-root",
            str(libero_root),
            "--suites",
            "libero_10",
            "--output",
            str(output),
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
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(result.stdout)
    assert payload["format"] == "libero_episode_replay_index"
    assert payload["episode_count"] == 2
    assert payload["node_count"] == 5
    assert payload["stride"] == 4
    assert summary["node_count"] == 5

    first_episode = payload["episodes"][0]
    assert first_episode["episode_id"] == "libero_10:turn_on_stove_demo:demo_0"
    assert first_episode["prompt"] == "turn on the stove"
    assert first_episode["required_visual_steps"] == [0, 2, 4]
    assert first_episode["required_visual_frame_count"] == 3
    assert first_episode["nodes"] == [
        {
            "action_valid_count": 4,
            "current_step": 0,
            "current_visual_step": 0,
            "executed_action_range": [0, 0],
            "executed_action_valid_count": 0,
            "future_action_range": [0, 4],
            "required_visual_steps": [0],
            "short_mask": [False, False],
            "short_visual_steps": [None, None],
        },
        {
            "action_valid_count": 4,
            "current_step": 4,
            "current_visual_step": 4,
            "executed_action_range": [0, 4],
            "executed_action_valid_count": 4,
            "future_action_range": [4, 8],
            "required_visual_steps": [0, 2, 4],
            "short_mask": [True, True],
            "short_visual_steps": [0, 2],
        },
    ]


def _write_libero_task(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "turn on the stove"})
        demo_0 = data.create_group("demo_0")
        demo_0.create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32))
        demo_1 = data.create_group("demo_1")
        demo_1.create_dataset("actions", data=np.zeros((12, 7), dtype=np.float32))
