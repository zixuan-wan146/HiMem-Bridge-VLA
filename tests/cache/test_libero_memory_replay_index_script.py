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
SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_libero_memory_replay_index.py"


def test_build_libero_memory_replay_index_script_writes_jsonl_and_manifest(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    hdf5_path = libero_root / "libero_spatial" / "pick_up_object_demo.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    _write_libero_task(hdf5_path)
    output = tmp_path / "libero_memory_replay.jsonl"
    manifest_output = tmp_path / "libero_memory_replay.manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--libero-root",
            str(libero_root),
            "--suites",
            "libero_spatial",
            "--output",
            str(output),
            "--manifest-output",
            str(manifest_output),
            "--action-horizon",
            "4",
            "--stride",
            "2",
            "--short-offsets",
            "4",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    summary = json.loads(result.stdout)
    assert [row["current_step"] for row in rows] == [0, 2, 0, 2, 4]
    assert rows[0]["benchmark"] == "LIBERO"
    assert rows[0]["source_path"] == "libero_spatial/pick_up_object_demo.hdf5"
    assert rows[0]["task_name"] == "pick_up_object"
    assert rows[0]["episode_key"] == "demo_0"
    assert rows[1]["short_steps"] == [None, 0]
    assert rows[-1]["episode_id"] == "libero_spatial:pick_up_object_demo:demo_1"
    assert manifest["sample_count"] == 5
    assert manifest["suite_episode_counts"] == {"libero_spatial": 2}
    assert manifest["suite_sample_counts"] == {"libero_spatial": 5}
    assert summary["task_episode_counts"] == {"libero_spatial:pick_up_object": 2}


def _write_libero_task(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo_0 = data.create_group("demo_0")
        demo_0.create_dataset("actions", data=np.zeros((6, 7), dtype=np.float32))
        demo_1 = data.create_group("demo_1")
        demo_1.create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32))
