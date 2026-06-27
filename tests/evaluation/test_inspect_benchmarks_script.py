from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import json
import subprocess
import sys

from himem_bridge_vla.cli.eval.inspect_benchmarks import build_inventory


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "eval" / "inspect_benchmarks.py"


def test_build_inventory_counts_libero_and_rmbench_assets(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    (libero_root / "libero_spatial").mkdir(parents=True)
    (libero_root / "libero_spatial" / "pick_demo.hdf5").write_bytes(b"")
    rmbench_root = tmp_path / "benchmarks" / "RMBench"
    manifest_path = rmbench_root / "data" / "rmbench_9tasks_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"repo_id": "TianxingChen/RMBench", "tasks": ["observe_and_pickup"], "file_count": 4}),
        encoding="utf-8",
    )
    task_root = rmbench_root / "data" / "observe_and_pickup" / "demo_clean"
    for subdir, filename in {
        "data": "episode0.hdf5",
        "_traj_data": "episode0.pkl",
        "instructions": "episode0.json",
        "video": "episode0.mp4",
    }.items():
        (task_root / subdir).mkdir(parents=True)
        (task_root / subdir / filename).write_bytes(b"")

    inventory = build_inventory(
        data_root=tmp_path,
        libero_root=libero_root,
        libero_plus_root=tmp_path / "libero_plus",
        rmbench_root=rmbench_root,
    )

    assert inventory["benchmarks"]["libero"]["total_demo_files"] == 1
    task = inventory["benchmarks"]["rmbench"]["tasks"]["observe_and_pickup"]
    assert task["hdf5_files"] == 1
    assert task["traj_files"] == 1
    assert task["instruction_files"] == 1
    assert task["video_files"] == 1
    assert inventory["benchmarks"]["libero_plus"]["exists"] is False


def test_build_inventory_keeps_name_similar_libero_plus_candidates_separate(tmp_path):
    (tmp_path / "LIBERO-PRO").mkdir()

    inventory = build_inventory(
        data_root=tmp_path,
        libero_root=tmp_path / "libero" / "datasets",
        libero_plus_root=tmp_path / "libero_plus",
        rmbench_root=tmp_path / "benchmarks" / "RMBench",
    )

    libero_plus = inventory["benchmarks"]["libero_plus"]
    assert libero_plus["exists"] is False
    assert libero_plus["status"] == "missing"
    assert libero_plus["related_candidates"] == [
        {
            "name": "LIBERO-PRO",
            "path": str(tmp_path / "LIBERO-PRO"),
            "exists": True,
            "relation": "name-similar-not-equivalent",
            "is_selected_root": False,
        }
    ]
    assert "not treated as LIBERO-Plus" in libero_plus["notes"][-1]


def test_inspect_benchmarks_allow_missing_exits_success(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-root",
            str(tmp_path),
            "--allow-missing",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["benchmarks"]["libero"]["exists"] is False
