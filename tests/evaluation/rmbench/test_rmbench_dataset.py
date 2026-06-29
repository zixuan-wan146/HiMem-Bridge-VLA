from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
from io import BytesIO
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from himem_bridge_vla.dataset.rmbench import compute_rmbench_normalization_result
from himem_bridge_vla.dataset.rmbench import RMBenchEpisodeReader
from himem_bridge_vla.dataset.rmbench import build_rmbench_state_vector
from himem_bridge_vla.dataset.rmbench import iter_rmbench_episode_files
from himem_bridge_vla.dataset.rmbench import read_rmbench_instruction
from himem_bridge_vla.dataset.rmbench import read_rmbench_state_action_arrays


h5py = pytest.importorskip("h5py")
REPO_ROOT = find_repo_root(__file__)
BUILD_STATS_SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_rmbench_norm_stats.py"
BUILD_REPLAY_INDEX_SCRIPT = REPO_ROOT / "scripts" / "cache" / "build_rmbench_memory_replay_index.py"


def test_rmbench_episode_reader_decodes_images_actions_and_state(tmp_path):
    episode_root = tmp_path / "RMBench" / "data" / "observe_and_pickup" / "demo_clean"
    hdf5_path = episode_root / "data" / "episode0.hdf5"
    instruction_path = episode_root / "instructions" / "episode0.json"
    hdf5_path.parent.mkdir(parents=True)
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text(json.dumps({"seen": ["pick up the marked block"]}), encoding="utf-8")
    _write_episode(hdf5_path)

    reader = RMBenchEpisodeReader(hdf5_path, camera_names=("head_camera", "left_camera", "right_camera"))
    frame = reader.read_frame(1)

    assert len(reader) == 2
    assert reader.action_dim == 14
    assert frame.tau == 1
    assert frame.instruction == "pick up the marked block"
    assert set(frame.images_by_view) == {"head_camera", "left_camera", "right_camera"}
    assert frame.images_by_view["head_camera"].size == (4, 3)
    assert frame.joint_action.shape == (14,)
    assert frame.joint_action[0] == pytest.approx(14.0)
    assert frame.endpose_by_arm["left"].shape == (7,)
    assert frame.gripper_by_arm["right"].shape == (1,)
    assert frame.state_vector.shape == (16,)
    assert frame.state_vector.dtype == np.float32


def test_iter_rmbench_episode_files_pairs_hdf5_with_instruction(tmp_path):
    episode_root = tmp_path / "RMBench" / "data" / "swap_blocks" / "demo_clean"
    hdf5_path = episode_root / "data" / "episode3.hdf5"
    instruction_path = episode_root / "instructions" / "episode3.json"
    hdf5_path.parent.mkdir(parents=True)
    instruction_path.parent.mkdir(parents=True)
    hdf5_path.write_bytes(b"")
    instruction_path.write_text(json.dumps({"seen": ["swap blocks"]}), encoding="utf-8")

    files = list(iter_rmbench_episode_files(tmp_path / "RMBench", tasks=("swap_blocks",)))

    assert len(files) == 1
    assert files[0].task_name == "swap_blocks"
    assert files[0].hdf5_path == hdf5_path
    assert files[0].instruction_path == instruction_path


def test_read_rmbench_state_action_arrays_and_normalization_stats(tmp_path):
    hdf5_path = _create_synthetic_episode(tmp_path, task_name="press_button", episode_name="episode0")

    arrays = read_rmbench_state_action_arrays(hdf5_path)
    result = compute_rmbench_normalization_result(tmp_path / "RMBench", tasks=("press_button",))

    assert arrays.states.shape == (2, 16)
    assert arrays.actions.shape == (2, 14)
    assert result.metadata["episodes"] == 1
    assert result.metadata["frames"] == 2
    assert result.metadata["state_dim"] == 16
    assert result.metadata["action_dim"] == 14
    assert result.stats["rmbench"]["action"]["min"][0] == pytest.approx(0.0)
    assert result.stats["rmbench"]["action"]["max"][0] == pytest.approx(14.0)


def test_build_rmbench_norm_stats_script_writes_stats_and_metadata(tmp_path):
    _create_synthetic_episode(tmp_path, task_name="battery_try", episode_name="episode0")
    stats_output = tmp_path / "norm_stats.json"
    metadata_output = tmp_path / "norm_stats.metadata.json"

    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_STATS_SCRIPT),
            "--rmbench-root",
            str(tmp_path / "RMBench"),
            "--tasks",
            "battery_try",
            "--output",
            str(stats_output),
            "--metadata-output",
            str(metadata_output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stats = json.loads(stats_output.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
    summary = json.loads(result.stdout)
    assert sorted(stats) == ["rmbench"]
    assert metadata["tasks"] == ["battery_try"]
    assert summary["frames"] == 2


def test_build_rmbench_memory_replay_index_script_writes_jsonl_and_manifest(tmp_path):
    _create_synthetic_episode(tmp_path, task_name="swap_T", episode_name="episode0")
    index_output = tmp_path / "memory_replay.jsonl"
    manifest_output = tmp_path / "memory_replay.manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_REPLAY_INDEX_SCRIPT),
            "--rmbench-root",
            str(tmp_path / "RMBench"),
            "--tasks",
            "swap_T",
            "--output",
            str(index_output),
            "--manifest-output",
            str(manifest_output),
            "--action-horizon",
            "1",
            "--stride",
            "1",
            "--short-offsets",
            "2",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in index_output.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    summary = json.loads(result.stdout)
    assert len(rows) == 1
    assert rows[0]["source_path"] == "data/swap_T/demo_clean/data/episode0.hdf5"
    assert rows[0]["current_step"] == 0
    assert rows[0]["action_start"] == 1
    assert rows[0]["action_end"] == 2
    assert rows[0]["short_steps"] == [None, None]
    assert rows[0]["short_mask"] == [False, False]
    assert manifest["action_start_offset"] == 1
    assert manifest["sample_count"] == 1
    assert manifest["task_counts"] == {"swap_T": 1}
    assert summary["episode_counts"] == {"swap_T": 1}


def test_read_rmbench_instruction_prefers_seen_then_falls_back(tmp_path):
    instruction_path = tmp_path / "episode.json"
    instruction_path.write_text(json.dumps({"seen": [], "unseen": ["fallback instruction"]}), encoding="utf-8")

    assert read_rmbench_instruction(instruction_path) == "fallback instruction"


def test_build_rmbench_state_vector_handles_missing_gripper():
    vector = build_rmbench_state_vector(
        {"left": np.ones((7,), dtype=np.float32), "right": np.full((7,), 2.0, dtype=np.float32)},
        {"left": np.array([0.5], dtype=np.float32)},
    )

    assert vector.shape == (15,)
    assert vector[-1] == pytest.approx(2.0)


def _write_episode(path):
    encoded_images = np.empty((2,), dtype=object)
    encoded_images[0] = np.frombuffer(_encoded_rgb((255, 0, 0)), dtype=np.uint8)
    encoded_images[1] = np.frombuffer(_encoded_rgb((0, 255, 0)), dtype=np.uint8)
    image_dtype = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(path, "w") as handle:
        for camera_name in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(f"observation/{camera_name}/rgb", data=encoded_images, dtype=image_dtype)
        handle.create_dataset("joint_action/vector", data=np.arange(28, dtype=np.float32).reshape(2, 14))
        handle.create_dataset("endpose/left_endpose", data=np.ones((2, 7), dtype=np.float32))
        handle.create_dataset("endpose/right_endpose", data=np.full((2, 7), 2.0, dtype=np.float32))
        handle.create_dataset("endpose/left_gripper", data=np.full((2, 1), 0.25, dtype=np.float32))
        handle.create_dataset("endpose/right_gripper", data=np.full((2, 1), 0.75, dtype=np.float32))


def _encoded_rgb(color):
    buffer = BytesIO()
    Image.new("RGB", (4, 3), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _create_synthetic_episode(tmp_path, *, task_name, episode_name):
    episode_root = tmp_path / "RMBench" / "data" / task_name / "demo_clean"
    hdf5_path = episode_root / "data" / f"{episode_name}.hdf5"
    instruction_path = episode_root / "instructions" / f"{episode_name}.json"
    hdf5_path.parent.mkdir(parents=True)
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text(json.dumps({"seen": [f"instruction for {task_name}"]}), encoding="utf-8")
    _write_episode(hdf5_path)
    return hdf5_path
