import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from himem_bridge_vla.dataset.libero_progress_warmup import ImageStatsVLSummaryEncoder
from himem_bridge_vla.dataset.memory_replay import write_memory_replay_jsonl
from himem_bridge_vla.dataset.rmbench_progress_warmup import build_rmbench_progress_vl_embedding_cache


def test_build_rmbench_progress_cache_uses_rmbench_action_protocol(tmp_path: Path):
    rmbench_root = tmp_path / "RMBench"
    hdf5_path = rmbench_root / "data" / "press_button" / "demo_clean" / "data" / "episode0.hdf5"
    instruction_path = rmbench_root / "data" / "press_button" / "demo_clean" / "instructions" / "episode0.json"
    _write_rmbench_episode(hdf5_path, length=12)
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text(json.dumps({"seen": "press the red button"}), encoding="utf-8")
    rows = []
    for step in range(0, 9):
        rows.append(
            {
                "benchmark": "RMBench",
                "episode_id": "press_button:episode0",
                "source_path": "data/press_button/demo_clean/data/episode0.hdf5",
                "instruction_path": "data/press_button/demo_clean/instructions/episode0.json",
                "task_name": "press_button",
                "current_step": step,
                "episode_length": 12,
                "action_start": step,
                "action_end": step + 4,
                "action_valid_count": 4,
                "short_steps": [None, step - 1 if step > 0 else None],
                "short_mask": [False, step > 0],
            }
        )
    index_path = write_memory_replay_jsonl(tmp_path / "rmbench_index.jsonl", rows)

    result = build_rmbench_progress_vl_embedding_cache(
        rmbench_root=rmbench_root,
        index_path=index_path,
        output_root=tmp_path / "progress",
        vl_encoder=ImageStatsVLSummaryEncoder(hidden_dim=8),
        action_horizon=4,
        replan_stride=2,
        burnin_replan_steps=2,
        loss_replan_steps=2,
        allow_short_burnin=True,
        storage_dtype=torch.float32,
        vl_batch_size=2,
    )

    payload = torch.load(result.output_root / "data.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((result.output_root / "manifest.json").read_text(encoding="utf-8"))
    first_step = payload["steps"][0]

    assert manifest["benchmark"] == "RMBench"
    assert result.step_count == 5
    assert result.window_count == 4
    assert first_step["state"].shape == (16,)
    assert first_step["executed_actions"].shape == (2, 14)
    assert first_step["target_intent"].shape == (128,)
    assert first_step["prompt"] == "press the red button"


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
