from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from himem_bridge_vla.dataset.memory_replay import write_memory_replay_jsonl
from himem_bridge_vla.dataset.memory_token_cache import ImageStatsVisualTokenEncoder
from himem_bridge_vla.dataset.memory_token_cache import ImageStatsVLMHiddenStateEncoder
from himem_bridge_vla.dataset.memory_token_cache import MemoryTokenCacheDataset
from himem_bridge_vla.dataset.memory_token_cache import build_memory_replay_token_cache
from himem_bridge_vla.dataset.memory_token_cache import collate_direct_bridge_token_cache_samples
from himem_bridge_vla.dataset.memory_token_cache import collate_memory_token_cache_samples


REPO_ROOT = Path(__file__).resolve().parents[1]
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")


def test_build_memory_replay_token_cache_writes_libero_shards_and_manifest(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    _write_libero_episode(libero_root / "libero_spatial" / "pick_demo.hdf5")
    index_path = write_memory_replay_jsonl(
        tmp_path / "libero_index.jsonl",
        [
            {
                "benchmark": "LIBERO",
                "episode_id": "libero_spatial:pick_demo:demo_0",
                "episode_key": "demo_0",
                "source_path": "libero_spatial/pick_demo.hdf5",
                "current_step": 4,
                "episode_length": 6,
                "action_start": 4,
                "action_end": 6,
                "action_valid_count": 2,
                "short_steps": [None, 2],
                "short_mask": [False, True],
            }
        ],
    )

    result = build_memory_replay_token_cache(
        benchmark="LIBERO",
        data_root=libero_root,
        index_path=index_path,
        output_root=tmp_path / "libero_tokens",
        encoder=ImageStatsVisualTokenEncoder(hidden_dim=8, tokens_per_view=1),
        storage_dtype="float32",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    shard_payload = _torch_load(result.shards[0].path)
    sample = shard_payload["samples"][0]

    assert manifest["format"] == "memory_replay_visual_token_cache"
    assert manifest["benchmark"] == "LIBERO"
    assert manifest["sample_count"] == 1
    assert manifest["hidden_dim"] == 8
    assert manifest["tokens_per_view"] == 1
    assert sample["current_tokens_by_view"]["agentview_rgb"].shape == (1, 8)
    assert sample["short_tokens_by_view"][0] is None
    assert sample["short_tokens_by_view"][1]["eye_in_hand_rgb"].shape == (1, 8)
    assert sample["short_steps"].tolist() == [-1, 2]
    assert sample["short_mask"].tolist() == [False, True]
    assert sample["current_state"].shape == (8,)
    assert sample["future_actions"].shape == (2, 7)
    assert sample["prompt"] == "pick up the test cup"


def test_build_memory_replay_token_cache_writes_rmbench_multi_view_tokens(tmp_path):
    rmbench_root = tmp_path / "RMBench"
    _write_rmbench_episode(rmbench_root / "data" / "press_button" / "demo_clean" / "data" / "episode0.hdf5")
    index_path = write_memory_replay_jsonl(
        tmp_path / "rmbench_index.jsonl",
        [
            {
                "benchmark": "RMBench",
                "episode_id": "press_button:episode0",
                "source_path": "data/press_button/demo_clean/data/episode0.hdf5",
                "current_step": 2,
                "episode_length": 5,
                "action_start": 2,
                "action_end": 5,
                "action_valid_count": 3,
                "short_steps": [0, 1],
                "short_mask": [True, True],
            }
        ],
    )

    result = build_memory_replay_token_cache(
        benchmark="RMBench",
        data_root=rmbench_root,
        index_path=index_path,
        output_root=tmp_path / "rmbench_tokens",
        encoder=ImageStatsVisualTokenEncoder(hidden_dim=12, tokens_per_view=2),
        storage_dtype="float32",
    )

    sample = _torch_load(result.shards[0].path)["samples"][0]

    assert set(sample["current_tokens_by_view"]) == {"head_camera", "left_camera", "right_camera"}
    assert sample["current_tokens_by_view"]["head_camera"].shape == (2, 12)
    assert sample["short_tokens_by_view"][0]["left_camera"].shape == (2, 12)
    assert sample["short_steps"].tolist() == [0, 1]
    assert sample["current_state"].shape == (16,)
    assert sample["future_actions"].shape == (3, 14)


def test_memory_token_cache_dataset_reads_shards_and_collates_short_visual_tokens(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    _write_libero_episode(libero_root / "libero_spatial" / "pick_demo.hdf5")
    rows = [
        {
            "benchmark": "LIBERO",
            "episode_id": "libero_spatial:pick_demo:demo_0",
            "episode_key": "demo_0",
            "source_path": "libero_spatial/pick_demo.hdf5",
            "current_step": step,
            "episode_length": 6,
            "action_start": step,
            "action_end": min(step + 2, 6),
            "action_valid_count": min(2, 6 - step),
            "short_steps": [None, step - 1 if step > 0 else None],
            "short_mask": [False, step > 0],
        }
        for step in (0, 2)
    ]
    index_path = write_memory_replay_jsonl(tmp_path / "index.jsonl", rows)
    result = build_memory_replay_token_cache(
        benchmark="LIBERO",
        data_root=libero_root,
        index_path=index_path,
        output_root=tmp_path / "tokens",
        encoder=ImageStatsVisualTokenEncoder(hidden_dim=8, tokens_per_view=1),
        max_samples_per_shard=1,
        storage_dtype="float32",
    )

    dataset = MemoryTokenCacheDataset(result.output_root)
    sample = dataset[1]
    batch = collate_memory_token_cache_samples([dataset[0], sample])

    assert len(dataset) == 2
    assert dataset.config.hidden_dim == 8
    assert sample["sample_index"] == 1
    assert sample["short_steps"].tolist() == [-1, 1]
    assert sample["short_mask"].tolist() == [False, True]
    assert tuple(sample["executed_actions"].shape) == (16, 7)
    assert tuple(sample["executed_action_mask"].shape) == (16,)
    assert sample["executed_action_mask"].sum().item() == 2
    assert sample["short_tokens_by_view"][0] is None
    assert sample["short_tokens_by_view"][1]["agentview_rgb"].shape == (1, 8)
    assert batch["current_step"].tolist() == [0, 2]
    assert tuple(batch["future_actions"].shape) == (2, 2, 7)
    assert batch["action_mask"].tolist() == [[True, True], [True, True]]
    assert batch["short_mask"].tolist() == [[False, False], [False, True]]
    assert batch["short_tokens_by_view"][1][1]["eye_in_hand_rgb"].shape == (1, 8)

    direct_batch = collate_direct_bridge_token_cache_samples(
        [dataset[0], sample],
        memory_entry_tokens=16,
        action_horizon=4,
    )
    assert tuple(direct_batch["fused_tokens"].shape) == (2, 2, 8)
    assert tuple(direct_batch["memory_context"].shape) == (2, 32, 8)
    assert tuple(direct_batch["memory_context_mask"].shape) == (2, 32)
    assert tuple(direct_batch["short_memory_time_ids"].shape) == (2, 32)
    assert tuple(direct_batch["executed_actions"].shape) == (2, 16, 7)
    assert tuple(direct_batch["executed_action_mask"].shape) == (2, 16)
    assert direct_batch["memory_context_mask"][0].sum().item() == 0
    assert direct_batch["memory_context_mask"][1].sum().item() == 2
    assert direct_batch["short_memory_time_ids"][1].tolist() == [0] * 16 + [1] * 16
    assert tuple(direct_batch["actions"].shape) == (2, 4, 7)
    assert tuple(direct_batch["action_mask"].shape) == (2, 4, 7)
    assert direct_batch["action_mask"][:, :2].all().item()
    assert not direct_batch["action_mask"][:, 2:].any().item()


def test_direct_bridge_collate_preserves_optional_vlm_hidden_states():
    samples = [
        {
            "sample_index": index,
            "benchmark": "LIBERO",
            "episode_id": f"episode_{index}",
            "current_step": index,
            "current_tokens_by_view": {"agentview_rgb": torch.randn(2, 8)},
            "current_hidden_states": tuple(torch.randn(2, 8) for _ in range(4)),
            "current_state": torch.randn(5),
            "short_tokens_by_view": (),
            "short_steps": [],
            "short_mask": [],
            "future_actions": torch.randn(3, 7),
            "action_valid_count": 3,
        }
        for index in range(2)
    ]

    batch = collate_direct_bridge_token_cache_samples(samples, action_horizon=3)

    assert "vlm_hidden_states" in batch
    assert len(batch["vlm_hidden_states"]) == 4
    assert [tuple(hidden.shape) for hidden in batch["vlm_hidden_states"]] == [(2, 2, 8)] * 4


def test_build_memory_replay_token_cache_can_write_vlm_hidden_states(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    _write_libero_episode(libero_root / "libero_spatial" / "pick_demo.hdf5")
    index_path = write_memory_replay_jsonl(
        tmp_path / "libero_index.jsonl",
        [
            {
                "benchmark": "LIBERO",
                "episode_id": "libero_spatial:pick_demo:demo_0",
                "episode_key": "demo_0",
                "source_path": "libero_spatial/pick_demo.hdf5",
                "current_step": 2,
                "episode_length": 6,
                "action_start": 2,
                "action_end": 5,
                "action_valid_count": 3,
                "short_steps": [0, 1],
                "short_mask": [True, True],
            }
        ],
    )

    result = build_memory_replay_token_cache(
        benchmark="LIBERO",
        data_root=libero_root,
        index_path=index_path,
        output_root=tmp_path / "libero_tokens",
        encoder=ImageStatsVisualTokenEncoder(hidden_dim=8, tokens_per_view=2),
        hidden_state_encoder=ImageStatsVLMHiddenStateEncoder(
            hidden_dim=8,
            tokens_per_view=2,
            selected_layers=(3, 6, 9, 12),
        ),
        storage_dtype="float32",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    sample = MemoryTokenCacheDataset(result.output_root)[0]
    batch = collate_direct_bridge_token_cache_samples([sample], action_horizon=3)

    assert manifest["hidden_state_encoder"] == "image_stats_vlm_hidden_states"
    assert manifest["hidden_state_layers"] == [3, 6, 9, 12]
    assert sample["prompt"] == "pick up the test cup"
    assert len(sample["current_hidden_states"]) == 4
    assert [tuple(hidden.shape) for hidden in sample["current_hidden_states"]] == [(4, 8)] * 4
    assert [tuple(hidden.shape) for hidden in batch["vlm_hidden_states"]] == [(1, 4, 8)] * 4


def test_build_memory_replay_token_cache_cli_image_stats_smoke(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    _write_libero_episode(libero_root / "libero_spatial" / "pick_demo.hdf5")
    index_path = write_memory_replay_jsonl(
        tmp_path / "index.jsonl",
        [
            {
                "benchmark": "LIBERO",
                "episode_id": "libero_spatial:pick_demo:demo_0",
                "episode_key": "demo_0",
                "source_path": "libero_spatial/pick_demo.hdf5",
                "current_step": 0,
                "episode_length": 6,
                "action_start": 0,
                "action_end": 2,
                "action_valid_count": 2,
                "short_steps": [None, None],
                "short_mask": [False, False],
            }
        ],
    )
    output_root = tmp_path / "tokens"

    command = [
        sys.executable,
        "scripts/build_memory_replay_token_cache.py",
        "--benchmark",
        "LIBERO",
        "--data-root",
        str(libero_root),
        "--index",
        str(index_path),
        "--output-root",
        str(output_root),
        "--encoder",
        "image_stats",
        "--storage-dtype",
        "float32",
        "--max-samples-per-shard",
        "1",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)

    assert payload["sample_count"] == 1
    assert payload["shard_count"] == 1
    assert (output_root / "manifest.json").exists()


def _write_libero_episode(path):
    path.parent.mkdir(parents=True)
    images = np.zeros((6, 2, 3, 3), dtype=np.uint8)
    images[:, :, :, 0] = np.arange(6, dtype=np.uint8).reshape(6, 1, 1)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "pick up the test cup"})
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.arange(42, dtype=np.float32).reshape(6, 7))
        demo.create_dataset("obs/agentview_rgb", data=images)
        demo.create_dataset("obs/eye_in_hand_rgb", data=images + 1)
        demo.create_dataset("obs/ee_states", data=np.ones((6, 7), dtype=np.float32))
        demo.create_dataset("obs/gripper_states", data=np.full((6, 1), 0.5, dtype=np.float32))


def _write_rmbench_episode(path):
    path.parent.mkdir(parents=True)
    images = np.zeros((5, 2, 3, 3), dtype=np.uint8)
    images[:, :, :, 1] = np.arange(5, dtype=np.uint8).reshape(5, 1, 1)
    with h5py.File(path, "w") as handle:
        for camera_name in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(f"observation/{camera_name}/rgb", data=images)
        handle.create_dataset("joint_action/vector", data=np.arange(70, dtype=np.float32).reshape(5, 14))
        handle.create_dataset("endpose/left_endpose", data=np.ones((5, 7), dtype=np.float32))
        handle.create_dataset("endpose/right_endpose", data=np.full((5, 7), 2.0, dtype=np.float32))
        handle.create_dataset("endpose/left_gripper", data=np.full((5, 1), 0.25, dtype=np.float32))
        handle.create_dataset("endpose/right_gripper", data=np.full((5, 1), 0.75, dtype=np.float32))


def _torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)
