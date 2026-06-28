from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import json
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


REPO_ROOT = find_repo_root(__file__)
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")


def test_build_memory_replay_token_cache_rejects_libero(tmp_path):
    with pytest.raises(ValueError, match="LIBERO Stage1 no longer uses"):
        build_memory_replay_token_cache(
            benchmark="LIBERO",
            data_root=tmp_path,
            index_path=tmp_path / "missing.jsonl",
            output_root=tmp_path / "libero_tokens",
            encoder=ImageStatsVisualTokenEncoder(hidden_dim=8, tokens_per_view=1),
            storage_dtype="float32",
        )


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
    rmbench_root = tmp_path / "RMBench"
    _write_rmbench_episode(rmbench_root / "data" / "press_button" / "demo_clean" / "data" / "episode0.hdf5")
    rows = [
        {
            "benchmark": "RMBench",
            "episode_id": "press_button:episode0",
            "source_path": "data/press_button/demo_clean/data/episode0.hdf5",
            "current_step": step,
            "episode_length": 5,
            "action_start": step,
            "action_end": min(step + 2, 5),
            "action_valid_count": min(2, 5 - step),
            "short_steps": [None, step - 1 if step > 0 else None],
            "short_mask": [False, step > 0],
        }
        for step in (0, 2)
    ]
    index_path = write_memory_replay_jsonl(tmp_path / "index.jsonl", rows)
    result = build_memory_replay_token_cache(
        benchmark="RMBench",
        data_root=rmbench_root,
        index_path=index_path,
        output_root=tmp_path / "tokens",
        encoder=ImageStatsVisualTokenEncoder(hidden_dim=8, tokens_per_view=1),
        max_samples_per_shard=1,
        storage_dtype="float32",
    )

    dataset = MemoryTokenCacheDataset(result.output_root)
    sample = dataset[1]
    batch = collate_memory_token_cache_samples([dataset[0], sample])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert len(dataset) == 2
    assert dataset.config.hidden_dim == 8
    assert manifest["normalization"]["type"] == "train_split_minmax_to_minus_one_one"
    assert dataset.arm2stats_dict is not None
    assert sample["sample_index"] == 1
    assert sample["short_steps"].tolist() == [-1, 1]
    assert sample["short_mask"].tolist() == [False, True]
    assert tuple(sample["executed_actions"].shape) == (16, 14)
    assert tuple(sample["executed_action_mask"].shape) == (16,)
    assert sample["executed_action_mask"].sum().item() == 2
    assert sample["future_actions"].amin().item() >= -1.0
    assert sample["future_actions"].amax().item() <= 1.0
    assert not sample["executed_actions"][:14].any().item()
    assert sample["short_tokens_by_view"][0] is None
    assert sample["short_tokens_by_view"][1]["head_camera"].shape == (1, 8)
    assert batch["current_step"].tolist() == [0, 2]
    assert tuple(batch["future_actions"].shape) == (2, 2, 14)
    assert batch["action_mask"].tolist() == [[True, True], [True, True]]
    assert batch["short_mask"].tolist() == [[False, False], [False, True]]
    assert batch["short_tokens_by_view"][1][1]["right_camera"].shape == (1, 8)

    direct_batch = collate_direct_bridge_token_cache_samples(
        [dataset[0], sample],
        memory_entry_tokens=16,
        action_horizon=4,
    )
    assert tuple(direct_batch["fused_tokens"].shape) == (2, 3, 8)
    assert tuple(direct_batch["memory_context"].shape) == (2, 32, 8)
    assert tuple(direct_batch["memory_context_mask"].shape) == (2, 32)
    assert tuple(direct_batch["short_memory_time_ids"].shape) == (2, 32)
    assert tuple(direct_batch["executed_actions"].shape) == (2, 16, 14)
    assert tuple(direct_batch["executed_action_mask"].shape) == (2, 16)
    assert direct_batch["memory_context_mask"][0].sum().item() == 0
    assert direct_batch["memory_context_mask"][1].sum().item() == 3
    assert direct_batch["short_memory_time_ids"][1].tolist() == [0] * 16 + [1] * 16
    assert tuple(direct_batch["actions"].shape) == (2, 4, 14)
    assert tuple(direct_batch["action_mask"].shape) == (2, 4, 14)
    assert direct_batch["action_mask"][:, :2].all().item()
    assert not direct_batch["action_mask"][:, 2:].any().item()


def test_build_memory_replay_token_cache_reuses_frame_visual_tokens(tmp_path):
    rmbench_root = tmp_path / "RMBench"
    _write_rmbench_episode(rmbench_root / "data" / "press_button" / "demo_clean" / "data" / "episode0.hdf5")
    rows = [
        {
            "benchmark": "RMBench",
            "episode_id": "press_button:episode0",
            "source_path": "data/press_button/demo_clean/data/episode0.hdf5",
            "current_step": 0,
            "episode_length": 5,
            "action_start": 0,
            "action_end": 2,
            "action_valid_count": 2,
            "short_steps": [None, None],
            "short_mask": [False, False],
        },
        {
            "benchmark": "RMBench",
            "episode_id": "press_button:episode0",
            "source_path": "data/press_button/demo_clean/data/episode0.hdf5",
            "current_step": 2,
            "episode_length": 5,
            "action_start": 2,
            "action_end": 4,
            "action_valid_count": 2,
            "short_steps": [0, None],
            "short_mask": [True, False],
        },
    ]
    index_path = write_memory_replay_jsonl(tmp_path / "index.jsonl", rows)
    encoder = _CountingImageStatsVisualTokenEncoder(hidden_dim=8, tokens_per_view=1)

    result = build_memory_replay_token_cache(
        benchmark="RMBench",
        data_root=rmbench_root,
        index_path=index_path,
        output_root=tmp_path / "tokens",
        encoder=encoder,
        storage_dtype="float32",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    dataset = MemoryTokenCacheDataset(result.output_root)

    assert encoder.encode_calls == 6
    assert manifest["builder_mode"] == "frame_token_dedup"
    assert manifest["visual_token_cache_entries"] == 2
    assert dataset[1]["short_tokens_by_view"][0]["head_camera"].shape == (1, 8)


def test_direct_bridge_collate_preserves_optional_vlm_hidden_states():
    samples = [
        {
            "sample_index": index,
            "benchmark": "LIBERO",
            "episode_id": f"episode_{index}",
            "current_step": index,
            "current_tokens_by_view": {"agentview_rgb": torch.randn(2, 8)},
            "current_hidden_states": tuple(torch.randn(2, 8) for _ in range(4)),
            "planner_vl_summary": torch.full((8,), float(index + 1)),
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
    assert batch["planner_vl_summary"].tolist() == [[1.0] * 8, [2.0] * 8]


def test_build_memory_replay_token_cache_can_write_vlm_hidden_states(tmp_path):
    rmbench_root = tmp_path / "RMBench"
    _write_rmbench_episode(rmbench_root / "data" / "press_button" / "demo_clean" / "data" / "episode0.hdf5")
    index_path = write_memory_replay_jsonl(
        tmp_path / "rmbench_index.jsonl",
        [
            {
                "benchmark": "RMBench",
                "episode_id": "press_button:episode0",
                "source_path": "data/press_button/demo_clean/data/episode0.hdf5",
                "task_name": "press_button",
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
    assert manifest["planner_vl_summary"]["enabled"] is True
    assert manifest["planner_vl_summary"]["source"] == "vlm_last_valid_token"
    assert sample["prompt"] == "press button"
    assert len(sample["current_hidden_states"]) == 4
    assert [tuple(hidden.shape) for hidden in sample["current_hidden_states"]] == [(6, 8)] * 4
    assert [tuple(hidden.shape) for hidden in batch["vlm_hidden_states"]] == [(1, 6, 8)] * 4
    assert tuple(sample["planner_vl_summary"].shape) == (8,)
    assert tuple(batch["planner_vl_summary"].shape) == (1, 8)


def test_build_memory_replay_token_cache_cli_image_stats_smoke(tmp_path):
    rmbench_root = tmp_path / "RMBench"
    _write_rmbench_episode(rmbench_root / "data" / "press_button" / "demo_clean" / "data" / "episode0.hdf5")
    index_path = write_memory_replay_jsonl(
        tmp_path / "index.jsonl",
        [
            {
                "benchmark": "RMBench",
                "episode_id": "press_button:episode0",
                "source_path": "data/press_button/demo_clean/data/episode0.hdf5",
                "current_step": 0,
                "episode_length": 5,
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
        "scripts/cache/build_memory_replay_token_cache.py",
        "--benchmark",
        "RMBench",
        "--data-root",
        str(rmbench_root),
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


class _CountingImageStatsVisualTokenEncoder(ImageStatsVisualTokenEncoder):
    def __init__(self, *, hidden_dim: int, tokens_per_view: int) -> None:
        super().__init__(hidden_dim=hidden_dim, tokens_per_view=tokens_per_view)
        self.encode_calls = 0

    def encode_image(self, image):
        self.encode_calls += 1
        return super().encode_image(image)
