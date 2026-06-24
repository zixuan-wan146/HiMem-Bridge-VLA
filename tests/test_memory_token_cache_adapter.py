from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from himem_bridge_vla.training import MemoryTokenCacheTrainingConfig
from himem_bridge_vla.training import masked_action_chunk_mse
from himem_bridge_vla.training import run_memory_token_cache_training


torch = pytest.importorskip("torch")
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_memory_token_cache_adapter_trains_and_writes_checkpoint(tmp_path):
    cache_root = _write_tiny_token_cache(tmp_path / "cache")
    output_dir = tmp_path / "run"

    result = run_memory_token_cache_training(
        MemoryTokenCacheTrainingConfig(
            cache_manifest=str(cache_root),
            output_dir=str(output_dir),
            device="cpu",
            batch_size=2,
            max_steps=2,
            lr=1e-3,
            num_heads=2,
            tokens_per_entry=1,
            repo_root=str(REPO_ROOT),
        )
    )

    assert result.steps == 2
    assert math.isfinite(result.final_loss)
    assert result.checkpoint_path.exists()
    assert (output_dir / "resolved_config.json").exists()
    assert (output_dir / "environment.json").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["final_loss"] == pytest.approx(result.final_loss)
    assert len(metrics["steps"]) == 2


def test_memory_token_cache_adapter_cli_smoke(tmp_path):
    cache_root = _write_tiny_token_cache(tmp_path / "cache")
    output_dir = tmp_path / "cli_run"
    command = [
        sys.executable,
        "scripts/train_memory_token_cache_adapter.py",
        "--cache-manifest",
        str(cache_root),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--max-steps",
        "1",
        "--num-heads",
        "2",
    ]

    completed = subprocess.run(command, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)

    assert payload["steps"] == 1
    assert math.isfinite(payload["final_loss"])
    assert (output_dir / "adapter_last.pt").exists()


def test_masked_action_chunk_mse_rejects_empty_mask():
    pred = torch.zeros(1, 2, 3)
    target = torch.zeros(1, 2, 3)
    mask = torch.zeros(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="no active"):
        masked_action_chunk_mse(pred, target, mask)


def _write_tiny_token_cache(cache_root: Path) -> Path:
    shard_dir = cache_root / "shards"
    shard_dir.mkdir(parents=True)
    samples = [_sample(index) for index in range(4)]
    shard_path = shard_dir / "shard_000000000_000000004.pt"
    torch.save(
        {
            "format": "memory_replay_visual_token_cache",
            "version": 1,
            "samples": samples,
        },
        shard_path,
    )
    manifest = {
        "format": "memory_replay_visual_token_cache",
        "version": 1,
        "benchmark": "LIBERO",
        "data_root": ".",
        "index_path": "index.jsonl",
        "output_root": str(cache_root),
        "encoder": "synthetic",
        "hidden_dim": 8,
        "tokens_per_view": 1,
        "storage_dtype": "float32",
        "sample_count": len(samples),
        "max_samples": None,
        "max_samples_per_shard": len(samples),
        "view_names": ["cam"],
        "shards": [
            {
                "path": "shards/shard_000000000_000000004.pt",
                "sample_count": len(samples),
                "start_index": 0,
                "end_index": len(samples),
            }
        ],
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cache_root


def _sample(index: int) -> dict:
    current_tokens = torch.full((2, 8), float(index + 1), dtype=torch.float32)
    short_token_a = torch.full((1, 8), float(index), dtype=torch.float32)
    short_token_b = torch.full((1, 8), float(index + 0.5), dtype=torch.float32)
    valid_history = index >= 2
    return {
        "sample_index": index,
        "benchmark": "LIBERO",
        "episode_id": f"episode:{index}",
        "current_step": index + 32,
        "current_tokens_by_view": {"cam": current_tokens},
        "current_state": torch.tensor([index, index + 1, index + 2], dtype=torch.float32),
        "short_tokens_by_view": (
            {"cam": short_token_a} if valid_history else None,
            {"cam": short_token_b} if valid_history else None,
        ),
        "short_steps": torch.tensor([index, index + 16] if valid_history else [-1, -1], dtype=torch.long),
        "short_mask": torch.tensor([valid_history, valid_history], dtype=torch.bool),
        "future_actions": torch.full((4, 3), float(index) / 10.0, dtype=torch.float32),
        "action_valid_count": 4,
    }
