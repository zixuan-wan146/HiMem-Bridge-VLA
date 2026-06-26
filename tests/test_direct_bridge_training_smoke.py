from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_direct_bridge_token_cache_training_smoke_script_runs_tiny_cpu():
    pytest.importorskip("torch")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_direct_bridge_token_cache_training.py",
            "--preset",
            "tiny",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--batch-size",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["device"] == "cpu"
    assert payload["finite"] is True
    assert payload["fused_tokens_shape"] == [2, 8, 32]
    assert payload["vlm_hidden_state_shapes"] == [[2, 8, 32]] * 4
    assert payload["memory_context_shape"] == [2, 8, 32]
    assert payload["actions_shape"] == [2, 4, 3]
    assert len(payload["losses"]) == 1
    assert payload["losses"][0] > 0
    assert payload["grad_norms"][0] > 0
    assert payload["plan_token_source"] == "random"


def test_direct_bridge_token_cache_training_smoke_script_uses_progress_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "tiny_progress_planner.pt"
    _write_tiny_progress_checkpoint(torch, checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_direct_bridge_token_cache_training.py",
            "--preset",
            "tiny",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--progress-planner-checkpoint",
            str(checkpoint_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["plan_token_source"] == "progress_planner"
    assert payload["vlm_hidden_state_shapes"] == [[2, 8, 32]] * 4
    assert payload["progress_planner"]["hidden_dim"] == 32
    assert payload["progress_planner"]["state_dim"] == 7
    assert payload["progress_planner"]["action_dim"] == 3
    assert payload["progress_planner"]["replan_stride"] == 16
    assert payload["finite"] is True


def test_direct_bridge_token_cache_training_smoke_script_reads_visual_token_manifest(tmp_path):
    torch = pytest.importorskip("torch")

    cache_root = tmp_path / "visual_tokens"
    shard_dir = cache_root / "shards"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "shard_000000000_000000002.pt"

    samples = [_make_token_cache_sample(torch, index) for index in range(2)]
    torch.save(
        {"format": "memory_replay_visual_token_cache", "version": 1, "samples": samples},
        shard_path,
    )
    manifest = {
        "format": "memory_replay_visual_token_cache",
        "version": 1,
        "benchmark": "LIBERO",
        "data_root": str(tmp_path),
        "index_path": str(tmp_path / "index.jsonl"),
        "sample_count": 2,
        "hidden_dim": 8,
        "storage_dtype": "float32",
        "tokens_per_view": 2,
        "shards": [
            {
                "path": "shards/shard_000000000_000000002.pt",
                "sample_count": 2,
                "start_index": 0,
                "end_index": 2,
            }
        ],
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_direct_bridge_token_cache_training.py",
            "--preset",
            "auto",
            "--manifest",
            str(cache_root),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--action-horizon",
            "3",
            "--memory-entry-tokens",
            "4",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["manifest_format"] == "memory_replay_visual_token_cache"
    assert payload["resolved_shape"]["embed_dim"] == 8
    assert payload["resolved_shape"]["horizon"] == 3
    assert payload["resolved_shape"]["per_action_dim"] == 7
    assert payload["resolved_shape"]["state_dim"] == 5
    assert payload["fused_tokens_shape"] == [2, 4, 8]
    assert payload["vlm_hidden_state_shapes"] == [[2, 4, 8]] * 4
    assert payload["memory_context_shape"] == [2, 8, 8]
    assert payload["actions_shape"] == [2, 3, 7]
    assert payload["finite"] is True


def _make_token_cache_sample(torch, index: int) -> dict:
    return {
        "sample_index": index,
        "benchmark": "LIBERO",
        "episode_id": f"episode_{index}",
        "current_step": index,
        "current_tokens_by_view": {
            "agentview_rgb": torch.randn(2, 8),
            "eye_in_hand_rgb": torch.randn(2, 8),
        },
        "current_hidden_states": tuple(torch.randn(4, 8) for _ in range(4)),
        "current_state": torch.randn(5),
        "short_tokens_by_view": (
            {
                "agentview_rgb": torch.randn(2, 8),
                "eye_in_hand_rgb": torch.randn(2, 8),
            },
            None,
        ),
        "short_steps": [max(0, index - 16), -1],
        "short_mask": [True, False],
        "future_actions": torch.randn(3, 7),
        "action_valid_count": 3,
    }


def _write_tiny_progress_checkpoint(torch, checkpoint_path: Path) -> None:
    from himem_bridge_vla.model.planner import ProgressStateConfig
    from himem_bridge_vla.model.planner import ProgressStatePlanner

    config = ProgressStateConfig(
        hidden_dim=32,
        state_dim=7,
        action_dim=3,
        replan_stride=16,
        latent_dim=6,
        action_summary_hidden_dim=32,
        state_hidden_dim=32,
        updater_hidden_dim=64,
        planner_ffn_dim=64,
        planner_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    model = ProgressStatePlanner(config)
    torch.save(
        {
            "format": "progress_state_planner_warmup",
            "model_config": config.__dict__,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
