from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "start_himem_server.sh"


def valid_checkpoint_config() -> dict:
    return {
        "horizon": 14,
        "per_action_dim": 7,
        "state_dim": 7,
        "action_dim": 98,
    }


def valid_norm_stats() -> dict:
    return {
        "libero": {
            "observation.state": {"min": [0.0] * 7, "max": [1.0] * 7},
            "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
        }
    }


def make_valid_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps(valid_checkpoint_config()))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(valid_norm_stats()))
    (ckpt_dir / "mp_rank_00_model_states.pt").write_bytes(b"checkpoint")
    return ckpt_dir


def run_script(ckpt_dir: Path, *, skip_preflight: bool = False) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HIMEM_PYTHON": "/bin/echo",
        "HIMEM_PREFLIGHT_PYTHON": sys.executable,
        "HIMEM_DEVICE": "cpu",
        "HIMEM_PORT": "9010",
    }
    if skip_preflight:
        env["HIMEM_SKIP_PREFLIGHT"] = "1"
    return subprocess.run(
        ["bash", str(SCRIPT), str(ckpt_dir)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_start_script_runs_checkpoint_preflight_before_server_exec(tmp_path):
    ckpt_dir = make_valid_checkpoint(tmp_path)

    result = run_script(ckpt_dir)

    assert result.returncode == 0
    assert "[OK] checkpoint:" in result.stdout
    assert "scripts/himem_server.py" in result.stdout
    assert "--ckpt_dir" in result.stdout
    assert str(ckpt_dir) in result.stdout
    assert "--device cpu" in result.stdout


def test_start_script_can_skip_preflight_for_debugging(tmp_path):
    ckpt_dir = tmp_path / "empty-ckpt"
    ckpt_dir.mkdir()

    result = run_script(ckpt_dir, skip_preflight=True)

    assert result.returncode == 0
    assert "[OK] checkpoint:" not in result.stdout
    assert "scripts/himem_server.py" in result.stdout


def test_start_script_fails_when_preflight_rejects_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "bad-ckpt"
    ckpt_dir.mkdir()

    result = run_script(ckpt_dir)

    assert result.returncode != 0
    assert "missing required files" in result.stdout
    assert "scripts/himem_server.py" not in result.stdout
