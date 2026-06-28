from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import json
import os
import subprocess
from pathlib import Path
import sys


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "serve" / "start_himem_server.sh"


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
    ckpt_dir = REPO_ROOT / "run_outputs" / "test_start_himem_server" / tmp_path.name / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(valid_checkpoint_config()))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(valid_norm_stats()))
    (ckpt_dir / "model.pt").write_bytes(b"checkpoint")
    return ckpt_dir


def run_script(
    ckpt_dir: Path,
    *,
    skip_preflight: bool = False,
    allow_unsafe_checkpoint_load: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    ckpt_arg = ckpt_dir.relative_to(REPO_ROOT).as_posix()
    env = {
        **os.environ,
        "HIMEM_PYTHON": "echo",
        "HIMEM_PREFLIGHT_PYTHON": sys.executable,
        "HIMEM_DEVICE": "cpu",
        "HIMEM_PORT": "9010",
    }
    if skip_preflight:
        env["HIMEM_SKIP_PREFLIGHT"] = "1"
    if allow_unsafe_checkpoint_load:
        env["HIMEM_ALLOW_UNSAFE_CHECKPOINT_LOAD"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT), ckpt_arg],
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
    assert "scripts/serve/serve_policy.py" in result.stdout
    assert "--ckpt_dir" in result.stdout
    assert ckpt_dir.relative_to(REPO_ROOT).as_posix() in result.stdout
    assert "--device cpu" in result.stdout
    assert "--vlm_local_files_only" in result.stdout


def test_start_script_can_skip_preflight_for_debugging(tmp_path):
    ckpt_dir = REPO_ROOT / "run_outputs" / "test_start_himem_server" / tmp_path.name / "empty-ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = run_script(ckpt_dir, skip_preflight=True)

    assert result.returncode == 0
    assert "[OK] checkpoint:" not in result.stdout
    assert "scripts/serve/serve_policy.py" in result.stdout


def test_start_script_passes_explicit_unsafe_checkpoint_flag(tmp_path):
    ckpt_dir = make_valid_checkpoint(tmp_path)

    result = run_script(ckpt_dir, allow_unsafe_checkpoint_load=True)

    assert result.returncode == 0
    assert "--allow_unsafe_checkpoint_load" in result.stdout


def test_start_script_passes_vlm_override_without_enabling_downloads(tmp_path):
    ckpt_dir = make_valid_checkpoint(tmp_path)

    result = run_script(
        ckpt_dir,
        extra_env={"HIMEM_VLM_NAME": "/models/InternVL3-1B"},
    )

    assert result.returncode == 0
    assert "--vlm_name /models/InternVL3-1B" in result.stdout
    assert "--vlm_local_files_only" in result.stdout


def test_start_script_can_explicitly_allow_vlm_downloads(tmp_path):
    ckpt_dir = make_valid_checkpoint(tmp_path)

    result = run_script(ckpt_dir, extra_env={"HIMEM_VLM_LOCAL_FILES_ONLY": "0"})

    assert result.returncode == 0
    assert "--vlm_local_files_only" not in result.stdout


def test_start_script_fails_when_preflight_rejects_checkpoint(tmp_path):
    ckpt_dir = REPO_ROOT / "run_outputs" / "test_start_himem_server" / tmp_path.name / "bad-ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = run_script(ckpt_dir)

    assert result.returncode != 0
    assert "missing required files" in result.stdout
    assert "scripts/serve/serve_policy.py" not in result.stdout
