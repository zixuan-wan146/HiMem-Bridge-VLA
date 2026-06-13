from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "download_libero_checkpoint.sh"


def run_dry_run(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HIMEM_DOWNLOAD_LIBERO_CHECKPOINT_DRY_RUN": "1",
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_key_values(stdout: str) -> dict[str, str]:
    values = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_download_libero_checkpoint_dry_run_uses_data_root(tmp_path: Path):
    data_root = tmp_path / "data"
    result = run_dry_run({"HIMEM_DATA_ROOT": str(data_root)})

    assert result.returncode == 0, result.stderr
    values = parse_key_values(result.stdout)
    assert values["HIMEM_LIBERO_CHECKPOINT_REPO"] == "MINT-SJTU/HiMem_LIBERO"
    assert values["HIMEM_LIBERO_CHECKPOINT_DIR"] == str(data_root / "checkpoints" / "HiMem_LIBERO")
    assert values["HF_HOME"] == str(data_root / "hf-home")
    assert values["HUGGINGFACE_HUB_CACHE"] == str(data_root / "hf-cache")
    assert values["HF_MAX_WORKERS"] == "1"
    assert values["HIMEM_HF_ENDPOINT"] == ""
    assert "hf download MINT-SJTU/HiMem_LIBERO" in values["COMMAND"]


def test_download_libero_checkpoint_dry_run_applies_endpoint_only_when_requested(tmp_path: Path):
    result = run_dry_run(
        {
            "HIMEM_DATA_ROOT": str(tmp_path / "data"),
            "HIMEM_HF_ENDPOINT": "https://hf-mirror.com",
            "HF_MAX_WORKERS": "2",
        }
    )

    assert result.returncode == 0, result.stderr
    values = parse_key_values(result.stdout)
    assert values["HIMEM_HF_ENDPOINT"] == "https://hf-mirror.com"
    assert values["HF_MAX_WORKERS"] == "2"
    assert "HF_ENDPOINT=https://hf-mirror.com" in values["COMMAND"]
