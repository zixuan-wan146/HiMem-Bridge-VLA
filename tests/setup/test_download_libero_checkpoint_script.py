from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import os
import subprocess


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "setup" / "download_libero_checkpoint.sh"


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


def test_download_libero_checkpoint_dry_run_uses_data_root():
    data_root = "run_outputs/libero_data_test"
    result = run_dry_run({"HIMEM_DATA_ROOT": data_root})

    assert result.returncode == 0, result.stderr
    values = parse_key_values(result.stdout)
    assert values["HIMEM_LIBERO_CHECKPOINT_REPO"] == "MINT-SJTU/HiMem_LIBERO"
    assert values["HIMEM_LIBERO_CHECKPOINT_DIR"] == f"{data_root}/checkpoints/HiMem_LIBERO"
    assert values["HF_HOME"] == f"{data_root}/hf-home"
    assert values["HUGGINGFACE_HUB_CACHE"] == f"{data_root}/hf-cache"
    assert values["HF_MAX_WORKERS"] == "1"
    assert values["HIMEM_HF_ENDPOINT"] == ""
    assert "hf download MINT-SJTU/HiMem_LIBERO" in values["COMMAND"]


def test_download_libero_checkpoint_dry_run_applies_endpoint_only_when_requested():
    result = run_dry_run(
        {
            "HIMEM_DATA_ROOT": "run_outputs/libero_data_test",
            "HIMEM_HF_ENDPOINT": "https://hf-mirror.com",
            "HF_MAX_WORKERS": "2",
        }
    )

    assert result.returncode == 0, result.stderr
    values = parse_key_values(result.stdout)
    assert values["HIMEM_HF_ENDPOINT"] == "https://hf-mirror.com"
    assert values["HF_MAX_WORKERS"] == "2"
    assert "HF_ENDPOINT=https://hf-mirror.com" in values["COMMAND"]
