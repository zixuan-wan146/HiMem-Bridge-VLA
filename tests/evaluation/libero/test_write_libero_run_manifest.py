from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import json
import os
import subprocess
from pathlib import Path
import sys


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "report" / "write_libero_run_manifest.py"


def test_write_libero_run_manifest_records_run_context(tmp_path: Path):
    output_path = tmp_path / "run" / "run_manifest.json"
    env = {
        **os.environ,
        "HIMEM_LIBERO_RUN_DIR": "run_outputs/libero_smoke",
        "HIMEM_LIBERO_RESULT_FILE": "run_outputs/libero_smoke/results/smoke_results.json",
        "HIMEM_LIBERO_CKPT_NAME": "smoke",
        "HIMEM_LIBERO_EPISODE_OFFSET": "5",
        "HIMEM_SERVER_URI": "ws://127.0.0.1:9000",
        "HIMEM_TOKEN": "should-not-be-written",
    }

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output_path),
            "--run-kind",
            "smoke",
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(output_path)
    payload = json.loads(output_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["run_kind"] == "smoke"
    assert payload["libero"]["HIMEM_LIBERO_CKPT_NAME"] == "smoke"
    assert payload["libero"]["HIMEM_LIBERO_RUN_DIR"] == "run_outputs/libero_smoke"
    assert payload["libero"]["HIMEM_LIBERO_EPISODE_OFFSET"] == "5"
    assert payload["libero"]["HIMEM_SERVER_URI"] == "ws://127.0.0.1:9000"
    assert payload["metadata"]["git"]["repo_root"] == "."
    assert "HIMEM_TOKEN" not in payload["metadata"]["environment"]
    assert "HIMEM_TOKEN" not in payload["libero"]
