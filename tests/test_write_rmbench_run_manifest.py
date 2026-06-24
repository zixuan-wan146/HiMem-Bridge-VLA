from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_rmbench_run_manifest.py"


def test_write_rmbench_run_manifest_records_run_context(tmp_path: Path):
    output_path = tmp_path / "run" / "run_manifest.json"
    env = {
        **os.environ,
        "HIMEM_RMBENCH_ROOT": "/root/autodl-tmp/benchmarks/RMBench",
        "HIMEM_RMBENCH_RUN_DIR": "run_outputs/rmbench_eval",
        "HIMEM_RMBENCH_POLICY_NAME": "HiMemBridgeVLA",
        "HIMEM_RMBENCH_TASKS": "press_button,swap_blocks",
        "HIMEM_RMBENCH_ACTION_HORIZON": "32",
        "HIMEM_RMBENCH_STATE_SOURCE": "endpose",
        "HIMEM_RMBENCH_PLAN_ONLY": "1",
        "HIMEM_SERVER_URI": "ws://127.0.0.1:9000",
        "HIMEM_TOKEN": "should-not-be-written",
    }

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output_path),
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
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["run_kind"] == "rmbench_eval"
    assert payload["rmbench"]["HIMEM_RMBENCH_POLICY_NAME"] == "HiMemBridgeVLA"
    assert payload["rmbench"]["HIMEM_RMBENCH_TASKS"] == "press_button,swap_blocks"
    assert payload["rmbench"]["HIMEM_RMBENCH_STATE_SOURCE"] == "endpose"
    assert payload["rmbench"]["HIMEM_RMBENCH_PLAN_ONLY"] == "1"
    assert payload["rmbench"]["HIMEM_SERVER_URI"] == "ws://127.0.0.1:9000"
    assert payload["metadata"]["git"]["repo_root"] == "."
    assert "HIMEM_TOKEN" not in payload["metadata"]["environment"]
    assert "HIMEM_TOKEN" not in payload["rmbench"]
