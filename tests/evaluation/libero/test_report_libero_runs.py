from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
from himem_bridge_vla.cli.report import report_libero_runs
import csv
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "report" / "report_libero_runs.py"


def load_report_module():
    return report_libero_runs


def write_run_dir(path: Path, *, success_rate: float = 0.5) -> Path:
    result_file = path / "results" / "run_results.json"
    manifest_file = path / "run_manifest.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    successful = int(round(success_rate * 10))
    failed = 10 - successful
    result_file.write_text(
        json.dumps(
            {
                "config": {"ckpt_name": "run"},
                "metadata": {
                    "created_at_utc": "2026-06-11T00:00:00Z",
                    "git": {"commit": "abc123", "is_dirty": False},
                },
                "summary": {
                    "total_episodes": 10,
                    "successful_episodes": successful,
                    "failed_episodes": failed,
                    "success_rate": success_rate,
                    "average_decision_steps": 4.0,
                    "average_control_steps": 56.0,
                    "average_success_decision_steps": 3.0,
                    "suites": {},
                },
                "episodes": [],
            }
        )
    )
    manifest_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_kind": "smoke",
                "metadata": {
                    "created_at_utc": "2026-06-11T00:00:00Z",
                    "git": {"commit": "abc123", "is_dirty": False},
                },
                "libero": {
                    "HIMEM_LIBERO_CKPT_NAME": "run",
                    "HIMEM_LIBERO_TASK_SUITES": "libero_spatial",
                    "HIMEM_LIBERO_EPISODES": "10",
                    "HIMEM_LIBERO_HORIZON": "14",
                    "HIMEM_LIBERO_MAX_STEPS": "25",
                    "HIMEM_LIBERO_RESULT_FILE": str(result_file),
                },
            }
        )
    )
    return path


def test_write_report_creates_inventory_summary_and_manifest(tmp_path: Path):
    module = load_report_module()
    run_dir = write_run_dir(tmp_path / "runs" / "run_a", success_rate=0.7)
    output_dir = tmp_path / "report"

    report = module.write_report([str(run_dir)], output_dir=output_dir)

    assert report["run_dir_count"] == 1
    assert report["result_file_count"] == 1
    assert (output_dir / "run_inventory.md").exists()
    assert (output_dir / "result_summary.csv").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "report_manifest.json").exists()
    readme = (output_dir / "README.md").read_text()
    assert "# LIBERO Run Report" in readme
    assert "`report_manifest.json`" in readme
    csv_rows = list(csv.DictReader((output_dir / "result_summary.csv").open()))
    assert csv_rows[0]["run_name"] == "run"
    assert csv_rows[0]["success_rate"] == "0.7"


def test_write_report_can_run_metric_gate(tmp_path: Path):
    module = load_report_module()
    run_dir = write_run_dir(tmp_path / "runs" / "run_a", success_rate=0.7)
    output_dir = tmp_path / "report"

    report = module.write_report(
        [str(run_dir)],
        output_dir=output_dir,
        min_success_rate=0.6,
        min_total_episodes=10,
    )

    assert report["metrics_gate"]["enabled"] is True
    assert report["metrics_gate"]["passed"] is True
    assert (output_dir / "metrics_gate.txt").read_text().startswith("enabled: True")


def test_main_returns_failure_when_metric_gate_fails(tmp_path: Path):
    run_dir = write_run_dir(tmp_path / "runs" / "run_a", success_rate=0.2)
    output_dir = tmp_path / "report"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--min-success-rate",
            "0.5",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "below minimum" in result.stderr
    assert (output_dir / "metrics_gate.txt").exists()
    assert "below minimum" in (output_dir / "README.md").read_text()
