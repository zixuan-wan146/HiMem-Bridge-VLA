from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_libero_metrics.py"


def load_metrics_module():
    module_path = REPO_ROOT / "scripts" / "check_libero_metrics.py"
    spec = importlib.util.spec_from_file_location("check_libero_metrics", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_result_file(path: Path, *, ckpt_name: str = "run", success_rate: float = 0.5) -> Path:
    successful = int(round(success_rate * 10))
    failed = 10 - successful
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"ckpt_name": ckpt_name},
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
            "suites": {
                "libero_spatial": {
                    "total_episodes": 10,
                    "successful_episodes": successful,
                    "failed_episodes": failed,
                    "success_rate": success_rate,
                    "average_decision_steps": 4.0,
                    "average_control_steps": 56.0,
                    "average_success_decision_steps": 3.0,
                }
            },
        },
        "episodes": [],
    }
    path.write_text(json.dumps(payload))
    return path


def test_check_metric_rows_accepts_absolute_thresholds():
    module = load_metrics_module()
    rows = [
        {
            "scope": "overall",
            "run_name": "candidate",
            "success_rate": 0.6,
            "total_episodes": 10,
        }
    ]

    failures = module.check_metric_rows(
        rows,
        scopes=["overall"],
        min_success_rate=0.5,
        min_total_episodes=10,
    )

    assert failures == []


def test_check_metric_rows_rejects_threshold_failures():
    module = load_metrics_module()
    rows = [
        {
            "scope": "overall",
            "run_name": "candidate",
            "success_rate": 0.4,
            "total_episodes": 8,
        }
    ]

    failures = module.check_metric_rows(
        rows,
        scopes=["overall"],
        min_success_rate=0.5,
        min_total_episodes=10,
    )

    assert any("success_rate" in failure for failure in failures)
    assert any("total_episodes" in failure for failure in failures)


def test_check_metric_rows_rejects_baseline_regression():
    module = load_metrics_module()
    rows = [{"scope": "overall", "run_name": "candidate", "success_rate": 0.65, "total_episodes": 10}]
    baseline_rows = [
        {"scope": "overall", "run_name": "baseline", "success_rate": 0.8, "total_episodes": 10}
    ]

    failures = module.check_metric_rows(
        rows,
        scopes=["overall"],
        baseline_rows=baseline_rows,
        max_regression=0.1,
    )

    assert failures
    assert "regressed below baseline" in failures[0]


def test_main_passes_for_matching_thresholds(tmp_path: Path):
    result_file = write_result_file(tmp_path / "candidate_results.json", success_rate=0.7)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            str(result_file),
            "--min-success-rate",
            "0.6",
            "--min-total-episodes",
            "10",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "[OK] scope=overall" in result.stdout


def test_main_fails_for_regression_against_baseline(tmp_path: Path):
    candidate = write_result_file(tmp_path / "candidate_results.json", ckpt_name="candidate", success_rate=0.6)
    baseline = write_result_file(tmp_path / "baseline_results.json", ckpt_name="baseline", success_rate=0.9)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            str(candidate),
            "--baseline",
            str(baseline),
            "--max-regression",
            "0.1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "regressed below baseline" in result.stderr
