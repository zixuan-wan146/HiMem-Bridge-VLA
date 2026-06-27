from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
from himem_bridge_vla.cli.eval import plan_libero_run
from pathlib import Path
import subprocess
import sys


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "eval" / "plan_libero_run.py"


def load_plan_module():
    return plan_libero_run


def test_plan_libero_run_writes_eval_plan(tmp_path: Path):
    output = f"run_outputs/test_plan_libero_run/{tmp_path.name}/run_plan.md"
    run_dir = f"run_outputs/test_plan_libero_run/{tmp_path.name}/run"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            run_dir,
            "--checkpoint",
            "checkpoints/HiMem_LIBERO",
            "--output",
            output,
            "--server-python",
            ".venv/bin/python",
            "--libero-python",
            "run_outputs/libero_data/envs/libero/bin/python",
            "--min-success-rate",
            "0.1",
            "--min-total-episodes",
            "10",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == output
    plan = (REPO_ROOT / output).read_text()
    assert "scripts/serve/start_himem_server.sh" in plan
    assert "scripts/eval/run_libero_eval.sh" in plan
    assert "scripts/quality/preflight.py" in plan
    assert "scripts/report/report_libero_runs.py" in plan
    assert "HIMEM_CKPT_DIR=" in plan
    assert "checkpoints/HiMem_LIBERO" in plan
    assert "HIMEM_LIBERO_PROFILE=" in plan
    assert "--min-success-rate 0.1" in plan
    assert "--min-total-episodes 10" in plan


def test_plan_libero_run_can_plan_smoke(tmp_path: Path):
    module = load_plan_module()
    args = module.parse_args(
        [
            "--kind",
            "smoke",
            "--run-dir",
            f"run_outputs/test_plan_libero_run/{tmp_path.name}/run",
            "--checkpoint",
            "checkpoints/HiMem_LIBERO",
            "--profile",
            "configs/runtime/libero_profiles/smoke.env",
        ]
    )
    plan = module.format_plan(module.build_plan(args))

    assert "scripts/eval/run_libero_smoke.sh" in plan
    assert "configs/runtime/libero_profiles/smoke.env" in plan


def test_plan_libero_run_includes_baseline_regression_gate(tmp_path: Path):
    output = f"run_outputs/test_plan_libero_run/{tmp_path.name}/run_plan.md"
    run_dir = f"run_outputs/test_plan_libero_run/{tmp_path.name}/run"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            run_dir,
            "--checkpoint",
            "checkpoints/HiMem_LIBERO",
            "--output",
            output,
            "--baseline",
            "run_outputs/baseline",
            "--max-regression",
            "0.02",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    plan = (REPO_ROOT / output).read_text()
    assert "--baseline" in plan
    assert "run_outputs/baseline" in plan
    assert "--max-regression 0.02" in plan


def test_plan_libero_run_rejects_invalid_port(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            f"run_outputs/test_plan_libero_run/{tmp_path.name}/run",
            "--checkpoint",
            "checkpoints/HiMem_LIBERO",
            "--port",
            "70000",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must be between 1 and 65535" in result.stderr
