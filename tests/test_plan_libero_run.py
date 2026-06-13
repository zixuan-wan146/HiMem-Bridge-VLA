from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "plan_libero_run.py"


def load_plan_module():
    module_path = REPO_ROOT / "scripts" / "plan_libero_run.py"
    spec = importlib.util.spec_from_file_location("plan_libero_run", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    assert "scripts/start_himem_server.sh" in plan
    assert "scripts/run_libero_eval.sh" in plan
    assert "scripts/preflight.py" in plan
    assert "scripts/report_libero_runs.py" in plan
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
            "configs/libero_profiles/smoke.env",
        ]
    )
    plan = module.format_plan(module.build_plan(args))

    assert "scripts/run_libero_smoke.sh" in plan
    assert "configs/libero_profiles/smoke.env" in plan


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
