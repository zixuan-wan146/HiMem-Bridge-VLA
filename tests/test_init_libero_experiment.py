from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "init_libero_experiment.py"


def run_output_prefix(tmp_path: Path) -> str:
    return f"run_outputs/test_init_libero_experiment/{tmp_path.parent.name}_{tmp_path.name}"


def load_module():
    module_path = REPO_ROOT / "scripts" / "init_libero_experiment.py"
    spec = importlib.util.spec_from_file_location("init_libero_experiment", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_create_experiment_writes_plan_manifest_profile_and_notes(tmp_path: Path):
    module = load_module()
    prefix = run_output_prefix(tmp_path)
    root = f"{prefix}/experiments"
    checkpoint = f"{prefix}/checkpoint"
    profile = "configs/libero_profiles/full_eval.env"

    args = module.parse_args(
        [
            "--name",
            "baseline_eval",
            "--root",
            root,
            "--checkpoint",
            checkpoint,
            "--profile",
            profile,
            "--kind",
            "eval",
            "--server-python",
            "run_outputs/envs/himem/bin/python",
            "--libero-python",
            "run_outputs/envs/libero/bin/python",
            "--min-total-episodes",
            "10",
            "--min-success-rate",
            "0.1",
        ]
    )

    manifest = module.create_experiment(args)

    experiment_rel = f"{root}/baseline_eval"
    experiment_dir = REPO_ROOT / experiment_rel
    assert "HIMEM_LIBERO_EPISODES=10" in (experiment_dir / "profile.env").read_text()
    assert (experiment_dir / "run_plan.md").exists()
    assert (experiment_dir / "notes.md").read_text().startswith("# baseline_eval")
    payload = json.loads((experiment_dir / "experiment_manifest.json").read_text())
    assert manifest["experiment_name"] == "baseline_eval"
    assert payload["paths"]["run_dir"] == f"{experiment_rel}/run"
    assert payload["paths"]["profile_snapshot"] == f"{experiment_rel}/profile.env"
    assert payload["gate"]["min_success_rate"] == 0.1
    plan_text = (experiment_dir / "run_plan.md").read_text()
    assert f"HIMEM_LIBERO_PROFILE={experiment_rel}/profile.env" in plan_text
    assert f"HIMEM_LIBERO_RUN_DIR={experiment_rel}/run" in plan_text
    assert "--min-success-rate 0.1" in plan_text


def test_create_experiment_dry_run_does_not_write_files(tmp_path: Path):
    module = load_module()
    prefix = run_output_prefix(tmp_path)
    root = f"{prefix}/experiments"

    args = module.parse_args(
        [
            "--name",
            "dry_run",
            "--root",
            root,
            "--checkpoint",
            f"{prefix}/checkpoint",
            "--profile",
            "configs/libero_profiles/smoke.env",
            "--kind",
            "smoke",
            "--dry-run",
        ]
    )

    manifest = module.create_experiment(args)

    assert manifest["kind"] == "smoke"
    assert not (REPO_ROOT / root / "dry_run").exists()


def test_main_refuses_existing_non_empty_experiment(tmp_path: Path):
    prefix = run_output_prefix(tmp_path)
    root = f"{prefix}/experiments"
    experiment_dir = REPO_ROOT / root / "existing"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "old.txt").write_text("keep")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--name",
            "existing",
            "--root",
            root,
            "--checkpoint",
            f"{prefix}/checkpoint",
            "--profile",
            "configs/libero_profiles/smoke.env",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert (experiment_dir / "old.txt").read_text() == "keep"


def test_main_refuses_experiment_path_that_is_file(tmp_path: Path):
    prefix = run_output_prefix(tmp_path)
    root = f"{prefix}/experiments"
    root_path = REPO_ROOT / root
    root_path.mkdir(parents=True)
    (root_path / "not_a_dir").write_text("keep")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--name",
            "not_a_dir",
            "--root",
            root,
            "--checkpoint",
            f"{prefix}/checkpoint",
            "--profile",
            "configs/libero_profiles/smoke.env",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "not a directory" in result.stderr
    assert (root_path / "not_a_dir").read_text() == "keep"
