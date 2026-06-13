from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "init_libero_experiment.py"


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
    profile = tmp_path / "full_eval.env"
    profile.write_text("HIMEM_LIBERO_EPISODES=10\n")
    checkpoint = tmp_path / "checkpoint"

    args = module.parse_args(
        [
            "--name",
            "baseline_eval",
            "--root",
            str(tmp_path / "experiments"),
            "--checkpoint",
            str(checkpoint),
            "--profile",
            str(profile),
            "--kind",
            "eval",
            "--server-python",
            "/envs/himem/bin/python",
            "--libero-python",
            "/envs/libero/bin/python",
            "--min-total-episodes",
            "10",
            "--min-success-rate",
            "0.1",
        ]
    )

    manifest = module.create_experiment(args)

    experiment_dir = tmp_path / "experiments" / "baseline_eval"
    assert (experiment_dir / "profile.env").read_text() == "HIMEM_LIBERO_EPISODES=10\n"
    assert (experiment_dir / "run_plan.md").exists()
    assert (experiment_dir / "notes.md").read_text().startswith("# baseline_eval")
    payload = json.loads((experiment_dir / "experiment_manifest.json").read_text())
    assert manifest["experiment_name"] == "baseline_eval"
    assert payload["paths"]["run_dir"] == str(experiment_dir / "run")
    assert payload["paths"]["profile_snapshot"] == str(experiment_dir / "profile.env")
    assert payload["gate"]["min_success_rate"] == 0.1
    plan_text = (experiment_dir / "run_plan.md").read_text()
    assert f"HIMEM_LIBERO_PROFILE={experiment_dir / 'profile.env'}" in plan_text
    assert f"HIMEM_LIBERO_RUN_DIR={experiment_dir / 'run'}" in plan_text
    assert "--min-success-rate 0.1" in plan_text


def test_create_experiment_dry_run_does_not_write_files(tmp_path: Path):
    module = load_module()
    profile = tmp_path / "smoke.env"
    profile.write_text("HIMEM_LIBERO_TASK_LIMIT=1\n")

    args = module.parse_args(
        [
            "--name",
            "dry_run",
            "--root",
            str(tmp_path / "experiments"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--profile",
            str(profile),
            "--kind",
            "smoke",
            "--dry-run",
        ]
    )

    manifest = module.create_experiment(args)

    assert manifest["kind"] == "smoke"
    assert not (tmp_path / "experiments" / "dry_run").exists()


def test_main_refuses_existing_non_empty_experiment(tmp_path: Path):
    profile = tmp_path / "smoke.env"
    profile.write_text("HIMEM_LIBERO_TASK_LIMIT=1\n")
    experiment_dir = tmp_path / "experiments" / "existing"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "old.txt").write_text("keep")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--name",
            "existing",
            "--root",
            str(tmp_path / "experiments"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--profile",
            str(profile),
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
    profile = tmp_path / "smoke.env"
    profile.write_text("HIMEM_LIBERO_TASK_LIMIT=1\n")
    root = tmp_path / "experiments"
    root.mkdir()
    (root / "not_a_dir").write_text("keep")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--name",
            "not_a_dir",
            "--root",
            str(root),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--profile",
            str(profile),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "not a directory" in result.stderr
    assert (root / "not_a_dir").read_text() == "keep"
