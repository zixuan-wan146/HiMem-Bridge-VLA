#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.path_utils import display_project_path, project_path  # noqa: E402
from plan_libero_run import build_plan_from_values, format_plan  # noqa: E402


EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible LIBERO experiment skeleton.")
    parser.add_argument("--name", required=True, type=_experiment_name, help="Experiment name.")
    parser.add_argument(
        "--root",
        default="run_outputs/libero_experiments",
        help="Parent directory for experiment skeletons.",
    )
    parser.add_argument("--checkpoint", required=True, help="HiMem-Bridge-VLA checkpoint directory planned for this run.")
    parser.add_argument(
        "--profile",
        default="configs/libero_profiles/full_eval.env",
        help="LIBERO profile file to snapshot into the experiment directory.",
    )
    parser.add_argument("--kind", choices=("smoke", "eval"), default="eval", help="LIBERO run kind.")
    parser.add_argument("--server-python", default="python", help="Python executable for the HiMem-Bridge-VLA server env.")
    parser.add_argument("--libero-python", default="python", help="Python executable for the LIBERO env.")
    parser.add_argument("--host", default="127.0.0.1", help="HiMem-Bridge-VLA server host.")
    parser.add_argument("--port", type=_port, default=9000, help="HiMem-Bridge-VLA server port.")
    parser.add_argument("--device", default="cuda:0", help="HiMem-Bridge-VLA server device.")
    parser.add_argument("--inference-steps", type=_positive_int, default=15, help="HiMem-Bridge-VLA inference steps.")
    parser.add_argument("--min-success-rate", type=_rate, help="Optional report metric gate.")
    parser.add_argument("--min-total-episodes", type=_non_negative_int, help="Optional report metric gate.")
    parser.add_argument("--baseline", action="append", default=[], help="Optional baseline input for reports.")
    parser.add_argument("--max-regression", type=_rate, default=0.0, help="Allowed success-rate regression.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the target experiment directory without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = create_experiment(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[DRY-RUN] experiment_dir={manifest['paths']['experiment_dir']}")
        print(f"[DRY-RUN] run_plan={manifest['paths']['run_plan']}")
    else:
        print(f"[OK] wrote {manifest['paths']['experiment_manifest']}")
        print(f"[OK] wrote {manifest['paths']['run_plan']}")
    return 0


def create_experiment(args: argparse.Namespace) -> dict[str, Any]:
    _validate_executable_ref(args.server_python, "--server-python")
    _validate_executable_ref(args.libero_python, "--libero-python")
    root = _resolve_path(args.root)
    experiment_dir = root / args.name
    run_dir = experiment_dir / "run"
    report_dir = experiment_dir / "report"
    source_profile = _resolve_path(args.profile)
    checkpoint = _resolve_path(args.checkpoint)
    profile_snapshot = experiment_dir / "profile.env"
    run_plan_path = experiment_dir / "run_plan.md"
    notes_path = experiment_dir / "notes.md"
    manifest_path = experiment_dir / "experiment_manifest.json"

    if not source_profile.is_file():
        raise FileNotFoundError(f"LIBERO profile does not exist: {display_project_path(source_profile, REPO_ROOT)}")
    if experiment_dir.exists():
        if not experiment_dir.is_dir():
            raise FileExistsError(
                "experiment path already exists and is not a directory: "
                f"{display_project_path(experiment_dir, REPO_ROOT)}"
            )
        if any(experiment_dir.iterdir()):
            raise FileExistsError(
                "experiment directory already exists and is not empty: "
                f"{display_project_path(experiment_dir, REPO_ROOT)}"
            )

    plan = build_plan_from_values(
        kind=args.kind,
        run_dir=run_dir,
        checkpoint=checkpoint,
        profile=profile_snapshot,
        output=run_plan_path,
        report_dir=report_dir,
        server_python=args.server_python,
        libero_python=args.libero_python,
        host=args.host,
        port=args.port,
        device=args.device,
        inference_steps=args.inference_steps,
        min_success_rate=args.min_success_rate,
        min_total_episodes=args.min_total_episodes,
        baseline=args.baseline,
        max_regression=args.max_regression,
    )
    manifest = _manifest(
        args=args,
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        report_dir=report_dir,
        checkpoint=checkpoint,
        source_profile=source_profile,
        profile_snapshot=profile_snapshot,
        run_plan_path=run_plan_path,
        notes_path=notes_path,
        manifest_path=manifest_path,
        plan=plan,
    )

    if args.dry_run:
        return manifest

    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_profile, profile_snapshot)
    run_plan_path.write_text(format_plan(plan))
    notes_path.write_text(_notes_template(args.name, args.kind))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _manifest(
    *,
    args: argparse.Namespace,
    experiment_dir: Path,
    run_dir: Path,
    report_dir: Path,
    checkpoint: Path,
    source_profile: Path,
    profile_snapshot: Path,
    run_plan_path: Path,
    notes_path: Path,
    manifest_path: Path,
    plan: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": args.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kind": args.kind,
        "paths": {
            "experiment_dir": display_project_path(experiment_dir, REPO_ROOT),
            "run_dir": display_project_path(run_dir, REPO_ROOT),
            "report_dir": display_project_path(report_dir, REPO_ROOT),
            "checkpoint": display_project_path(checkpoint, REPO_ROOT),
            "source_profile": display_project_path(source_profile, REPO_ROOT),
            "profile_snapshot": display_project_path(profile_snapshot, REPO_ROOT),
            "run_plan": display_project_path(run_plan_path, REPO_ROOT),
            "notes": display_project_path(notes_path, REPO_ROOT),
            "experiment_manifest": display_project_path(manifest_path, REPO_ROOT),
        },
        "gate": {
            "min_success_rate": args.min_success_rate,
            "min_total_episodes": args.min_total_episodes,
            "baseline": list(args.baseline),
            "max_regression": args.max_regression,
        },
        "server": {
            "host": args.host,
            "port": args.port,
            "device": args.device,
            "inference_steps": args.inference_steps,
            "server_python": args.server_python,
            "libero_python": args.libero_python,
        },
        "git": _git_metadata(REPO_ROOT),
        "plan": _json_ready(asdict(plan)),
    }


def _notes_template(name: str, kind: str) -> str:
    return "\n".join(
        [
            f"# {name}",
            "",
            f"- Kind: `{kind}`",
            "- Status: planned",
            "",
            "## Objective",
            "",
            "",
            "## Hypothesis",
            "",
            "",
            "## Changes From Baseline",
            "",
            "",
            "## Results",
            "",
            "",
            "## Decision",
            "",
            "",
        ]
    )


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": _git(["rev-parse", "HEAD"], repo_root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "is_dirty": bool(_git(["status", "--porcelain"], repo_root)),
    }


def _git(args: list[str], repo_root: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return display_project_path(value, REPO_ROOT)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _resolve_path(value: str | Path) -> Path:
    return project_path(value, REPO_ROOT)


def _validate_executable_ref(value: str, label: str) -> None:
    path = Path(value).expanduser()
    if path.is_absolute():
        raise ValueError(f"{label} must be a command name or project-relative path, got {value!r}")


def _experiment_name(value: str) -> str:
    if not EXPERIMENT_NAME_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "experiment name must start with an alphanumeric character and only use letters, numbers, '.', '_' or '-'"
        )
    return value


def _rate(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a float") from exc
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError(f"{value!r} must be between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be non-negative")
    return parsed


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError(f"{value!r} must be between 1 and 65535")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
