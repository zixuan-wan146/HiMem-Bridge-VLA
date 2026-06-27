#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import sys
from typing import Sequence
from himem_bridge_vla.path_utils import find_repo_root


REPO_ROOT = find_repo_root(__file__)
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from himem_bridge_vla.path_utils import normalize_project_relative_path  # noqa: E402


@dataclass(frozen=True)
class LiberoRunPlan:
    kind: str
    run_dir: Path
    checkpoint: Path
    profile: Path
    output: Path
    report_dir: Path
    server_python: str
    libero_python: str
    host: str
    port: int
    device: str
    inference_steps: int
    min_success_rate: float | None
    min_total_episodes: int | None
    baseline: tuple[str, ...]
    max_regression: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducible LIBERO run plan.")
    parser.add_argument("--run-dir", required=True, help="Directory that will hold one LIBERO run.")
    parser.add_argument("--checkpoint", required=True, help="HiMem-Bridge-VLA checkpoint directory.")
    parser.add_argument(
        "--profile",
        default="configs/runtime/libero_profiles/full_eval.env",
        help="LIBERO profile file to use.",
    )
    parser.add_argument("--kind", choices=("smoke", "eval"), default="eval", help="LIBERO client script to plan.")
    parser.add_argument("--output", help="Plan Markdown path. Defaults to <run-dir>/run_plan.md.")
    parser.add_argument("--report-dir", help="Report directory. Defaults to <run-dir>/report.")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    args = parse_args(argv)
    plan = build_plan(args)
    write_plan(plan)
    print(plan.output)
    return 0


def build_plan(args: argparse.Namespace) -> LiberoRunPlan:
    return build_plan_from_values(
        kind=args.kind,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        profile=args.profile,
        output=args.output,
        report_dir=args.report_dir,
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


def build_plan_from_values(
    *,
    kind: str,
    run_dir: str | Path,
    checkpoint: str | Path,
    profile: str | Path,
    output: str | Path | None = None,
    report_dir: str | Path | None = None,
    server_python: str = "python",
    libero_python: str = "python",
    host: str = "127.0.0.1",
    port: int = 9000,
    device: str = "cuda:0",
    inference_steps: int = 15,
    min_success_rate: float | None = None,
    min_total_episodes: int | None = None,
    baseline: Sequence[str] = (),
    max_regression: float = 0.0,
) -> LiberoRunPlan:
    run_dir = _resolve_path(run_dir)
    output_path = _resolve_path(output) if output else run_dir / "run_plan.md"
    report_dir_path = _resolve_path(report_dir) if report_dir else run_dir / "report"
    return LiberoRunPlan(
        kind=kind,
        run_dir=run_dir,
        checkpoint=_resolve_path(checkpoint),
        profile=_resolve_path(profile),
        output=output_path,
        report_dir=report_dir_path,
        server_python=server_python,
        libero_python=libero_python,
        host=host,
        port=port,
        device=device,
        inference_steps=inference_steps,
        min_success_rate=min_success_rate,
        min_total_episodes=min_total_episodes,
        baseline=tuple(baseline),
        max_regression=max_regression,
    )


def write_plan(plan: LiberoRunPlan) -> Path:
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    plan.output.write_text(format_plan(plan))
    return plan.output


def format_plan(plan: LiberoRunPlan) -> str:
    client_script = "scripts/eval/run_libero_smoke.sh" if plan.kind == "smoke" else "scripts/eval/run_libero_eval.sh"
    server_command = _command(
        [
            "env",
            f"HIMEM_PYTHON={plan.server_python}",
            f"HIMEM_CKPT_DIR={plan.checkpoint}",
            f"HIMEM_HOST={plan.host}",
            f"HIMEM_PORT={plan.port}",
            f"HIMEM_DEVICE={plan.device}",
            f"HIMEM_INFERENCE_STEPS={plan.inference_steps}",
            "scripts/serve/start_himem_server.sh",
        ]
    )
    eval_command = _command(
        [
            "env",
            f"HIMEM_LIBERO_PROFILE={plan.profile}",
            f"HIMEM_LIBERO_RUN_DIR={plan.run_dir}",
            f"HIMEM_SERVER_URI=ws://{plan.host}:{plan.port}",
            f"LIBERO_PYTHON={plan.libero_python}",
            client_script,
        ]
    )
    validate_command = _command(
        [
            "python3",
            "scripts/quality/preflight.py",
            "--dataset-config",
            "",
            "--libero-run-dir",
            str(plan.run_dir),
        ]
    )
    report_command = _command(_report_command_args(plan))
    dry_run_command = _command(
        [
            "env",
            "HIMEM_LIBERO_DRY_RUN=1",
            f"HIMEM_LIBERO_PROFILE={plan.profile}",
            f"HIMEM_LIBERO_RUN_DIR={plan.run_dir}",
            client_script,
        ]
    )

    return "\n".join(
        [
            "# LIBERO Run Plan",
            "",
            "## Paths",
            "",
            f"- Kind: `{plan.kind}`",
            f"- Run directory: `{plan.run_dir}`",
            f"- Checkpoint: `{plan.checkpoint}`",
            f"- Profile: `{plan.profile}`",
            f"- Report directory: `{plan.report_dir}`",
            "",
            "## 1. Dry-Run LIBERO Settings",
            "",
            "```bash",
            dry_run_command,
            "```",
            "",
            "## 2. Start HiMem-Bridge-VLA Server",
            "",
            "Run this in one shell and keep it running:",
            "",
            "```bash",
            server_command,
            "```",
            "",
            "## 3. Run LIBERO Client",
            "",
            "Run this from another shell after the server is listening:",
            "",
            "```bash",
            eval_command,
            "```",
            "",
            "## 4. Validate Run Artifacts",
            "",
            "```bash",
            validate_command,
            "```",
            "",
            "## 5. Generate Report",
            "",
            "```bash",
            report_command,
            "```",
            "",
        ]
    )


def _report_command_args(plan: LiberoRunPlan) -> list[str]:
    args = [
        "python3",
        "scripts/report/report_libero_runs.py",
        str(plan.run_dir),
        "--output-dir",
        str(plan.report_dir),
    ]
    if plan.min_success_rate is not None:
        args.extend(["--min-success-rate", str(plan.min_success_rate)])
    if plan.min_total_episodes is not None:
        args.extend(["--min-total-episodes", str(plan.min_total_episodes)])
    for baseline in plan.baseline:
        args.extend(["--baseline", baseline])
    if plan.baseline:
        args.extend(["--max-regression", str(plan.max_regression)])
    return args


def _command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _resolve_path(value: str | Path) -> Path:
    return Path(normalize_project_relative_path(value, REPO_ROOT))


def _rate(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError(f"{value!r} must be between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be non-negative")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError(f"{value!r} must be between 1 and 65535")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
