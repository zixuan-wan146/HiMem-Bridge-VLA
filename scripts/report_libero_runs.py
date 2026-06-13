#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_libero_metrics import check_metric_rows  # noqa: E402
from summarize_libero_results import (  # noqa: E402
    RUN_TABLE_COLUMNS,
    TABLE_COLUMNS,
    collect_result_rows,
    collect_run_rows,
    discover_result_files,
    discover_run_dirs,
    write_csv,
    write_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a LIBERO run report directory.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="LIBERO run directories, result directories, result JSON files, or glob patterns.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated report files.")
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Metric scope for the optional gate. Defaults to overall when a gate option is used.",
    )
    parser.add_argument("--min-success-rate", type=_rate, help="Optional success_rate gate.")
    parser.add_argument("--min-total-episodes", type=_non_negative_int, help="Optional total episode gate.")
    parser.add_argument("--baseline", action="append", default=[], help="Optional baseline result/run input.")
    parser.add_argument(
        "--max-regression",
        type=_rate,
        default=0.0,
        help="Allowed success_rate drop versus the best matching baseline scope. Defaults to 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = write_report(
            args.inputs,
            output_dir=Path(args.output_dir).expanduser(),
            scopes=args.scope,
            min_success_rate=args.min_success_rate,
            min_total_episodes=args.min_total_episodes,
            baseline_inputs=args.baseline,
            max_regression=args.max_regression,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    for path in report["files"]:
        print(f"[OK] wrote {path}")
    if report["metrics_gate"]["enabled"] and not report["metrics_gate"]["passed"]:
        for failure in report["metrics_gate"]["failures"]:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1
    return 0


def write_report(
    inputs: Iterable[str],
    *,
    output_dir: Path,
    scopes: list[str] | None = None,
    min_success_rate: float | None = None,
    min_total_episodes: int | None = None,
    baseline_inputs: list[str] | None = None,
    max_regression: float = 0.0,
) -> dict[str, Any]:
    input_list = list(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = _try_discover_run_dirs(input_list)
    result_files = discover_result_files(input_list)

    if not run_dirs and not result_files:
        raise FileNotFoundError("no LIBERO run directories or result files found")

    files: list[str] = []
    run_rows = collect_run_rows(run_dirs) if run_dirs else []
    if run_rows:
        files.append(str(write_markdown(run_rows, output_dir / "run_inventory.md", columns=RUN_TABLE_COLUMNS)))
        files.append(str(write_csv(run_rows, output_dir / "run_inventory.csv", columns=RUN_TABLE_COLUMNS)))

    result_rows = collect_result_rows(result_files) if result_files else []
    if result_rows:
        files.append(str(write_markdown(result_rows, output_dir / "result_summary.md", columns=TABLE_COLUMNS)))
        files.append(str(write_csv(result_rows, output_dir / "result_summary.csv", columns=TABLE_COLUMNS)))

    gate = _run_metric_gate(
        result_rows,
        scopes=scopes or [],
        min_success_rate=min_success_rate,
        min_total_episodes=min_total_episodes,
        baseline_inputs=baseline_inputs or [],
        max_regression=max_regression,
    )
    if gate["enabled"]:
        gate_path = output_dir / "metrics_gate.txt"
        gate_path.write_text(_format_gate(gate))
        files.append(str(gate_path))

    readme_path = output_dir / "README.md"
    manifest_path = output_dir / "report_manifest.json"
    files.append(
        str(
            _write_report_readme(
                readme_path,
                input_count=len(input_list),
                run_dir_count=len(run_dirs),
                result_file_count=len(result_files),
                generated_files=[*files, str(manifest_path)],
                gate=gate,
            )
        )
    )

    manifest = {
        "input_count": len(input_list),
        "run_dir_count": len(run_dirs),
        "result_file_count": len(result_files),
        "files": files,
        "metrics_gate": gate,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    files.append(str(manifest_path))
    manifest["files"] = files
    return manifest


def _try_discover_run_dirs(inputs: Iterable[str]) -> list[Path]:
    run_dirs: set[Path] = set()
    for raw_input in inputs:
        try:
            run_dirs.update(discover_run_dirs([raw_input]))
        except FileNotFoundError:
            continue
    return sorted(run_dirs)


def _run_metric_gate(
    result_rows: list[Mapping[str, Any]],
    *,
    scopes: list[str],
    min_success_rate: float | None,
    min_total_episodes: int | None,
    baseline_inputs: list[str],
    max_regression: float,
) -> dict[str, Any]:
    enabled = (
        min_success_rate is not None
        or min_total_episodes is not None
        or bool(baseline_inputs)
    )
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "scopes": [],
            "failures": [],
        }

    gate_scopes = scopes or ["overall"]
    baseline_rows = collect_result_rows(discover_result_files(baseline_inputs)) if baseline_inputs else []
    failures = check_metric_rows(
        result_rows,
        scopes=gate_scopes,
        min_success_rate=min_success_rate,
        min_total_episodes=min_total_episodes,
        baseline_rows=baseline_rows,
        max_regression=max_regression,
    )
    return {
        "enabled": True,
        "passed": not failures,
        "scopes": gate_scopes,
        "min_success_rate": min_success_rate,
        "min_total_episodes": min_total_episodes,
        "baseline_input_count": len(baseline_inputs),
        "max_regression": max_regression,
        "failures": failures,
    }


def _format_gate(gate: Mapping[str, Any]) -> str:
    lines = [
        f"enabled: {gate['enabled']}",
        f"passed: {gate['passed']}",
        f"scopes: {', '.join(gate.get('scopes', []))}",
    ]
    failures = gate.get("failures", [])
    if failures:
        lines.append("failures:")
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n"


def _write_report_readme(
    path: Path,
    *,
    input_count: int,
    run_dir_count: int,
    result_file_count: int,
    generated_files: list[str],
    gate: Mapping[str, Any],
) -> Path:
    lines = [
        "# LIBERO Run Report",
        "",
        "## Inputs",
        "",
        f"- Input patterns: {input_count}",
        f"- Run directories: {run_dir_count}",
        f"- Result files: {result_file_count}",
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{Path(file_path).name}`" for file_path in generated_files)
    lines.extend(
        [
            "",
            "## Metric Gate",
            "",
            f"- Enabled: {gate['enabled']}",
            f"- Passed: {gate['passed']}",
        ]
    )
    scopes = gate.get("scopes", [])
    if scopes:
        lines.append(f"- Scopes: {', '.join(str(scope) for scope in scopes)}")
    failures = gate.get("failures", [])
    if failures:
        lines.extend(["", "Failures:"])
        lines.extend(f"- {failure}" for failure in failures)
    path.write_text("\n".join(lines) + "\n")
    return path


def _rate(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a float") from exc
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError(f"{value!r} must be between 0 and 1")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
