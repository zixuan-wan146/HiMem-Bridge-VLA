#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_libero_results import collect_result_rows, discover_result_files  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LIBERO result metrics against reproducibility gates.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Candidate LIBERO result JSON files, directories, run directories, or glob patterns.",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Metric scope to check, such as overall or suite:libero_spatial. Defaults to overall.",
    )
    parser.add_argument(
        "--min-success-rate",
        type=_rate,
        help="Fail when a checked scope has success_rate below this value.",
    )
    parser.add_argument(
        "--min-total-episodes",
        type=_non_negative_int,
        help="Fail when a checked scope has fewer total episodes than this value.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Baseline result JSON file, directory, or glob. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-regression",
        type=_rate,
        default=0.0,
        help="Allowed success_rate drop versus the best matching baseline scope. Defaults to 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scopes = args.scope or ["overall"]

    try:
        candidate_rows = collect_result_rows(discover_result_files(args.inputs))
        baseline_rows = collect_result_rows(discover_result_files(args.baseline)) if args.baseline else []
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    failures = check_metric_rows(
        candidate_rows,
        scopes=scopes,
        min_success_rate=args.min_success_rate,
        min_total_episodes=args.min_total_episodes,
        baseline_rows=baseline_rows,
        max_regression=args.max_regression,
    )
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1

    checked_rows = _rows_for_scopes(candidate_rows, scopes)
    for row in checked_rows:
        print(
            "[OK] "
            f"scope={row['scope']} "
            f"run={row['run_name']} "
            f"success_rate={float(row['success_rate']):.4f} "
            f"total_episodes={int(row['total_episodes'])}"
        )
    return 0


def check_metric_rows(
    candidate_rows: list[Mapping[str, Any]],
    *,
    scopes: Iterable[str],
    min_success_rate: float | None = None,
    min_total_episodes: int | None = None,
    baseline_rows: list[Mapping[str, Any]] | None = None,
    max_regression: float = 0.0,
) -> list[str]:
    failures: list[str] = []
    baseline_best = _best_baseline_by_scope(baseline_rows or [])
    for scope in scopes:
        rows = [row for row in candidate_rows if row.get("scope") == scope]
        if not rows:
            failures.append(f"no candidate rows found for scope={scope}")
            continue
        baseline_row = baseline_best.get(scope)
        for row in rows:
            label = f"scope={scope} run={row.get('run_name', '')}"
            success_rate = float(row.get("success_rate", 0.0))
            total_episodes = int(row.get("total_episodes", 0))
            if min_success_rate is not None and success_rate < min_success_rate:
                failures.append(
                    f"{label} success_rate={success_rate:.4f} is below minimum {min_success_rate:.4f}"
                )
            if min_total_episodes is not None and total_episodes < min_total_episodes:
                failures.append(
                    f"{label} total_episodes={total_episodes} is below minimum {min_total_episodes}"
                )
            if baseline_row is not None:
                baseline_rate = float(baseline_row.get("success_rate", 0.0))
                allowed = baseline_rate - max_regression
                if success_rate < allowed:
                    failures.append(
                        f"{label} success_rate={success_rate:.4f} regressed below "
                        f"baseline {baseline_rate:.4f} with tolerance {max_regression:.4f}"
                    )
    return failures


def _rows_for_scopes(rows: list[Mapping[str, Any]], scopes: Iterable[str]) -> list[Mapping[str, Any]]:
    scope_set = set(scopes)
    return [row for row in rows if row.get("scope") in scope_set]


def _best_baseline_by_scope(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    best: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        scope = str(row.get("scope", ""))
        current = best.get(scope)
        if current is None or float(row.get("success_rate", 0.0)) > float(current.get("success_rate", 0.0)):
            best[scope] = row
    return best


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
