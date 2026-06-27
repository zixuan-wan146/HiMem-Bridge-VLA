#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


TABLE_COLUMNS = (
    "result_file",
    "run_name",
    "created_at_utc",
    "git_commit",
    "git_dirty",
    "scope",
    "total_episodes",
    "successful_episodes",
    "failed_episodes",
    "success_rate",
    "average_decision_steps",
    "average_control_steps",
    "average_success_decision_steps",
)

RUN_TABLE_COLUMNS = (
    "run_dir",
    "status",
    "run_kind",
    "run_name",
    "created_at_utc",
    "git_commit",
    "git_dirty",
    "task_suites",
    "episodes",
    "horizon",
    "max_steps",
    "manifest_file",
    "result_file",
    "total_episodes",
    "successful_episodes",
    "failed_episodes",
    "success_rate",
)


def discover_result_files(inputs: Iterable[str]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_input in inputs:
        matches = glob.glob(raw_input, recursive=True)
        candidate_paths = matches if matches else [raw_input]
        for candidate in candidate_paths:
            path = Path(candidate).expanduser()
            if path.is_dir():
                discovered.update(path.rglob("*_results.json"))
            elif path.is_file():
                discovered.add(path)
            else:
                raise FileNotFoundError(f"LIBERO result path not found: {raw_input}")
    return sorted(path.resolve() for path in discovered)


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        payload = json.load(f)

    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")

    config = payload.get("config", {})
    metadata = payload.get("metadata", {})
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"{path} has no summary object")

    run_name = _run_name(path, config)
    rows = [_summary_row(path, run_name, metadata, "overall", summary)]

    suites = summary.get("suites", {})
    if suites is not None and not isinstance(suites, Mapping):
        raise ValueError(f"{path} summary.suites must be an object")
    for suite_name, suite_summary in sorted((suites or {}).items()):
        if not isinstance(suite_summary, Mapping):
            raise ValueError(f"{path} suite summary {suite_name!r} must be an object")
        rows.append(_summary_row(path, run_name, metadata, f"suite:{suite_name}", suite_summary))

    return rows


def collect_result_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_result_rows(path))
    return rows


def discover_run_dirs(inputs: Iterable[str]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_input in inputs:
        matches = glob.glob(raw_input, recursive=True)
        candidate_paths = matches if matches else [raw_input]
        for candidate in candidate_paths:
            path = Path(candidate).expanduser()
            if path.is_dir():
                direct_manifest = _single_manifest_under(path)
                if direct_manifest is not None:
                    discovered.add(path)
                else:
                    discovered.update(manifest.parent for manifest in _manifest_files_under(path))
            elif path.is_file() and _looks_like_manifest(path):
                discovered.add(path.parent)
            else:
                raise FileNotFoundError(f"LIBERO run path not found: {raw_input}")
    return sorted(path.resolve() for path in discovered)


def collect_run_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [load_run_row(path) for path in paths]


def load_run_row(run_dir: Path) -> dict[str, Any]:
    row = _empty_run_row(run_dir)
    manifest_path = _single_manifest_under(run_dir)
    if manifest_path is None:
        row["status"] = "missing_manifest"
        return row
    row["manifest_file"] = str(manifest_path.resolve())

    try:
        with manifest_path.open("r") as f:
            manifest = json.load(f)
    except json.JSONDecodeError:
        row["status"] = "invalid_manifest"
        return row
    if not isinstance(manifest, Mapping):
        row["status"] = "invalid_manifest"
        return row

    metadata = manifest.get("metadata", {})
    git = metadata.get("git", {}) if isinstance(metadata, Mapping) else {}
    libero = manifest.get("libero", {})
    if not isinstance(libero, Mapping):
        row["status"] = "invalid_manifest"
        return row

    row.update(
        {
            "status": "manifest_only",
            "run_kind": str(manifest.get("run_kind", "")),
            "run_name": str(libero.get("HIMEM_LIBERO_CKPT_NAME", "")),
            "created_at_utc": metadata.get("created_at_utc", "") if isinstance(metadata, Mapping) else "",
            "git_commit": git.get("commit", "") if isinstance(git, Mapping) else "",
            "git_dirty": git.get("is_dirty", "") if isinstance(git, Mapping) else "",
            "task_suites": str(libero.get("HIMEM_LIBERO_TASK_SUITES", "")),
            "episodes": str(libero.get("HIMEM_LIBERO_EPISODES", "")),
            "horizon": str(libero.get("HIMEM_LIBERO_HORIZON", "")),
            "max_steps": str(libero.get("HIMEM_LIBERO_MAX_STEPS", "")),
        }
    )

    result_value = libero.get("HIMEM_LIBERO_RESULT_FILE")
    if not result_value:
        row["status"] = "missing_result_path"
        return row
    result_path = _resolve_artifact_path(run_dir, str(result_value))
    row["result_file"] = str(result_path)
    if not result_path.exists():
        row["status"] = "missing_result"
        return row

    try:
        with result_path.open("r") as f:
            result_payload = json.load(f)
    except json.JSONDecodeError:
        row["status"] = "invalid_result"
        return row
    if not isinstance(result_payload, Mapping):
        row["status"] = "invalid_result"
        return row

    summary = result_payload.get("summary")
    if not isinstance(summary, Mapping):
        row["status"] = "invalid_result"
        return row

    row.update(
        {
            "status": "complete",
            "total_episodes": int(summary.get("total_episodes", 0)),
            "successful_episodes": int(summary.get("successful_episodes", 0)),
            "failed_episodes": int(summary.get("failed_episodes", 0)),
            "success_rate": _float(summary.get("success_rate", 0.0)),
        }
    )
    return row


def write_csv(rows: list[Mapping[str, Any]], path: Path, columns: tuple[str, ...] = TABLE_COLUMNS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def write_markdown(rows: list[Mapping[str, Any]], path: Path, columns: tuple[str, ...] = TABLE_COLUMNS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_markdown_table(rows, columns=columns))
    return path


def format_markdown_table(rows: list[Mapping[str, Any]], columns: tuple[str, ...] = TABLE_COLUMNS) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + body) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one or more LIBERO *_results.json files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Result JSON files, directories, or glob patterns such as run_outputs/libero/log_file/*_results.json.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format.",
    )
    parser.add_argument(
        "--table",
        choices=("results", "runs"),
        default="results",
        help="Summarize result JSON rows or LIBERO run-directory inventory rows.",
    )
    parser.add_argument("--output", help="Optional output path. Defaults to stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.table == "runs":
            paths = discover_run_dirs(args.inputs)
            rows = collect_run_rows(paths)
            columns = RUN_TABLE_COLUMNS
            source_label = "run dir(s)"
        else:
            paths = discover_result_files(args.inputs)
            rows = collect_result_rows(paths)
            columns = TABLE_COLUMNS
            source_label = "result file(s)"
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output).expanduser()
        if args.format == "csv":
            write_csv(rows, output_path, columns=columns)
        else:
            write_markdown(rows, output_path, columns=columns)
        print(f"Wrote {len(rows)} row(s) from {len(paths)} {source_label} to {output_path}")
    else:
        if args.format == "csv":
            writer = csv.DictWriter(_StdoutWriter(), fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        else:
            print(format_markdown_table(rows, columns=columns), end="")

    return 0


def _summary_row(
    path: Path,
    run_name: str,
    metadata: Any,
    scope: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    git = metadata.get("git", {}) if isinstance(metadata, Mapping) else {}
    return {
        "result_file": str(path),
        "run_name": run_name,
        "created_at_utc": metadata.get("created_at_utc", "") if isinstance(metadata, Mapping) else "",
        "git_commit": git.get("commit", "") if isinstance(git, Mapping) else "",
        "git_dirty": git.get("is_dirty", "") if isinstance(git, Mapping) else "",
        "scope": scope,
        "total_episodes": int(summary.get("total_episodes", 0)),
        "successful_episodes": int(summary.get("successful_episodes", 0)),
        "failed_episodes": int(summary.get("failed_episodes", 0)),
        "success_rate": _float(summary.get("success_rate", 0.0)),
        "average_decision_steps": _float(summary.get("average_decision_steps", 0.0)),
        "average_control_steps": _float(summary.get("average_control_steps", 0.0)),
        "average_success_decision_steps": _float(summary.get("average_success_decision_steps", 0.0)),
    }


def _run_name(path: Path, config: Any) -> str:
    if isinstance(config, Mapping):
        for key in ("ckpt_name", "run_name", "checkpoint", "ckpt_dir"):
            value = config.get(key)
            if value:
                return str(value)
    return path.stem.replace("_results", "")


def _empty_run_row(run_dir: Path) -> dict[str, Any]:
    return {column: "" for column in RUN_TABLE_COLUMNS} | {
        "run_dir": str(run_dir.resolve()),
        "status": "unknown",
    }


def _manifest_files_under(path: Path) -> list[Path]:
    manifests = set(path.rglob("run_manifest.json"))
    manifests.update(path.rglob("*_run_manifest.json"))
    return sorted(manifests)


def _single_manifest_under(path: Path) -> Path | None:
    manifests = []
    direct_manifest = path / "run_manifest.json"
    if direct_manifest.is_file():
        manifests.append(direct_manifest)
    manifests.extend(sorted(path.glob("*_run_manifest.json")))
    unique_manifests = sorted({manifest.resolve() for manifest in manifests})
    if len(unique_manifests) == 1:
        return unique_manifests[0]
    return None


def _looks_like_manifest(path: Path) -> bool:
    return path.name == "run_manifest.json" or path.name.endswith("_run_manifest.json")


def _resolve_artifact_path(run_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        rendered = f"{value:.4f}"
    else:
        rendered = str(value)
    return rendered.replace("|", "\\|").replace("\n", " ")


class _StdoutWriter:
    def write(self, value: str) -> int:
        print(value, end="")
        return len(value)


if __name__ == "__main__":
    raise SystemExit(main())
