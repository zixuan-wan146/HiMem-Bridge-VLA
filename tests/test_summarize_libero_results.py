from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


def load_summary_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_libero_results.py"
    spec = importlib.util.spec_from_file_location("summarize_libero_results", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_result_file(path: Path, ckpt_name: str = "run_a") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": {"ckpt_name": ckpt_name},
                "metadata": {
                    "created_at_utc": "2026-06-11T00:00:00Z",
                    "git": {"commit": "abc123", "is_dirty": False},
                },
                "summary": {
                    "total_episodes": 3,
                    "successful_episodes": 2,
                    "failed_episodes": 1,
                    "success_rate": 2 / 3,
                    "average_decision_steps": 4.5,
                    "average_control_steps": 60.0,
                    "average_success_decision_steps": 3.0,
                    "suites": {
                        "libero_spatial": {
                            "total_episodes": 1,
                            "successful_episodes": 1,
                            "failed_episodes": 0,
                            "success_rate": 1.0,
                            "average_decision_steps": 2.0,
                            "average_control_steps": 28.0,
                            "average_success_decision_steps": 2.0,
                        },
                        "libero_goal": {
                            "total_episodes": 2,
                            "successful_episodes": 1,
                            "failed_episodes": 1,
                            "success_rate": 0.5,
                            "average_decision_steps": 5.75,
                            "average_control_steps": 76.0,
                            "average_success_decision_steps": 4.0,
                        },
                    },
                },
                "episodes": [],
            }
        )
    )
    return path


def write_manifest_file(path: Path, result_file: Path, ckpt_name: str = "run_a") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_kind": "smoke",
                "metadata": {
                    "created_at_utc": "2026-06-11T00:00:00Z",
                    "git": {"commit": "abc123", "is_dirty": False},
                },
                "libero": {
                    "HIMEM_LIBERO_CKPT_NAME": ckpt_name,
                    "HIMEM_LIBERO_TASK_SUITES": "libero_spatial",
                    "HIMEM_LIBERO_EPISODES": "1",
                    "HIMEM_LIBERO_HORIZON": "14",
                    "HIMEM_LIBERO_MAX_STEPS": "25",
                    "HIMEM_LIBERO_RESULT_FILE": str(result_file),
                },
            }
        )
    )
    return path


def write_run_dir(path: Path, ckpt_name: str = "run_a", with_result: bool = True) -> Path:
    result_file = path / "results" / f"{ckpt_name}_results.json"
    if with_result:
        write_result_file(result_file, ckpt_name)
    write_manifest_file(path / "run_manifest.json", result_file, ckpt_name)
    return path


def test_discover_result_files_accepts_directories_and_globs(tmp_path):
    module = load_summary_module()
    first = write_result_file(tmp_path / "run_a_results.json", "run_a")
    second = write_result_file(tmp_path / "nested" / "run_b_results.json", "run_b")

    by_directory = module.discover_result_files([str(tmp_path)])
    by_glob = module.discover_result_files([str(tmp_path / "**" / "*_results.json")])

    assert set(by_directory) == {first.resolve(), second.resolve()}
    assert set(by_glob) == {first.resolve(), second.resolve()}


def test_load_result_rows_expands_overall_and_suite_rows(tmp_path):
    module = load_summary_module()
    result_file = write_result_file(tmp_path / "run_a_results.json")

    rows = module.load_result_rows(result_file)

    assert [row["scope"] for row in rows] == ["overall", "suite:libero_goal", "suite:libero_spatial"]
    assert rows[0]["run_name"] == "run_a"
    assert rows[0]["created_at_utc"] == "2026-06-11T00:00:00Z"
    assert rows[0]["git_commit"] == "abc123"
    assert rows[0]["git_dirty"] is False
    assert rows[0]["success_rate"] == 2 / 3
    assert rows[1]["successful_episodes"] == 1


def test_write_markdown_and_csv_tables(tmp_path):
    module = load_summary_module()
    rows = module.load_result_rows(write_result_file(tmp_path / "run_a_results.json"))
    markdown_path = tmp_path / "summary.md"
    csv_path = tmp_path / "summary.csv"

    module.write_markdown(rows, markdown_path)
    module.write_csv(rows, csv_path)

    markdown = markdown_path.read_text()
    assert "| result_file | run_name | created_at_utc | git_commit | git_dirty | scope |" in markdown
    assert "suite:libero_spatial" in markdown

    csv_rows = list(csv.DictReader(csv_path.open()))
    assert csv_rows[0]["scope"] == "overall"
    assert csv_rows[0]["run_name"] == "run_a"
    assert csv_rows[0]["git_commit"] == "abc123"
    assert csv_rows[0]["total_episodes"] == "3"


def test_main_reports_missing_inputs_without_traceback(capsys):
    module = load_summary_module()

    exit_code = module.main(["run_outputs/definitely_missing_libero_results.json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("ERROR: LIBERO result path not found:")
    assert "Traceback" not in captured.err


def test_discover_run_dirs_accepts_parent_directories_and_manifest_files(tmp_path):
    module = load_summary_module()
    first = write_run_dir(tmp_path / "run_a", "run_a")
    second = write_run_dir(tmp_path / "nested" / "run_b", "run_b")

    by_parent = module.discover_run_dirs([str(tmp_path)])
    by_manifest = module.discover_run_dirs([str(first / "run_manifest.json")])

    assert set(by_parent) == {first.resolve(), second.resolve()}
    assert by_manifest == [first.resolve()]


def test_load_run_row_reports_complete_run_metrics(tmp_path):
    module = load_summary_module()
    run_dir = write_run_dir(tmp_path / "run_a", "run_a")

    row = module.load_run_row(run_dir)

    assert row["status"] == "complete"
    assert row["run_name"] == "run_a"
    assert row["run_kind"] == "smoke"
    assert row["git_commit"] == "abc123"
    assert row["git_dirty"] is False
    assert row["task_suites"] == "libero_spatial"
    assert row["episodes"] == "1"
    assert row["horizon"] == "14"
    assert row["max_steps"] == "25"
    assert row["total_episodes"] == 3
    assert row["successful_episodes"] == 2
    assert row["success_rate"] == 2 / 3


def test_load_run_row_reports_manifest_only_runs(tmp_path):
    module = load_summary_module()
    run_dir = write_run_dir(tmp_path / "run_a", "run_a", with_result=False)

    row = module.load_run_row(run_dir)

    assert row["status"] == "missing_result"
    assert row["run_name"] == "run_a"
    assert row["total_episodes"] == ""


def test_write_run_inventory_markdown_and_csv(tmp_path):
    module = load_summary_module()
    rows = module.collect_run_rows([write_run_dir(tmp_path / "run_a", "run_a")])
    markdown_path = tmp_path / "runs.md"
    csv_path = tmp_path / "runs.csv"

    module.write_markdown(rows, markdown_path, columns=module.RUN_TABLE_COLUMNS)
    module.write_csv(rows, csv_path, columns=module.RUN_TABLE_COLUMNS)

    markdown = markdown_path.read_text()
    assert "| run_dir | status | run_kind | run_name |" in markdown
    assert "complete" in markdown

    csv_rows = list(csv.DictReader(csv_path.open()))
    assert csv_rows[0]["status"] == "complete"
    assert csv_rows[0]["run_name"] == "run_a"


def test_main_can_write_run_inventory(tmp_path, capsys):
    module = load_summary_module()
    write_run_dir(tmp_path / "run_a", "run_a")
    output_path = tmp_path / "runs.csv"

    exit_code = module.main(
        [
            str(tmp_path),
            "--table",
            "runs",
            "--format",
            "csv",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Wrote 1 row(s)" in captured.out
    csv_rows = list(csv.DictReader(output_path.open()))
    assert csv_rows[0]["status"] == "complete"
