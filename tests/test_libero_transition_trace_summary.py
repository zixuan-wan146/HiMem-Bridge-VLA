from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from evaluations.libero.libero_transition_trace_summary import (
    format_transition_trace_summary,
    load_transition_trace,
    main,
    summarize_transition_trace,
    summarize_transition_trace_files,
)


def _row(
    *,
    episode_key: str = "libero_spatial:task0:episode0",
    decision_step: int,
    score: float | None,
    transition_ready: bool = False,
    soft_plan: bool = False,
    hard_plan: bool = False,
    memory_write: bool = False,
    should_plan: bool = False,
    chunk_shortened: bool = False,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "task_suite": "libero_spatial",
        "task_id": 0,
        "episode_id": 0 if episode_key.endswith("episode0") else 1,
        "episode_key": episode_key,
        "task_description": "put the object in the bowl",
        "decision_step": decision_step,
        "control_step_before": decision_step - 1,
        "transition_frame_index": decision_step - 1 if transition_ready else None,
        "transition_ready": transition_ready,
        "score": score,
        "soft_plan": soft_plan,
        "hard_plan": hard_plan,
        "memory_write": memory_write,
        "should_plan": should_plan,
        "chunk_shortened": chunk_shortened,
        "error": error,
    }


def test_summarize_transition_trace_reports_trigger_counts_and_episode_timeline():
    rows = [
        _row(decision_step=1, score=None),
        _row(decision_step=2, score=0.62, transition_ready=True),
        _row(
            decision_step=3,
            score=0.75,
            transition_ready=True,
            soft_plan=True,
            should_plan=True,
            chunk_shortened=True,
        ),
        _row(
            episode_key="libero_spatial:task0:episode1",
            decision_step=1,
            score=0.82,
            transition_ready=True,
            hard_plan=True,
            memory_write=True,
            should_plan=True,
        ),
    ]

    summary = summarize_transition_trace(rows)

    assert summary["trace_rows"] == 4
    assert summary["ready_rows"] == 3
    assert summary["scored_rows"] == 3
    assert summary["score_min"] == 0.62
    assert summary["score_mean"] == (0.62 + 0.75 + 0.82) / 3
    assert summary["score_max"] == 0.82
    assert summary["soft_plan_count"] == 1
    assert summary["hard_plan_count"] == 1
    assert summary["memory_write_count"] == 1
    assert summary["should_plan_count"] == 2
    assert summary["chunk_shortened_count"] == 1
    assert [episode["episode_key"] for episode in summary["episodes"]] == [
        "libero_spatial:task0:episode0",
        "libero_spatial:task0:episode1",
    ]
    assert summary["episodes"][0]["trigger_timeline_rows"] == 2
    assert summary["episodes"][0]["trigger_timeline"][1]["decision_step"] == 3


def test_summarize_transition_trace_respects_timeline_limit():
    rows = [
        _row(decision_step=1, score=0.5, transition_ready=True),
        _row(decision_step=2, score=0.6, transition_ready=True),
    ]

    summary = summarize_transition_trace(rows, timeline_limit=1)

    episode = summary["episodes"][0]
    assert episode["trigger_timeline_rows"] == 2
    assert len(episode["trigger_timeline"]) == 1
    assert episode["trigger_timeline_truncated"] == 1


def test_load_and_summarize_transition_trace_files(tmp_path: Path):
    trace_path = tmp_path / "run_transition_trace.jsonl"
    rows = [
        _row(decision_step=1, score=None),
        _row(decision_step=2, score=0.7, transition_ready=True, soft_plan=True),
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    loaded_rows = load_transition_trace(trace_path)
    summary = summarize_transition_trace_files([trace_path])

    assert len(loaded_rows) == 2
    assert loaded_rows[0]["source_file"] == str(trace_path)
    assert summary["files"] == [str(trace_path)]
    assert summary["soft_plan_count"] == 1


def test_format_transition_trace_summary_includes_compact_counts():
    summary = summarize_transition_trace(
        [_row(decision_step=2, score=0.7, transition_ready=True, soft_plan=True)]
    )

    text = format_transition_trace_summary(summary)

    assert "trace_rows: 1" in text
    assert "score_min / score_mean / score_max: 0.7000 / 0.7000 / 0.7000" in text
    assert "soft_plan_count: 1" in text
    assert "libero_spatial:task0:episode0" in text


def test_transition_trace_summary_cli_prints_json_and_text(tmp_path: Path, capsys):
    trace_path = tmp_path / "run_transition_trace.jsonl"
    trace_path.write_text(json.dumps(_row(decision_step=2, score=0.7, transition_ready=True)) + "\n")

    assert main([str(trace_path)]) == 0
    json_output = capsys.readouterr().out
    assert json.loads(json_output)["trace_rows"] == 1

    assert main([str(trace_path), "--format", "text"]) == 0
    text_output = capsys.readouterr().out
    assert "ready_rows: 1" in text_output


def test_transition_trace_summary_script_runs_from_repo_root(tmp_path: Path):
    trace_path = tmp_path / "run_transition_trace.jsonl"
    trace_path.write_text(json.dumps(_row(decision_step=2, score=0.7, transition_ready=True)) + "\n")
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_libero_transition_trace.py",
            str(trace_path),
            "--format",
            "text",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "trace_rows: 1" in result.stdout
