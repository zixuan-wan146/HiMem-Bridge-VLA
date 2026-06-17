from __future__ import annotations

import json
from pathlib import Path

from evaluations.libero.libero_transition_trace import (
    append_transition_trace,
    build_transition_error_trace_record,
    build_transition_trace_record,
)


def test_build_transition_trace_record_flattens_trigger_metadata():
    record = build_transition_trace_record(
        task_suite="libero_spatial",
        task_id=0,
        episode_id=1,
        episode_key="libero_spatial:task0:episode1",
        task_description="put the bowl on the plate",
        decision_step=32,
        control_step_before=31,
        transition_frame_index=31,
        reset_transition_trigger=False,
        has_transition_frame=True,
        transition_trigger={
            "ready": True,
            "score": 0.74,
            "soft_plan": True,
            "hard_plan": False,
            "memory_write": False,
            "should_plan": True,
        },
        raw_action_chunk_len=8,
        executed_action_chunk_len=1,
        replan_action_limit=1,
    )

    assert record["transition_ready"] is True
    assert record["score"] == 0.74
    assert record["soft_plan"] is True
    assert record["memory_write"] is False
    assert record["chunk_shortened"] is True
    assert record["raw_action_chunk_len"] == 8
    assert record["executed_action_chunk_len"] == 1


def test_append_transition_trace_writes_jsonl(tmp_path: Path):
    trace_path = tmp_path / "trace" / "transition.jsonl"
    append_transition_trace(trace_path, {"decision_step": 0, "score": None})
    append_transition_trace(trace_path, {"decision_step": 1, "score": 0.5})

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]

    assert rows == [
        {"decision_step": 0, "score": None},
        {"decision_step": 1, "score": 0.5},
    ]


def test_build_transition_error_trace_record_limits_response_preview():
    record = build_transition_error_trace_record(
        task_suite="libero_spatial",
        task_id=0,
        episode_id=0,
        episode_key="episode",
        task_description="task",
        decision_step=3,
        control_step_before=2,
        transition_frame_index=None,
        reset_transition_trigger=False,
        has_transition_frame=False,
        error=ValueError("bad response"),
        response_preview="x" * 600,
    )

    assert record["error"] == "bad response"
    assert len(record["response_preview"]) == 500
