from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TIMELINE_LIMIT = 200


def load_transition_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path).expanduser()
    rows: list[dict[str, Any]] = []
    with trace_path.open() as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace_path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{trace_path}:{line_number}: expected JSON object")
            payload.setdefault("source_file", str(trace_path))
            rows.append(payload)
    return rows


def summarize_transition_trace_files(
    paths: Sequence[str | Path],
    *,
    timeline_limit: int = DEFAULT_TIMELINE_LIMIT,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    files: list[str] = []
    for path in paths:
        trace_path = Path(path).expanduser()
        files.append(str(trace_path))
        rows.extend(load_transition_trace(trace_path))
    summary = summarize_transition_trace(rows, timeline_limit=timeline_limit)
    summary["files"] = files
    return summary


def summarize_transition_trace(
    rows: Sequence[Mapping[str, Any]],
    *,
    timeline_limit: int = DEFAULT_TIMELINE_LIMIT,
) -> dict[str, Any]:
    normalized_rows = [dict(row) for row in rows]
    episodes = _group_by_episode(normalized_rows)
    episode_summaries = [
        _summarize_episode(episode_rows, timeline_limit=timeline_limit)
        for _, episode_rows in sorted(episodes.items(), key=lambda item: item[0])
    ]

    summary = _summarize_rows(normalized_rows)
    summary["episodes"] = episode_summaries
    return summary


def format_transition_trace_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"trace_rows: {summary['trace_rows']}",
        f"ready_rows: {summary['ready_rows']}",
        f"scored_rows: {summary['scored_rows']}",
        "score_min / score_mean / score_max: "
        f"{_format_float(summary['score_min'])} / "
        f"{_format_float(summary['score_mean'])} / "
        f"{_format_float(summary['score_max'])}",
        f"soft_plan_count: {summary['soft_plan_count']}",
        f"hard_plan_count: {summary['hard_plan_count']}",
        f"memory_write_count: {summary['memory_write_count']}",
        f"should_plan_count: {summary['should_plan_count']}",
        f"chunk_shortened_count: {summary['chunk_shortened_count']}",
        f"error_rows: {summary['error_rows']}",
    ]
    if summary.get("files"):
        lines.append("files:")
        lines.extend(f"  - {path}" for path in summary["files"])
    lines.append("episodes:")
    for episode in summary["episodes"]:
        lines.append(
            "  - "
            f"{episode['episode_key']}: "
            f"rows={episode['trace_rows']}, "
            f"ready={episode['ready_rows']}, "
            f"scored={episode['scored_rows']}, "
            "score="
            f"{_format_float(episode['score_min'])}/"
            f"{_format_float(episode['score_mean'])}/"
            f"{_format_float(episode['score_max'])}, "
            f"soft/hard/write={episode['soft_plan_count']}/"
            f"{episode['hard_plan_count']}/"
            f"{episode['memory_write_count']}"
        )
        for event in episode["trigger_timeline"]:
            lines.append(
                "      "
                f"step={event['decision_step']} "
                f"control={event['control_step_before']} "
                f"score={_format_float(event.get('score'))} "
                f"soft={event.get('soft_plan')} "
                f"hard={event.get('hard_plan')} "
                f"write={event.get('memory_write')} "
                f"plan={event.get('should_plan')}"
            )
        if episode["trigger_timeline_truncated"]:
            lines.append(
                f"      ... {episode['trigger_timeline_truncated']} timeline rows omitted"
            )
    return "\n".join(lines)


def _group_by_episode(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_episode_key(row)].append(dict(row))
    return dict(grouped)


def _summarize_episode(
    rows: Sequence[Mapping[str, Any]],
    *,
    timeline_limit: int,
) -> dict[str, Any]:
    first_row = rows[0] if rows else {}
    summary = _summarize_rows(rows)
    summary.update(
        {
            "episode_key": _episode_key(first_row),
            "task_suite": str(first_row.get("task_suite", "")),
            "task_id": _int_or_none(first_row.get("task_id")),
            "episode_id": _int_or_none(first_row.get("episode_id")),
            "task_description": str(first_row.get("task_description", "")),
        }
    )
    timeline = [_timeline_event(row) for row in rows if _is_timeline_row(row)]
    summary["trigger_timeline"] = timeline[: max(timeline_limit, 0)]
    summary["trigger_timeline_rows"] = len(timeline)
    summary["trigger_timeline_truncated"] = max(0, len(timeline) - max(timeline_limit, 0))
    return summary


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [_float_or_none(row.get("score")) for row in rows]
    scored = [score for score in scores if score is not None]
    return {
        "trace_rows": len(rows),
        "ready_rows": _count_true(rows, "transition_ready"),
        "scored_rows": len(scored),
        "score_min": min(scored) if scored else None,
        "score_mean": _mean(scored),
        "score_max": max(scored) if scored else None,
        "soft_plan_count": _count_true(rows, "soft_plan"),
        "hard_plan_count": _count_true(rows, "hard_plan"),
        "memory_write_count": _count_true(rows, "memory_write"),
        "should_plan_count": _count_true(rows, "should_plan"),
        "chunk_shortened_count": _count_true(rows, "chunk_shortened"),
        "error_rows": sum(1 for row in rows if row.get("error")),
    }


def _timeline_event(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_step": _int_or_none(row.get("decision_step")),
        "control_step_before": _int_or_none(row.get("control_step_before")),
        "transition_frame_index": _int_or_none(row.get("transition_frame_index")),
        "score": _float_or_none(row.get("score")),
        "soft_plan": _bool_or_none(row.get("soft_plan")),
        "hard_plan": _bool_or_none(row.get("hard_plan")),
        "memory_write": _bool_or_none(row.get("memory_write")),
        "should_plan": _bool_or_none(row.get("should_plan")),
        "chunk_shortened": _bool_or_none(row.get("chunk_shortened")),
        "error": row.get("error"),
    }


def _is_timeline_row(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("transition_ready"))
        or bool(row.get("soft_plan"))
        or bool(row.get("hard_plan"))
        or bool(row.get("memory_write"))
        or bool(row.get("should_plan"))
        or bool(row.get("chunk_shortened"))
        or bool(row.get("error"))
    )


def _episode_key(row: Mapping[str, Any]) -> str:
    if row.get("episode_key"):
        return str(row["episode_key"])
    return ":".join(
        [
            str(row.get("task_suite", "unknown_suite")),
            f"task{row.get('task_id', 'unknown')}",
            f"episode{row.get('episode_id', 'unknown')}",
        ]
    )


def _count_true(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _format_float(value: Any) -> str:
    if value is None:
        return "none"
    return f"{float(value):.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize LIBERO transition trigger JSONL traces."
    )
    parser.add_argument("trace_files", nargs="+", help="One or more *_transition_trace.jsonl files.")
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format. Default: json.",
    )
    parser.add_argument(
        "--timeline-limit",
        type=int,
        default=DEFAULT_TIMELINE_LIMIT,
        help="Maximum per-episode timeline rows to include. Default: 200.",
    )
    args = parser.parse_args(argv)

    summary = summarize_transition_trace_files(
        args.trace_files,
        timeline_limit=args.timeline_limit,
    )
    if args.format == "text":
        print(format_transition_trace_summary(summary))
    else:
        json.dump(summary, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
