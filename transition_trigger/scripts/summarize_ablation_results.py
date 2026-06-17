from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize transition trigger ablation metrics.")
    parser.add_argument("--run-root", default="/root/autodl-tmp/runs/transition_trigger/robomme_rmbench_ablations")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    rows = collect_rows(run_root)
    rows.sort(
        key=lambda row: (
            float(row.get("eval_best_event_f1") or -1.0),
            float(row.get("test_best_event_f1") or -1.0),
            float(row.get("eval_auprc") or -1.0),
        ),
        reverse=True,
    )
    text = to_tsv(rows)
    output_path = Path(args.output) if args.output else run_root / "ablation_summary.tsv"
    output_path.write_text(text)
    print(text)
    return 0


def collect_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        checkpoint_names = sorted(
            {
                path.name.removeprefix("eval_metrics_").removesuffix(".json")
                for path in run_dir.glob("eval_metrics_*.json")
            }
        )
        for checkpoint_name in checkpoint_names:
            eval_path = run_dir / f"eval_metrics_{checkpoint_name}.json"
            test_path = run_dir / f"test_metrics_{checkpoint_name}.json"
            if not eval_path.exists() or not test_path.exists():
                continue
            eval_metrics = json.loads(eval_path.read_text())
            test_metrics = json.loads(test_path.read_text())
            row = {
                "run_name": run_dir.name,
                "checkpoint": checkpoint_name,
                "eval_auprc": eval_metrics.get("auprc"),
                "test_auprc": test_metrics.get("auprc"),
            }
            row.update(prefix_metrics("eval", eval_metrics))
            row.update(prefix_metrics("test", test_metrics))
            rows.append(row)
    return rows


def prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    thresholds = metrics.get("thresholds") or {}
    best = thresholds.get("best_f1_metrics") or {}
    memory = thresholds.get("memory_write_metrics") or {}
    replan = thresholds.get("replan_metrics") or {}
    return {
        f"{prefix}_best_event_f1": best.get("f1"),
        f"{prefix}_best_event_precision": best.get("precision"),
        f"{prefix}_best_event_recall": best.get("recall"),
        f"{prefix}_best_event_threshold": thresholds.get("best_f1_threshold"),
        f"{prefix}_memory_f1": memory.get("f1"),
        f"{prefix}_memory_precision": memory.get("precision"),
        f"{prefix}_memory_recall": memory.get("recall"),
        f"{prefix}_memory_threshold": thresholds.get("memory_write_threshold"),
        f"{prefix}_replan_f1": replan.get("f1"),
        f"{prefix}_replan_precision": replan.get("precision"),
        f"{prefix}_replan_recall": replan.get("recall"),
        f"{prefix}_replan_threshold": thresholds.get("replan_threshold"),
    }


def to_tsv(rows: list[dict[str, Any]]) -> str:
    columns = [
        "run_name",
        "checkpoint",
        "eval_auprc",
        "eval_best_event_f1",
        "eval_best_event_precision",
        "eval_best_event_recall",
        "eval_best_event_threshold",
        "test_auprc",
        "test_best_event_f1",
        "test_best_event_precision",
        "test_best_event_recall",
        "test_best_event_threshold",
        "eval_memory_f1",
        "eval_memory_precision",
        "eval_memory_recall",
        "test_memory_f1",
        "test_memory_precision",
        "test_memory_recall",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(format_value(row.get(column)) for column in columns))
    return "\n".join(lines) + "\n"


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
