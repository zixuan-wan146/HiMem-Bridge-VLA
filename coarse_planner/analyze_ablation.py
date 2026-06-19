from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LIBERO CoarsePlanner horizon ablation runs.")
    parser.add_argument("--run", action="append", default=[], help="Run dir. Repeat for H32/H48/H64.")
    parser.add_argument("--runs", nargs="*", default=[], help="Run dirs. Alternative to repeated --run.")
    parser.add_argument("--output", default=None, help="Optional markdown report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [Path(path).expanduser() for path in [*args.run, *args.runs]]
    if not run_dirs:
        raise ValueError("provide at least one --run or --runs path")
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    rows.sort(key=lambda item: int(item["planning_horizon"]))
    report = format_report(rows)
    print(report)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
    return 0


def summarize_run(run_dir: Path) -> dict[str, Any]:
    history_path = run_dir / "train_history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.exists():
        raise FileNotFoundError(f"missing train history: {history_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing best checkpoint: {checkpoint_path}")
    history = json.loads(history_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    planner_config = checkpoint.get("planner_config", {})
    val_metrics = checkpoint.get("val_metrics", {})
    best_epoch = min(history, key=lambda item: float(item.get("val_loss", float("inf")))) if history else {}
    last_epoch = history[-1] if history else {}
    return {
        "run_dir": str(run_dir),
        "planning_horizon": int(planner_config.get("planning_horizon", 0)),
        "num_plan_steps": int(planner_config.get("num_plan_steps", 0)),
        "best_val_loss": float(val_metrics.get("loss", best_epoch.get("val_loss", 0.0))),
        "best_val_latent_mse": float(val_metrics.get("latent_mse", best_epoch.get("val_latent_mse", 0.0))),
        "best_epoch": int(best_epoch.get("epoch", 0)),
        "last_train_loss": float(last_epoch.get("train_loss", 0.0)),
        "peak_reserved_gb": float(max((item.get("cuda_peak_reserved_gb", 0.0) for item in history), default=0.0)),
        "peak_allocated_gb": float(max((item.get("cuda_peak_allocated_gb", 0.0) for item in history), default=0.0)),
        "epochs": len(history),
    }


def format_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LIBERO Coarse Planner Horizon Ablation",
        "",
        "| horizon | K | best epoch | val loss | val latent MSE | train loss last | peak reserved GB | run |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {planning_horizon} | {num_plan_steps} | {best_epoch} | "
            "{best_val_loss:.6f} | {best_val_latent_mse:.6f} | {last_train_loss:.6f} | "
            "{peak_reserved_gb:.3f} | `{run_dir}` |".format(**row)
        )
    best = min(rows, key=lambda item: float(item["best_val_loss"]))
    lines.extend(
        [
            "",
            "Recommended by validation loss: "
            f"H={best['planning_horizon']}, K={best['num_plan_steps']} "
            f"(val loss {best['best_val_loss']:.6f}).",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
