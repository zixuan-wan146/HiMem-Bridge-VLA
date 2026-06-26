#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.training import ProgressWarmupTrainingConfig  # noqa: E402
from himem_bridge_vla.training import run_progress_warmup_training  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up the progress-state long memory and planner.")
    parser.add_argument("--cache-manifest", required=True, help="Warm-up cache directory or manifest.json path.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--samples-per-epoch", type=int, default=8192)
    parser.add_argument("--sampling-alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--replan-stride", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--action-summary-hidden-dim", type=int, default=512)
    parser.add_argument("--state-hidden-dim", type=int, default=512)
    parser.add_argument("--updater-hidden-dim", type=int, default=1792)
    parser.add_argument("--planner-ffn-dim", type=int, default=3584)
    parser.add_argument("--planner-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lambda-plan", type=float, default=1.0)
    parser.add_argument("--lambda-stage", type=float, default=0.5)
    parser.add_argument("--lambda-mem-pool", type=float, default=0.1)
    parser.add_argument("--lambda-order", type=float, default=0.02)
    parser.add_argument("--use-order-loss", action="store_true")
    parser.add_argument("--min-order-gap", type=int, default=2)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-by-window", action="store_true", help="Split validation by window instead of by episode.")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--ckpt-interval", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProgressWarmupTrainingConfig(
        cache_manifest=args.cache_manifest,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        samples_per_epoch=args.samples_per_epoch,
        sampling_alpha=args.sampling_alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        seed=args.seed,
        deterministic=args.deterministic,
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        replan_stride=args.replan_stride,
        latent_dim=args.latent_dim,
        action_summary_hidden_dim=args.action_summary_hidden_dim,
        state_hidden_dim=args.state_hidden_dim,
        updater_hidden_dim=args.updater_hidden_dim,
        planner_ffn_dim=args.planner_ffn_dim,
        planner_layers=args.planner_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        lambda_plan=args.lambda_plan,
        lambda_stage=args.lambda_stage,
        lambda_mem_pool=args.lambda_mem_pool,
        lambda_order=args.lambda_order,
        use_order_loss=args.use_order_loss,
        min_order_gap=args.min_order_gap,
        cosine_weight=args.cosine_weight,
        val_fraction=args.val_fraction,
        split_by_episode=not args.split_by_window,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
        max_val_batches=args.max_val_batches,
        log_interval=args.log_interval,
        ckpt_interval=args.ckpt_interval,
        repo_root=str(REPO_ROOT),
    )
    result = run_progress_warmup_training(config)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "checkpoint_path": str(result.checkpoint_path),
                "best_checkpoint_path": str(result.best_checkpoint_path),
                "steps": result.steps,
                "final_loss": result.final_loss,
                "best_loss": result.best_loss,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
