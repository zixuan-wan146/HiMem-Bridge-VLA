#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.training import MemoryTokenCacheTrainingConfig  # noqa: E402
from himem_bridge_vla.training import run_memory_token_cache_training  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small action adapter from replay visual-token cache.")
    parser.add_argument("--cache-manifest", required=True, help="Token cache directory or manifest.json path.")
    parser.add_argument("--output-dir", required=True, help="Directory for snapshot, metrics, and adapter checkpoint.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tokens-per-entry", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden-multiplier", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--ckpt-interval", type=int, default=0)
    parser.add_argument("--view-names", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = MemoryTokenCacheTrainingConfig(
        cache_manifest=args.cache_manifest,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
        deterministic=args.deterministic,
        tokens_per_entry=args.tokens_per_entry,
        num_heads=args.num_heads,
        dropout=args.dropout,
        hidden_multiplier=args.hidden_multiplier,
        log_interval=args.log_interval,
        ckpt_interval=args.ckpt_interval,
        view_names=None if args.view_names is None else tuple(args.view_names),
        repo_root=str(REPO_ROOT),
    )
    result = run_memory_token_cache_training(config)
    payload = {
        "output_dir": str(result.output_dir),
        "checkpoint_path": str(result.checkpoint_path),
        "steps": result.steps,
        "final_loss": result.final_loss,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
