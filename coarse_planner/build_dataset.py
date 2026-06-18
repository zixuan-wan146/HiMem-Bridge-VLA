from __future__ import annotations

import argparse
import json
from pathlib import Path

from coarse_planner.config import load_config
from coarse_planner.data import build_planner_feature_cache, build_synthetic_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone CoarsePlanner feature cache.")
    parser.add_argument("--config", default="coarse_planner/configs/default.yaml")
    parser.add_argument("--output", default=None, help="Override data.root.")
    parser.add_argument("--input", action="append", default=None, help="Feature source .pt/.npz path; can repeat.")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Build a tiny synthetic cache for pipeline checks.")
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.output:
        config["data"]["root"] = args.output
    if args.input:
        config["data"]["input_paths"] = args.input

    output_root = Path(config["data"]["root"]).expanduser()
    if args.synthetic_smoke:
        manifest = build_synthetic_feature_cache(
            config,
            output_root,
            num_episodes=args.num_episodes,
            episode_length=args.episode_length,
            hidden_dim=args.hidden_dim,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            num_tokens=args.num_tokens,
        )
    else:
        manifest = build_planner_feature_cache(config, output_root=output_root)
    print(json.dumps({"root": str(output_root), "num_samples": manifest["num_samples"], "split_counts": manifest["split_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
