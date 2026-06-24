#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.dataset.memory_replay import read_memory_replay_jsonl  # noqa: E402
from himem_bridge_vla.dataset.memory_token_cache import ImageStatsVisualTokenEncoder  # noqa: E402
from himem_bridge_vla.dataset.memory_token_cache import InternVL3VisualTokenEncoder  # noqa: E402
from himem_bridge_vla.dataset.memory_token_cache import build_memory_replay_token_cache  # noqa: E402
from himem_bridge_vla.path_utils import display_project_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual-token shards from a memory replay JSONL index.")
    parser.add_argument("--benchmark", required=True, choices=("LIBERO", "RMBench", "rmbench", "libero"))
    parser.add_argument("--data-root", required=True, help="Root used by the replay index source_path values.")
    parser.add_argument("--index", required=True, help="Memory replay JSONL index path.")
    parser.add_argument("--output-root", required=True, help="Directory for manifest.json and shard .pt files.")
    parser.add_argument("--encoder", choices=("internvl3", "image_stats"), default="internvl3")
    parser.add_argument("--model-name", default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-shard", type=int, default=1024)
    parser.add_argument("--view-names", nargs="*", default=None)
    parser.add_argument("--image-stats-hidden-dim", type=int, default=16)
    parser.add_argument("--image-stats-tokens-per-view", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Inspect the index and planned encoder without writing shards.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.max_samples_per_shard <= 0:
        raise ValueError("--max-samples-per-shard must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")

    index_path = Path(args.index).expanduser()
    rows = read_memory_replay_jsonl(index_path)
    planned_samples = len(rows) if args.max_samples is None else min(len(rows), args.max_samples)
    if args.dry_run:
        payload = {
            "benchmark": args.benchmark.upper(),
            "data_root": str(Path(args.data_root).expanduser()),
            "index": display_project_path(index_path, REPO_ROOT),
            "index_rows": len(rows),
            "planned_samples": planned_samples,
            "encoder": args.encoder,
            "storage_dtype": args.storage_dtype,
            "max_samples_per_shard": args.max_samples_per_shard,
            "view_names": args.view_names,
            "output_root": display_project_path(args.output_root, REPO_ROOT),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    encoder = build_encoder(args)
    result = build_memory_replay_token_cache(
        benchmark=args.benchmark,
        data_root=args.data_root,
        index_path=index_path,
        output_root=args.output_root,
        encoder=encoder,
        view_names=args.view_names,
        max_samples=args.max_samples,
        max_samples_per_shard=args.max_samples_per_shard,
        storage_dtype=args.storage_dtype,
        manifest_extra={
            "model_name": args.model_name if args.encoder == "internvl3" else None,
            "image_size": args.image_size if args.encoder == "internvl3" else None,
        },
    )
    payload = {
        "format": "memory_replay_visual_token_cache",
        "manifest": display_project_path(result.manifest_path, REPO_ROOT),
        "output_root": display_project_path(result.output_root, REPO_ROOT),
        "sample_count": result.sample_count,
        "shard_count": len(result.shards),
        "shards": [display_project_path(shard.path, REPO_ROOT) for shard in result.shards],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_encoder(args: argparse.Namespace):
    if args.encoder == "image_stats":
        return ImageStatsVisualTokenEncoder(
            hidden_dim=args.image_stats_hidden_dim,
            tokens_per_view=args.image_stats_tokens_per_view,
        )
    return InternVL3VisualTokenEncoder(
        model_name=args.model_name,
        image_size=args.image_size,
        device=args.device,
        storage_dtype=args.storage_dtype,
    )


if __name__ == "__main__":
    raise SystemExit(main())
