#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from himem_bridge_vla.path_utils import find_repo_root


REPO_ROOT = find_repo_root(__file__)
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from himem_bridge_vla.dataset.libero_progress_warmup import ImageStatsVLSummaryEncoder  # noqa: E402
from himem_bridge_vla.dataset.libero_progress_warmup import InternVL3VLSummaryEncoder  # noqa: E402
from himem_bridge_vla.dataset.libero_progress_warmup import action_normalizer_from_stats  # noqa: E402
from himem_bridge_vla.dataset.libero_progress_warmup import build_libero_progress_vl_embedding_cache  # noqa: E402
from himem_bridge_vla.dataset.libero_progress_warmup import load_action_segment_autoencoder  # noqa: E402
from himem_bridge_vla.dataset.libero_progress_warmup import resolve_storage_dtype  # noqa: E402
from himem_bridge_vla.dataset.memory_replay import read_memory_replay_jsonl  # noqa: E402
from himem_bridge_vla.path_utils import display_project_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LIBERO progress warm-up cache with pooled VL embeddings h_k.")
    parser.add_argument("--data-root", default=None, help="Defaults to <AUTODL_TMP>/libero/datasets.")
    parser.add_argument("--index", required=True, help="LIBERO replay JSONL index.")
    parser.add_argument("--output-root", required=True, help="Output directory for progress warm-up cache.")
    parser.add_argument("--encoder", choices=("internvl3", "image_stats"), default="internvl3")
    parser.add_argument("--model-name", default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segment-ae-checkpoint", default=None, help="Frozen H32 action intent AE checkpoint.")
    parser.add_argument("--norm-stats", default=None, help="Optional norm_stats.json used for action normalization.")
    parser.add_argument("--robot-key", default=None, help="Robot key inside norm_stats.json when needed.")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--replan-stride", type=int, default=16)
    parser.add_argument("--burnin-replan-steps", type=int, default=8)
    parser.add_argument("--loss-replan-steps", type=int, default=8)
    parser.add_argument("--require-full-burnin", action="store_true", help="Disable short burn-in windows.")
    parser.add_argument("--storage-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16"])
    parser.add_argument("--view-names", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Debug limit over replay-index rows.")
    parser.add_argument("--image-stats-hidden-dim", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--vl-batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Inspect index/window inputs without loading the VL model.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = resolve_libero_root(args.data_root)
    index_path = Path(args.index).expanduser()
    if args.dry_run:
        rows = read_memory_replay_jsonl(index_path)
        planned = _count_planned_replan_steps(
            rows[: int(args.max_steps)] if args.max_steps else rows,
            horizon=args.horizon,
            replan_stride=args.replan_stride,
        )
        payload = {
            "data_root": display_project_path(data_root, REPO_ROOT),
            "index": display_project_path(index_path, REPO_ROOT),
            "index_rows": len(rows),
            "planned_replan_steps": planned,
            "encoder": args.encoder,
            "output_root": display_project_path(args.output_root, REPO_ROOT),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    intent_encoder = None
    if args.segment_ae_checkpoint:
        intent_encoder = load_action_segment_autoencoder(args.segment_ae_checkpoint, device=args.device)
    normalizer = action_normalizer_from_stats(args.norm_stats, robot_key=args.robot_key)
    result = build_libero_progress_vl_embedding_cache(
        data_root=data_root,
        index_path=index_path,
        output_root=args.output_root,
        vl_encoder=build_encoder(args),
        action_horizon=args.horizon,
        replan_stride=args.replan_stride,
        burnin_replan_steps=args.burnin_replan_steps,
        loss_replan_steps=args.loss_replan_steps,
        allow_short_burnin=not bool(args.require_full_burnin),
        intent_encoder=intent_encoder,
        intent_encoder_checkpoint=args.segment_ae_checkpoint,
        action_normalizer=normalizer,
        norm_stats_path=args.norm_stats,
        robot_key=args.robot_key,
        storage_dtype=resolve_storage_dtype(args.storage_dtype),
        view_names=args.view_names,
        max_steps=args.max_steps,
        progress_interval=args.progress_interval,
        vl_batch_size=args.vl_batch_size,
    )
    summary = {
        "output_root": str(result.output_root),
        "manifest_path": str(result.manifest_path),
        "step_count": result.step_count,
        "window_count": result.window_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_encoder(args: argparse.Namespace):
    if args.encoder == "image_stats":
        return ImageStatsVLSummaryEncoder(hidden_dim=args.image_stats_hidden_dim)
    return InternVL3VLSummaryEncoder(
        model_name=args.model_name,
        image_size=args.image_size,
        device=args.device,
        storage_dtype=args.storage_dtype,
    )


def resolve_libero_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    data_root = Path(os.environ.get("AUTODL_TMP", str(Path.home() / "autodl-tmp"))).expanduser()
    return data_root / "libero" / "datasets"


def _count_planned_replan_steps(rows: list[dict], *, horizon: int, replan_stride: int) -> int:
    count = 0
    for row in rows:
        if str(row.get("benchmark", "LIBERO")).upper() != "LIBERO":
            continue
        if int(row["current_step"]) % int(replan_stride) != 0:
            continue
        if int(row["action_valid_count"]) < int(horizon):
            continue
        count += 1
    return count


if __name__ == "__main__":
    raise SystemExit(main())
