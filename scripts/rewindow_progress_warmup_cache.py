#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.dataset.libero_progress_warmup import LIBERO_PROGRESS_WARMUP_FORMAT  # noqa: E402
from himem_bridge_vla.dataset.libero_progress_warmup import LIBERO_PROGRESS_WARMUP_VERSION  # noqa: E402
from himem_bridge_vla.dataset.libero_progress_warmup import build_libero_progress_windows  # noqa: E402
from himem_bridge_vla.dataset.libero_progress_warmup import resolve_libero_progress_manifest_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild progress warm-up windows without recomputing cached VL summaries.")
    parser.add_argument("--source-cache", required=True, help="Source cache directory or manifest.json path.")
    parser.add_argument("--output-root", required=True, help="Output cache directory.")
    parser.add_argument("--burnin-replan-steps", type=int, default=None)
    parser.add_argument("--loss-replan-steps", type=int, required=True)
    parser.add_argument("--require-full-burnin", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_manifest_path = resolve_libero_progress_manifest_path(args.source_cache)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("format") != LIBERO_PROGRESS_WARMUP_FORMAT:
        raise ValueError(f"invalid progress warm-up cache: {source_manifest_path}")
    if int(source_manifest.get("version", -1)) != LIBERO_PROGRESS_WARMUP_VERSION:
        raise ValueError(f"unsupported cache version: {source_manifest.get('version')!r}")

    payload = torch.load(source_manifest_path.parent / source_manifest["data_path"], map_location="cpu", weights_only=False)
    if payload.get("format") != LIBERO_PROGRESS_WARMUP_FORMAT:
        raise ValueError(f"invalid cache payload under {source_manifest_path.parent}")
    steps = list(payload["steps"])
    burnin_steps = int(
        source_manifest.get("burnin_replan_steps", 8)
        if args.burnin_replan_steps is None
        else args.burnin_replan_steps
    )
    allow_short_burnin = not bool(args.require_full_burnin)
    windows = build_libero_progress_windows(
        steps,
        burnin_replan_steps=burnin_steps,
        loss_replan_steps=int(args.loss_replan_steps),
        allow_short_burnin=allow_short_burnin,
    )

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = output_root / "data.pt"
    torch.save(
        {
            "format": LIBERO_PROGRESS_WARMUP_FORMAT,
            "version": LIBERO_PROGRESS_WARMUP_VERSION,
            "steps": steps,
            "windows": windows,
        },
        data_path,
    )
    manifest = dict(source_manifest)
    manifest.update(
        {
            "data_path": data_path.name,
            "source_cache": str(source_manifest_path.parent),
            "burnin_replan_steps": burnin_steps,
            "loss_replan_steps": int(args.loss_replan_steps),
            "allow_short_burnin": allow_short_burnin,
            "step_count": len(steps),
            "window_count": len(windows),
            "suite_window_counts": _window_suite_counts(windows),
        }
    )
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "step_count": len(steps), "window_count": len(windows)}, indent=2))
    return 0


def _window_suite_counts(windows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for window in windows:
        suite = str(window["suite"])
        counts[suite] = counts.get(suite, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
