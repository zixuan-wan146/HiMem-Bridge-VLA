#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from himem_bridge_vla.path_utils import find_repo_root


REPO_ROOT = find_repo_root(__file__)
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from himem_bridge_vla.dataset.memory_token_cache import MEMORY_TOKEN_CACHE_FORMAT  # noqa: E402
from himem_bridge_vla.dataset.memory_token_cache import MEMORY_TOKEN_CACHE_VERSION  # noqa: E402
from himem_bridge_vla.dataset.memory_token_cache import read_token_cache_manifest  # noqa: E402
from himem_bridge_vla.path_utils import display_project_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge independently built memory-token-cache part manifests.")
    parser.add_argument("--parts-root", required=True, help="Directory containing partXX/manifest.json directories.")
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Output manifest path. Defaults to <parts-root>/manifest.json.",
    )
    parser.add_argument("--part-glob", default="part*/manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    parts_root = Path(args.parts_root).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser() if args.output_manifest else parts_root / "manifest.json"
    manifest_paths = sorted(parts_root.glob(args.part_glob))
    if not manifest_paths:
        raise FileNotFoundError(f"no part manifests matched {args.part_glob!r} under {parts_root}")

    manifests = [read_token_cache_manifest(path) for path in manifest_paths]
    merged = _merge_manifests(parts_root=parts_root, manifest_paths=manifest_paths, manifests=manifests)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
        handle.write("\n")

    payload = {
        "format": MEMORY_TOKEN_CACHE_FORMAT,
        "parts": len(manifest_paths),
        "sample_count": merged["sample_count"],
        "shard_count": len(merged["shards"]),
        "manifest": display_project_path(output_manifest, REPO_ROOT),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _merge_manifests(
    *,
    parts_root: Path,
    manifest_paths: list[Path],
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    first = dict(manifests[0])
    _validate_part_manifest(first, manifest_paths[0])
    for path, manifest in zip(manifest_paths[1:], manifests[1:]):
        _validate_part_manifest(manifest, path)
        for key in (
            "format",
            "version",
            "benchmark",
            "encoder",
            "hidden_state_encoder",
            "hidden_state_layers",
            "hidden_dim",
            "tokens_per_view",
            "storage_dtype",
            "view_names",
        ):
            if manifest.get(key) != first.get(key):
                raise ValueError(f"part manifest {path} has mismatched {key}: {manifest.get(key)!r} != {first.get(key)!r}")

    shards = []
    sample_count = 0
    for path, manifest in zip(manifest_paths, manifests):
        part_root = path.parent
        for raw_shard in manifest["shards"]:
            shard_count = int(raw_shard["sample_count"])
            shard_path = (part_root / str(raw_shard["path"])).resolve()
            shards.append(
                {
                    "path": str(shard_path.relative_to(parts_root)),
                    "sample_count": shard_count,
                    "start_index": sample_count,
                    "end_index": sample_count + shard_count,
                }
            )
            sample_count += shard_count

    first["output_root"] = str(parts_root)
    first["sample_count"] = sample_count
    first["max_samples"] = None
    first["shards"] = shards
    first["merged_from_parts"] = [str(path.parent.relative_to(parts_root)) for path in manifest_paths]
    first["normalization"] = _merge_normalization(manifests)
    if first["normalization"] is not None:
        first["action_normalization"] = {
            "enabled": True,
            "type": "train_split_minmax_to_minus_one_one",
            "clip_after_normalization": True,
            "clip_range": [-1.0, 1.0],
            "statistics_from": "merged_cache_parts",
        }
    return first


def _validate_part_manifest(manifest: dict[str, Any], path: Path) -> None:
    if manifest.get("format") != MEMORY_TOKEN_CACHE_FORMAT:
        raise ValueError(f"{path} has invalid format {manifest.get('format')!r}")
    if int(manifest.get("version", -1)) != MEMORY_TOKEN_CACHE_VERSION:
        raise ValueError(f"{path} has unsupported version {manifest.get('version')!r}")
    if not manifest.get("shards"):
        raise ValueError(f"{path} has no shards")


def _merge_normalization(manifests: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalizations = [manifest.get("normalization") for manifest in manifests]
    if all(normalization is None for normalization in normalizations):
        return None
    if any(not isinstance(normalization, dict) for normalization in normalizations):
        raise ValueError("either every part manifest must contain normalization or none should")
    first = dict(normalizations[0])
    if first.get("type") != "train_split_minmax_to_minus_one_one":
        raise ValueError(f"unsupported normalization type: {first.get('type')!r}")
    robot_key = str(first.get("robot_key") or "")
    if not robot_key:
        raise ValueError("normalization lacks robot_key")

    state_min = None
    state_max = None
    action_min = None
    action_max = None
    for normalization in normalizations:
        if normalization.get("type") != first.get("type"):
            raise ValueError("part normalizations use different types")
        if str(normalization.get("robot_key")) != robot_key:
            raise ValueError("part normalizations use different robot keys")
        stats = normalization["stats"][robot_key]
        state_min = _elementwise_min(state_min, stats["observation.state"]["min"])
        state_max = _elementwise_max(state_max, stats["observation.state"]["max"])
        action_min = _elementwise_min(action_min, stats["action"]["min"])
        action_max = _elementwise_max(action_max, stats["action"]["max"])

    first["statistics_from"] = "merged_cache_parts"
    first["stats"] = {
        robot_key: {
            "observation.state": {"min": state_min, "max": state_max},
            "action": {"min": action_min, "max": action_max},
        }
    }
    return first


def _elementwise_min(current: list[float] | None, values: list[float]) -> list[float]:
    values = [float(value) for value in values]
    if current is None:
        return values
    if len(current) != len(values):
        raise ValueError("normalization stat dimensions do not match")
    return [min(a, b) for a, b in zip(current, values)]


def _elementwise_max(current: list[float] | None, values: list[float]) -> list[float]:
    values = [float(value) for value in values]
    if current is None:
        return values
    if len(current) != len(values):
        raise ValueError("normalization stat dimensions do not match")
    return [max(a, b) for a, b in zip(current, values)]


if __name__ == "__main__":
    raise SystemExit(main())
