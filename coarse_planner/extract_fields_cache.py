from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch


_METADATA_FIELDS = ("frame_index", "episode_id", "source_path", "task_suite", "task_description")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selected fields from a planner feature cache.")
    parser.add_argument("--input-root", required=True, help="Source planner feature cache root.")
    parser.add_argument("--output-root", required=True, help="Destination cache root.")
    parser.add_argument("--manifest", default="manifest.json", help="Manifest filename under the cache root.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["action_segments", "action_segment_mask"],
        help="Shard tensor/list fields to retain.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    manifest_path = input_root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    retained_fields = tuple(str(field) for field in args.fields)
    keep_fields = set(retained_fields) | set(_METADATA_FIELDS)
    output_shards = []
    for raw_shard in manifest.get("shards", []):
        relative_path = Path(raw_shard["path"])
        source_path = input_root / relative_path
        dest_path = output_root / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        shard = torch.load(source_path, map_location="cpu", weights_only=False)
        missing = [field for field in retained_fields if field not in shard]
        if missing:
            raise KeyError(f"{source_path} is missing requested fields: {missing}")
        slim_shard = {key: value for key, value in shard.items() if key in keep_fields}
        torch.save(slim_shard, dest_path)
        output_shards.append(dict(raw_shard))

    output_manifest = dict(manifest)
    output_manifest["feature_fields"] = list(retained_fields)
    output_manifest["source_cache_root"] = str(input_root)
    output_manifest["shards"] = output_shards
    (output_root / args.manifest).write_text(json.dumps(output_manifest, indent=2, sort_keys=True))
    print(json.dumps({"output_root": str(output_root), "fields": list(retained_fields), "shards": len(output_shards)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
