from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from coarse_planner.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CoarsePlanner feature cache from SimulationDataset and InternVL3/VLA tokens."
    )
    parser.add_argument("--config", default="coarse_planner/configs/default.yaml")
    parser.add_argument("--dataset-config", default=None, help="Override simulation.dataset_config.")
    parser.add_argument("--output", default=None, help="Override data.root.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset paths and video availability.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override data.max_samples.")
    parser.add_argument("--max-samples-per-file", type=int, default=None, help="Override simulation.max_samples_per_file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.dataset_config:
        config["simulation"]["dataset_config"] = args.dataset_config
    if args.output:
        config["data"]["root"] = args.output
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.max_samples_per_file is not None:
        config["simulation"]["max_samples_per_file"] = args.max_samples_per_file

    dataset_config_path = Path(config["simulation"]["dataset_config"]).expanduser()
    if not dataset_config_path.is_absolute():
        dataset_config_path = Path.cwd() / dataset_config_path
    dataset_config = load_yaml(dataset_config_path)

    source_summary = validate_simulation_sources(dataset_config)
    if args.dry_run:
        print(json.dumps(source_summary, sort_keys=True))
        return 0
    if source_summary["missing_video_files"] > 0:
        raise FileNotFoundError(
            "simulation dataset is missing video files; run with --dry-run for details: "
            f"{source_summary['missing_video_files']} missing"
        )

    manifest = build_cache_from_simulation_dataset(config, dataset_config, device=args.device)
    print(
        json.dumps(
            {
                "root": str(Path(config["data"]["root"]).expanduser()),
                "num_samples": manifest["num_samples"],
                "split_counts": manifest["split_counts"],
                "feature_source": manifest["feature"]["source"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_cache_from_simulation_dataset(
    config: dict[str, Any], dataset_config: dict[str, Any], *, device: str
) -> dict[str, Any]:
    import torch

    from himem_bridge_vla.dataset.simulation_dataset import SimulationDataset
    from himem_bridge_vla.model.internvl3.internvl3_embedder import InternVL3Embedder

    root = Path(config["data"]["root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target_config = config["target"]
    feature_config = config["feature"]
    simulation_config = config["simulation"]

    action_segment_config = {
        "enabled": True,
        "num_plan_steps": int(target_config["num_plan_steps"]),
        "planning_horizon": int(target_config["planning_horizon"]),
        "action_dim": int(dataset_config["max_action_dim"]),
    }
    action_horizon = simulation_config.get("action_horizon") or int(target_config["planning_horizon"])
    dataset = SimulationDataset(
        dataset_config,
        image_size=int(feature_config.get("image_size", 448)),
        max_samples_per_file=simulation_config.get("max_samples_per_file"),
        video_backend=str(simulation_config.get("video_backend", "av")),
        video_backend_kwargs=simulation_config.get("video_backend_kwargs", {}),
        action_horizon=int(action_horizon),
        cache_dir=simulation_config.get("cache_dir"),
        action_segment_config=action_segment_config,
    )
    if len(dataset) == 0:
        raise ValueError("SimulationDataset produced no samples")

    embedder = InternVL3Embedder(
        model_name=str(feature_config.get("model_name", "OpenGVLab/InternVL3-1B")),
        image_size=int(feature_config.get("image_size", 448)),
        device=device,
        allow_image_token_truncation=bool(feature_config.get("allow_image_token_truncation", False)),
    )
    embedder.eval()

    max_samples = config["data"].get("max_samples")
    max_samples = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    shard_limit = int(config["data"].get("max_samples_per_shard", 4096))
    if shard_limit <= 0:
        raise ValueError("data.max_samples_per_shard must be positive")

    buffers = {
        str(config["data"].get("train_split", "train")): [],
        str(config["data"].get("eval_split", "eval")): [],
    }
    split_counts = {split: 0 for split in buffers}
    shards: list[dict[str, Any]] = []
    samples_written = 0

    for index in range(max_samples):
        sample = dataset[index]
        planner_images = sample.get("planner_images", sample["images"])
        planner_image_mask = sample.get("planner_image_mask", sample["image_mask"])
        planner_prompt = sample.get("planner_prompt", sample.get("prompt", ""))
        planner_state = sample.get("planner_state", sample["state"])
        with torch.no_grad():
            tokens = extract_planner_tokens(
                embedder,
                planner_images,
                planner_image_mask,
                planner_prompt,
                feature_config,
            )
        record = {
            "vlm_tokens": tokens.detach().cpu(),
            "state": planner_state.detach().cpu().float(),
            "action_segments": sample["action_segments"].detach().cpu().float(),
            "action_segment_mask": sample["action_segment_mask"].detach().cpu().float(),
            "episode_id": str(sample.get("episode_id", f"sample_{index}")),
            "frame_index": int(sample.get("frame_index", index)),
            "source_path": "SimulationDataset",
        }
        split = split_for_episode(
            record["episode_id"],
            seed=int(config.get("seed", 42)),
            val_fraction=float(config["data"].get("val_fraction", 0.1)),
            train_split=str(config["data"].get("train_split", "train")),
            eval_split=str(config["data"].get("eval_split", "eval")),
        )
        buffers[split].append(record)
        split_counts[split] += 1
        samples_written += 1
        if len(buffers[split]) >= shard_limit:
            shards.append(flush_planner_shard(root, split, len(shards), buffers[split], feature_config))
            buffers[split] = []

    for split, buffer in list(buffers.items()):
        if buffer:
            shards.append(flush_planner_shard(root, split, len(shards), buffer, feature_config))

    manifest = {
        "format": "planner_feature_cache",
        "version": 1,
        "supervision": "action_segment_latent",
        "num_samples": samples_written,
        "split_counts": split_counts,
        "target": target_config,
        "feature": feature_config,
        "simulation": simulation_config,
        "shards": shards,
    }
    manifest_path = root / str(config["data"].get("manifest", "manifest.json"))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def extract_planner_tokens(
    embedder: Any,
    images: Any,
    image_mask: Any,
    prompt: str,
    feature_config: dict[str, Any],
) -> Any:
    source = str(feature_config.get("source", "fused"))
    image_list = [image for image in images]
    if source == "fused":
        tokens = embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=image_list,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=False,
            return_hidden_states=False,
        )
    elif source == "hidden_state":
        output = embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=image_list,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=False,
            return_hidden_states=True,
            selected_layers=[feature_config.get("hidden_state_layer", "deep")],
        )
        tokens = output.hidden_states[0]
    else:
        raise ValueError("feature.source must be one of: fused, hidden_state")
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)
    if tokens.ndim != 2:
        raise ValueError(f"planner tokens must have shape [L, D], got {tuple(tokens.shape)}")
    return tokens


def flush_planner_shard(
    root: Path,
    split: str,
    shard_index: int,
    samples: list[dict[str, Any]],
    feature_config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"planner_samples_{shard_index:05d}.pt"
    dtype = storage_dtype(feature_config)
    payload = {
        "vlm_tokens": torch.stack([sample["vlm_tokens"].to(dtype=dtype) for sample in samples], dim=0),
        "state": torch.stack([sample["state"] for sample in samples], dim=0),
        "action_segments": torch.stack([sample["action_segments"] for sample in samples], dim=0),
        "action_segment_mask": torch.stack([sample["action_segment_mask"] for sample in samples], dim=0),
        "episode_id": [sample["episode_id"] for sample in samples],
        "frame_index": torch.tensor([sample["frame_index"] for sample in samples], dtype=torch.long),
        "source_path": [sample["source_path"] for sample in samples],
    }
    torch.save(payload, path)
    return {
        "path": str(path.relative_to(root)),
        "split": split,
        "num_samples": len(samples),
    }


def storage_dtype(feature_config: dict[str, Any]) -> Any:
    import torch

    dtype_name = str(feature_config.get("storage_dtype", "float16"))
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError("feature.storage_dtype must be one of: float16, bfloat16, float32")


def split_for_episode(
    episode_id: str,
    *,
    seed: int,
    val_fraction: float,
    train_split: str,
    eval_split: str,
) -> str:
    if val_fraction <= 0.0:
        return train_split
    if val_fraction >= 1.0:
        return eval_split
    digest = hashlib.sha1(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    return eval_split if value < val_fraction else train_split


def validate_simulation_sources(dataset_config: dict[str, Any]) -> dict[str, Any]:
    groups = []
    total_parquet = 0
    missing_video_files = 0
    for arm_name, arm_config in dataset_config.get("data_groups", {}).items():
        for dataset_name, raw_dataset in arm_config.items():
            dataset_root = Path(raw_dataset["path"]).expanduser()
            parquet_files = sorted(dataset_root.glob("data/*/*.parquet"))
            total_parquet += len(parquet_files)
            view_map = raw_dataset.get("view_map", {})
            missing_for_dataset = 0
            checked = 0
            for parquet_path in parquet_files[: min(8, len(parquet_files))]:
                base_video_path = dataset_root / "videos" / parquet_path.parent.name
                for aliases in view_map.values():
                    alias_values = aliases if isinstance(aliases, (list, tuple)) else (aliases,)
                    if not any((base_video_path / str(alias) / f"{parquet_path.stem}.mp4").exists() for alias in alias_values):
                        missing_for_dataset += 1
                checked += 1
            missing_video_files += missing_for_dataset
            groups.append(
                {
                    "arm": arm_name,
                    "dataset": dataset_name,
                    "path": str(dataset_root),
                    "parquet_files": len(parquet_files),
                    "checked_parquets": checked,
                    "missing_video_files_in_checked": missing_for_dataset,
                    "tasks_jsonl": (dataset_root / "meta" / "tasks.jsonl").exists(),
                    "stats_json": (dataset_root / "meta" / "stats.json").exists(),
                    "episodes_stats_jsonl": (dataset_root / "meta" / "episodes_stats.jsonl").exists(),
                }
            )
    return {
        "groups": groups,
        "total_parquet_files": total_parquet,
        "missing_video_files": missing_video_files,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
