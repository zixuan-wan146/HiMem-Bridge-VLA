from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from coarse_planner.config import load_config
from coarse_planner.build_from_simulation import extract_planner_tokens, storage_dtype
from himem_bridge_vla.dataset.coarse_actions import build_coarse_action_target
from himem_bridge_vla.utils.normalization import minmax_normalize


DEFAULT_LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class LiberoEpisode:
    suite: str
    task_id: int
    hdf5_path: str
    demo_key: str
    episode_id: str
    task_description: str
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LIBERO CoarsePlanner feature caches for one or more horizon ablations."
    )
    parser.add_argument("--config", default="coarse_planner/configs/libero_horizon_ablation_build.yaml")
    parser.add_argument("--output", default=None, help="Override base output root. Multi-horizon builds append _h{H}.")
    parser.add_argument("--sample-index", default=None, help="Override libero.sample_index_path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Override data.max_samples.")
    parser.add_argument("--samples-per-episode", type=int, default=None, help="Override libero.samples_per_episode.")
    parser.add_argument("--regenerate-sample-index", action="store_true")
    parser.add_argument("--horizons", nargs="*", type=int, default=None, help="Override libero.horizons.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.max_samples is not None:
        config["data"]["max_samples"] = int(args.max_samples)
    if args.samples_per_episode is not None:
        config.setdefault("libero", {})["samples_per_episode"] = int(args.samples_per_episode)
    if args.sample_index is not None:
        config.setdefault("libero", {})["sample_index_path"] = args.sample_index
    if args.horizons:
        config.setdefault("libero", {})["horizons"] = [int(value) for value in args.horizons]

    summary = dry_run_summary(config, output_override=args.output)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    manifests = build_libero_horizon_caches(
        config,
        device=args.device,
        output_override=args.output,
        regenerate_sample_index=bool(args.regenerate_sample_index),
    )
    print(
        json.dumps(
            {
                "horizons": sorted(int(key) for key in manifests),
                "num_samples": {str(key): value["num_samples"] for key, value in manifests.items()},
                "roots": {str(key): str(_root_for_horizon(config, key, output_override=args.output)) for key in manifests},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_libero_horizon_caches(
    config: dict[str, Any],
    *,
    device: str,
    output_override: str | None = None,
    regenerate_sample_index: bool = False,
) -> dict[int, dict[str, Any]]:
    import h5py
    import torch

    from himem_bridge_vla.model.internvl3.internvl3_embedder import InternVL3Embedder

    libero_config = _libero_config(config)
    feature_config = config["feature"]
    horizons = _horizons(config)
    chunk_size = int(libero_config.get("chunk_size", 8))
    _validate_horizons(horizons, chunk_size)

    sample_index_path = _sample_index_path(libero_config)
    entries = _load_or_create_sample_index(
        config,
        sample_index_path,
        max_horizon=max(horizons),
        regenerate=regenerate_sample_index,
    )
    if not entries:
        raise ValueError("LIBERO sample index has no entries")

    roots = {horizon: _root_for_horizon(config, horizon, output_override=output_override) for horizon in horizons}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)

    embedder = InternVL3Embedder(
        model_name=str(feature_config.get("model_name", "OpenGVLab/InternVL3-1B")),
        image_size=int(feature_config.get("image_size", 448)),
        device=device,
        allow_image_token_truncation=bool(feature_config.get("allow_image_token_truncation", False)),
    )
    embedder.eval()

    train_split = str(config["data"].get("train_split", "train"))
    eval_split = str(config["data"].get("eval_split", "eval"))
    shard_limit = int(config["data"].get("max_samples_per_shard", 512))
    if shard_limit <= 0:
        raise ValueError("data.max_samples_per_shard must be positive")

    buffers = {horizon: {train_split: [], eval_split: []} for horizon in horizons}
    shards = {horizon: [] for horizon in horizons}
    split_counts = {horizon: {train_split: 0, eval_split: 0} for horizon in horizons}
    num_samples = {horizon: 0 for horizon in horizons}
    normalizer = LiberoNormalizer(libero_config)

    grouped = _group_entries_by_hdf5(entries)
    for hdf5_path, path_entries in grouped.items():
        with h5py.File(hdf5_path, "r") as handle:
            for entry in path_entries:
                demo = handle[f"data/{entry['demo_key']}"]
                frame_index = int(entry["frame_index"])
                images = _load_images(demo, frame_index, libero_config)
                state = normalizer.state(demo, frame_index)
                actions = normalizer.actions(demo["actions"][:])
                image_mask = torch.ones(len(images), dtype=torch.bool)
                with torch.no_grad():
                    tokens = extract_planner_tokens(
                        embedder,
                        images,
                        image_mask,
                        str(entry["task_description"]),
                        feature_config,
                    )

                split = _split_for_episode(
                    str(entry["episode_id"]),
                    seed=int(config.get("seed", 42)),
                    val_fraction=float(config["data"].get("val_fraction", 0.1)),
                    train_split=train_split,
                    eval_split=eval_split,
                )
                for horizon in horizons:
                    target = _coarse_target_for_horizon(
                        actions,
                        frame_index,
                        horizon=horizon,
                        chunk_size=chunk_size,
                        config=config,
                    )
                    record = {
                        "vlm_tokens": tokens.detach().cpu(),
                        "state": torch.from_numpy(state).float(),
                        "coarse_actions": torch.from_numpy(target["coarse_actions"]).float(),
                        "coarse_action_mask": torch.from_numpy(target["coarse_action_mask"]).float(),
                        "episode_id": str(entry["episode_id"]),
                        "frame_index": int(frame_index),
                        "source_path": str(hdf5_path),
                        "task_suite": str(entry["suite"]),
                        "task_description": str(entry["task_description"]),
                    }
                    buffers[horizon][split].append(record)
                    split_counts[horizon][split] += 1
                    num_samples[horizon] += 1
                    if len(buffers[horizon][split]) >= shard_limit:
                        shards[horizon].append(
                            _flush_libero_shard(
                                roots[horizon],
                                split,
                                len(shards[horizon]),
                                buffers[horizon][split],
                                feature_config,
                            )
                        )
                        buffers[horizon][split] = []

    for horizon in horizons:
        for split, buffer in list(buffers[horizon].items()):
            if buffer:
                shards[horizon].append(
                    _flush_libero_shard(roots[horizon], split, len(shards[horizon]), buffer, feature_config)
                )

    manifests: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        manifest = {
            "format": "planner_feature_cache",
            "version": 1,
            "num_samples": num_samples[horizon],
            "split_counts": split_counts[horizon],
            "target": _target_config_for_horizon(config, horizon, chunk_size),
            "feature": feature_config,
            "libero": _manifest_libero_config(libero_config, sample_index_path),
            "sample_index": str(sample_index_path),
            "shards": shards[horizon],
        }
        manifest_path = roots[horizon] / str(config["data"].get("manifest", "manifest.json"))
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        manifests[horizon] = manifest
    return manifests


class LiberoNormalizer:
    def __init__(self, libero_config: dict[str, Any]) -> None:
        stats_path = Path(str(libero_config["norm_stats_path"])).expanduser()
        stats = json.loads(stats_path.read_text())
        robot_key = str(libero_config.get("robot_key") or next(iter(stats)))
        robot_stats = stats[robot_key]
        self.state_min = np.asarray(robot_stats["observation.state"]["min"], dtype=np.float32)
        self.state_max = np.asarray(robot_stats["observation.state"]["max"], dtype=np.float32)
        self.action_min = np.asarray(robot_stats["action"]["min"], dtype=np.float32)
        self.action_max = np.asarray(robot_stats["action"]["max"], dtype=np.float32)
        self.state_dim = int(libero_config.get("state_dim", 24))
        self.gripper_to_binary = bool(libero_config.get("gripper_to_binary", True))
        self.normalize_actions = bool(libero_config.get("normalize_actions", False))

    def state(self, demo: Any, frame_index: int) -> np.ndarray:
        raw = np.concatenate(
            [
                np.asarray(demo["obs/ee_states"][frame_index], dtype=np.float32),
                np.asarray(demo["obs/gripper_states"][frame_index], dtype=np.float32),
            ],
            axis=0,
        )
        normalized = _minmax_np(raw, self.state_min, self.state_max)
        return _pad_np(normalized, self.state_dim)

    def actions(self, actions: Any) -> np.ndarray:
        action_array = np.asarray(actions, dtype=np.float32).copy()
        if self.gripper_to_binary and action_array.shape[1] >= 7:
            action_array[:, 6] = (action_array[:, 6] < 0.0).astype(np.float32)
        if self.normalize_actions:
            action_array = _minmax_np(action_array, self.action_min, self.action_max)
        return action_array


def dry_run_summary(config: dict[str, Any], *, output_override: str | None = None) -> dict[str, Any]:
    libero_config = _libero_config(config)
    horizons = _horizons(config)
    chunk_size = int(libero_config.get("chunk_size", 8))
    _validate_horizons(horizons, chunk_size)
    episodes = discover_libero_episodes(libero_config)
    lengths = np.asarray([episode.length for episode in episodes], dtype=np.int64)
    entries = generate_sample_index_entries(config, episodes, max_horizon=max(horizons))
    token_shape = libero_config.get("estimated_token_shape", [1024, 896])
    bytes_per_token = 2 if str(config["feature"].get("storage_dtype", "float16")) in {"float16", "bfloat16"} else 4
    bytes_per_sample = int(token_shape[0]) * int(token_shape[1]) * bytes_per_token
    return {
        "dataset_root": str(Path(str(libero_config["dataset_root"])).expanduser()),
        "suites": list(_suites(libero_config)),
        "horizons": horizons,
        "chunk_size": chunk_size,
        "num_plan_steps": {str(horizon): horizon // chunk_size for horizon in horizons},
        "episodes": len(episodes),
        "candidate_samples": len(entries),
        "length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "length_min": int(lengths.min()) if len(lengths) else 0,
        "length_max": int(lengths.max()) if len(lengths) else 0,
        "roots": {str(horizon): str(_root_for_horizon(config, horizon, output_override=output_override)) for horizon in horizons},
        "estimated_feature_cache_gb_per_horizon": round(len(entries) * bytes_per_sample / (1024**3), 3),
    }


def discover_libero_episodes(libero_config: dict[str, Any]) -> list[LiberoEpisode]:
    import h5py

    dataset_root = Path(str(libero_config["dataset_root"])).expanduser()
    episodes: list[LiberoEpisode] = []
    for suite in _suites(libero_config):
        suite_dir = dataset_root / suite
        if not suite_dir.exists():
            raise FileNotFoundError(f"LIBERO suite directory not found: {suite_dir}")
        for task_id, hdf5_path in enumerate(sorted(suite_dir.glob("*.hdf5"))):
            task_description = _task_description_from_path(hdf5_path)
            with h5py.File(hdf5_path, "r") as handle:
                for demo_key in sorted(handle["data"].keys(), key=_demo_sort_key):
                    length = int(handle[f"data/{demo_key}/actions"].shape[0])
                    episode_id = f"{suite}:{hdf5_path.stem}:{demo_key}"
                    episodes.append(
                        LiberoEpisode(
                            suite=suite,
                            task_id=task_id,
                            hdf5_path=str(hdf5_path),
                            demo_key=str(demo_key),
                            episode_id=episode_id,
                            task_description=task_description,
                            length=length,
                        )
                    )
    return episodes


def generate_sample_index_entries(
    config: dict[str, Any],
    episodes: list[LiberoEpisode],
    *,
    max_horizon: int,
) -> list[dict[str, Any]]:
    data_config = config["data"]
    libero_config = _libero_config(config)
    rng = random.Random(int(config.get("seed", 42)))
    samples_per_episode = int(libero_config.get("samples_per_episode", 4))
    if samples_per_episode <= 0:
        raise ValueError("libero.samples_per_episode must be positive")
    require_full = bool(libero_config.get("require_full_max_horizon", True))
    entries: list[dict[str, Any]] = []
    for episode in episodes:
        max_start = episode.length - max_horizon if require_full else episode.length - 1
        if max_start < 0:
            continue
        candidates = list(range(0, max_start + 1))
        if len(candidates) <= samples_per_episode:
            starts = candidates
        else:
            starts = sorted(rng.sample(candidates, samples_per_episode))
        for start in starts:
            entries.append(
                {
                    "suite": episode.suite,
                    "task_id": episode.task_id,
                    "hdf5_path": episode.hdf5_path,
                    "demo_key": episode.demo_key,
                    "episode_id": episode.episode_id,
                    "task_description": episode.task_description,
                    "frame_index": int(start),
                    "episode_length": episode.length,
                }
            )
    rng.shuffle(entries)
    max_samples = data_config.get("max_samples")
    if max_samples is not None:
        entries = entries[: max(0, int(max_samples))]
    entries.sort(key=lambda item: (item["hdf5_path"], item["demo_key"], item["frame_index"]))
    return entries


def _load_or_create_sample_index(
    config: dict[str, Any],
    sample_index_path: Path,
    *,
    max_horizon: int,
    regenerate: bool,
) -> list[dict[str, Any]]:
    if sample_index_path.exists() and not regenerate:
        payload = json.loads(sample_index_path.read_text())
        return list(payload["entries"])

    episodes = discover_libero_episodes(_libero_config(config))
    entries = generate_sample_index_entries(config, episodes, max_horizon=max_horizon)
    sample_index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "seed": int(config.get("seed", 42)),
        "max_horizon": int(max_horizon),
        "samples_per_episode": int(_libero_config(config).get("samples_per_episode", 4)),
        "max_samples": config["data"].get("max_samples"),
        "entry_count": len(entries),
        "entries": entries,
    }
    sample_index_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return entries


def _coarse_target_for_horizon(
    actions: np.ndarray,
    frame_index: int,
    *,
    horizon: int,
    chunk_size: int,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    valid_count = max(0, min(horizon, int(actions.shape[0]) - int(frame_index)))
    future = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    if valid_count > 0:
        future[:valid_count] = actions[frame_index : frame_index + valid_count]
    coarse_actions, coarse_mask = build_coarse_action_target(
        future,
        num_plan_steps=horizon // chunk_size,
        planning_horizon=horizon,
        valid_action_count=valid_count,
        action_convention=str(config["target"].get("action_convention", "relative")),
        motion_indices=config["target"].get("motion_indices"),
        gripper_indices=config["target"].get("gripper_indices"),
    )
    return {"coarse_actions": coarse_actions, "coarse_action_mask": coarse_mask.astype(np.float32)}


def _flush_libero_shard(
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
        "coarse_actions": torch.stack([sample["coarse_actions"] for sample in samples], dim=0),
        "coarse_action_mask": torch.stack([sample["coarse_action_mask"] for sample in samples], dim=0),
        "episode_id": [sample["episode_id"] for sample in samples],
        "frame_index": torch.tensor([sample["frame_index"] for sample in samples], dtype=torch.long),
        "source_path": [sample["source_path"] for sample in samples],
        "task_suite": [sample["task_suite"] for sample in samples],
        "task_description": [sample["task_description"] for sample in samples],
    }
    torch.save(payload, path)
    return {"path": str(path.relative_to(root)), "split": split, "num_samples": len(samples)}


def _load_images(demo: Any, frame_index: int, libero_config: dict[str, Any]) -> list[Image.Image]:
    views = list(libero_config.get("views", ["agentview_rgb", "eye_in_hand_rgb"]))
    images = []
    for view in views:
        key = f"obs/{view}"
        if key not in demo:
            raise KeyError(f"LIBERO demo is missing image view: {key}")
        images.append(Image.fromarray(np.asarray(demo[key][frame_index], dtype=np.uint8)))
    return images


def _group_entries_by_hdf5(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["hdf5_path"]), []).append(entry)
    for path_entries in grouped.values():
        path_entries.sort(key=lambda item: (str(item["demo_key"]), int(item["frame_index"])))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _target_config_for_horizon(config: dict[str, Any], horizon: int, chunk_size: int) -> dict[str, Any]:
    target = dict(config["target"])
    target["planning_horizon"] = int(horizon)
    target["num_plan_steps"] = int(horizon) // int(chunk_size)
    target["chunk_size"] = int(chunk_size)
    return target


def _root_for_horizon(config: dict[str, Any], horizon: int, *, output_override: str | None = None) -> Path:
    horizons = _horizons(config)
    chunk_size = int(_libero_config(config).get("chunk_size", 8))
    if output_override:
        base = Path(output_override).expanduser()
        return base if len(horizons) == 1 else Path(f"{base}_h{horizon}")
    template = _libero_config(config).get("output_root_template")
    if template:
        return Path(str(template).format(horizon=horizon, num_plan_steps=horizon // chunk_size)).expanduser()
    base = Path(str(config["data"]["root"])).expanduser()
    return base if len(horizons) == 1 else Path(f"{base}_h{horizon}")


def _sample_index_path(libero_config: dict[str, Any]) -> Path:
    value = libero_config.get("sample_index_path")
    if value is None:
        raise ValueError("libero.sample_index_path is required")
    return Path(str(value)).expanduser()


def _libero_config(config: dict[str, Any]) -> dict[str, Any]:
    libero_config = dict(config.get("libero", {}))
    if "dataset_root" not in libero_config:
        raise ValueError("libero.dataset_root is required")
    if "norm_stats_path" not in libero_config:
        raise ValueError("libero.norm_stats_path is required")
    return libero_config


def _manifest_libero_config(libero_config: dict[str, Any], sample_index_path: Path) -> dict[str, Any]:
    manifest_config = dict(libero_config)
    manifest_config["sample_index_path"] = str(sample_index_path)
    return manifest_config


def _horizons(config: dict[str, Any]) -> list[int]:
    raw = config.get("libero", {}).get("horizons")
    if raw is None:
        raw = [int(config["target"]["planning_horizon"])]
    horizons = sorted({int(value) for value in raw})
    if not horizons:
        raise ValueError("at least one planning horizon is required")
    return horizons


def _validate_horizons(horizons: list[int], chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError("libero.chunk_size must be positive")
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"planning horizon must be positive, got {horizon}")
        if horizon % chunk_size != 0:
            raise ValueError(f"planning horizon {horizon} must be divisible by chunk_size {chunk_size}")


def _suites(libero_config: dict[str, Any]) -> tuple[str, ...]:
    suites = libero_config.get("suites", DEFAULT_LIBERO_SUITES)
    if isinstance(suites, str):
        suites = [item.strip() for item in suites.split(",") if item.strip()]
    return tuple(str(suite) for suite in suites)


def _task_description_from_path(path: Path) -> str:
    name = path.name
    if name.endswith("_demo.hdf5"):
        name = name[: -len("_demo.hdf5")]
    return name.replace("_", " ")


def _demo_sort_key(name: str) -> tuple[int, str]:
    suffix = str(name).split("_")[-1]
    return (int(suffix), str(name)) if suffix.isdigit() else (10**9, str(name))


def _split_for_episode(
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


def _minmax_np(value: np.ndarray, min_value: np.ndarray, max_value: np.ndarray) -> np.ndarray:
    import torch

    normalized = minmax_normalize(
        torch.as_tensor(value, dtype=torch.float32),
        torch.as_tensor(min_value, dtype=torch.float32),
        torch.as_tensor(max_value, dtype=torch.float32),
    )
    return normalized.detach().cpu().numpy().astype(np.float32)


def _pad_np(value: np.ndarray, target_dim: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.shape[0] > target_dim:
        raise ValueError(f"value length {value.shape[0]} exceeds target_dim {target_dim}")
    if value.shape[0] == target_dim:
        return value
    return np.pad(value, (0, target_dim - value.shape[0]), mode="constant").astype(np.float32)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
