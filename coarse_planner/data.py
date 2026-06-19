from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from himem_bridge_vla.dataset.action_segments import build_action_segment_target


@dataclass(frozen=True)
class PlannerShard:
    path: Path
    split: str
    num_samples: int


class PlannerFeatureDataset(Dataset):
    """Dataset over precomputed VLM/state features and coarse action targets."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        manifest: str | Path = "manifest.json",
        shard_cache_size: int = 8,
    ) -> None:
        self.root = Path(root).expanduser()
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"planner feature manifest not found: {manifest_path}")
        with manifest_path.open("r") as f:
            self.manifest = json.load(f)

        self.shards: list[PlannerShard] = []
        self.index: list[tuple[int, int]] = []
        for raw_shard in self.manifest.get("shards", []):
            if str(raw_shard.get("split", "train")) != split:
                continue
            shard_path = Path(raw_shard["path"])
            if not shard_path.is_absolute():
                shard_path = self.root / shard_path
            shard = PlannerShard(
                path=shard_path,
                split=split,
                num_samples=int(raw_shard["num_samples"]),
            )
            shard_index = len(self.shards)
            self.shards.append(shard)
            self.index.extend((shard_index, row) for row in range(shard.num_samples))
        if not self.index:
            raise ValueError(f"no planner samples found for split={split!r} in {manifest_path}")

        self._max_cached_shards = max(1, int(shard_cache_size))
        self._cached_shards: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, row = self.index[index]
        shard = self._load_shard(shard_index)
        item = {
            "vlm_tokens": shard["vlm_tokens"][row].float(),
            "state": shard["state"][row].float(),
            "frame_index": int(shard["frame_index"][row]),
            "episode_id": str(shard["episode_id"][row]),
        }
        item["action_segments"] = shard["action_segments"][row].float()
        item["action_segment_mask"] = shard["action_segment_mask"][row].float()
        if "source_path" in shard:
            item["source_path"] = str(shard["source_path"][row])
        if "task_suite" in shard:
            item["task_suite"] = str(shard["task_suite"][row])
        if "task_description" in shard:
            item["task_description"] = str(shard["task_description"][row])
        return item

    def _load_shard(self, shard_index: int) -> dict[str, Any]:
        if shard_index in self._cached_shards:
            shard = self._cached_shards.pop(shard_index)
            self._cached_shards[shard_index] = shard
            return shard
        shard = torch.load(self.shards[shard_index].path, map_location="cpu", weights_only=False)
        self._cached_shards[shard_index] = shard
        while len(self._cached_shards) > self._max_cached_shards:
            self._cached_shards.popitem(last=False)
        return shard

    @property
    def sample_shapes(self) -> dict[str, tuple[int, ...]]:
        sample = self[0]
        return {
            "vlm_tokens": tuple(sample["vlm_tokens"].shape),
            "state": tuple(sample["state"].shape),
            "action_segments": tuple(sample["action_segments"].shape),
            "action_segment_mask": tuple(sample["action_segment_mask"].shape),
        }


def build_datasets(config: dict[str, Any]) -> tuple[PlannerFeatureDataset, PlannerFeatureDataset]:
    data_config = config["data"]
    if str(data_config.get("format", "planner_feature_cache")) != "planner_feature_cache":
        raise ValueError(f"unsupported coarse_planner data.format: {data_config.get('format')!r}")
    root = data_config["root"]
    manifest = data_config.get("manifest", "manifest.json")
    shard_cache_size = int(data_config.get("shard_cache_size", 8))
    train_dataset = PlannerFeatureDataset(
        root,
        split=str(data_config.get("train_split", "train")),
        manifest=manifest,
        shard_cache_size=shard_cache_size,
    )
    eval_dataset = PlannerFeatureDataset(
        root,
        split=str(data_config.get("eval_split", "eval")),
        manifest=manifest,
        shard_cache_size=shard_cache_size,
    )
    return train_dataset, eval_dataset


def build_planner_feature_cache(config: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    data_config = config["data"]
    target_config = config["target"]
    root = Path(output_root or data_config["root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path).expanduser() for path in data_config.get("input_paths", [])]
    if not input_paths:
        raise ValueError("data.input_paths must contain at least one feature source")

    episodes = []
    for input_path in input_paths:
        episodes.extend(_load_episodes(input_path, data_config))
    if not episodes:
        raise ValueError("no planner episodes were loaded")

    train_ids, eval_ids = _split_episode_ids(
        [episode["episode_id"] for episode in episodes],
        seed=int(config.get("seed", 42)),
        val_fraction=float(data_config.get("val_fraction", 0.1)),
    )
    split_by_episode = str(data_config.get("split_by", "episode")) == "episode"
    max_samples = data_config.get("max_samples")
    max_samples = None if max_samples is None else int(max_samples)
    shard_limit = int(data_config.get("max_samples_per_shard", 4096))
    if shard_limit <= 0:
        raise ValueError("data.max_samples_per_shard must be positive")

    buffers = {
        str(data_config.get("train_split", "train")): [],
        str(data_config.get("eval_split", "eval")): [],
    }
    shards: list[dict[str, Any]] = []
    sample_count = 0
    split_counts = {split: 0 for split in buffers}

    for episode in episodes:
        episode_split = str(data_config.get("eval_split", "eval")) if episode["episode_id"] in eval_ids else str(
            data_config.get("train_split", "train")
        )
        for sample in _iter_episode_samples(episode, data_config, target_config):
            if max_samples is not None and sample_count >= max_samples:
                break
            split = episode_split if split_by_episode else _sample_split(sample_count, train_ids, config, data_config)
            buffers[split].append(sample)
            sample_count += 1
            split_counts[split] = split_counts.get(split, 0) + 1
            if len(buffers[split]) >= shard_limit:
                shards.append(_flush_shard(root, split, len(shards), buffers[split]))
                buffers[split] = []
        if max_samples is not None and sample_count >= max_samples:
            break

    for split, buffer in list(buffers.items()):
        if buffer:
            shards.append(_flush_shard(root, split, len(shards), buffer))

    if not shards:
        raise ValueError("no planner samples were built")

    manifest = {
        "format": "planner_feature_cache",
        "version": 1,
        "supervision": "action_segment_latent",
        "target": target_config,
        "num_samples": sample_count,
        "split_counts": split_counts,
        "shards": shards,
    }
    manifest_path = root / str(data_config.get("manifest", "manifest.json"))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def build_synthetic_feature_cache(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    num_episodes: int = 4,
    episode_length: int = 12,
    hidden_dim: int = 8,
    state_dim: int = 4,
    action_dim: int = 3,
    num_tokens: int = 6,
) -> dict[str, Any]:
    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    episodes = []
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    for episode_index in range(num_episodes):
        actions = torch.randn(episode_length, action_dim, generator=generator)
        states = torch.randn(episode_length, state_dim, generator=generator)
        vlm_tokens = torch.randn(episode_length, num_tokens, hidden_dim, generator=generator)
        episodes.append(
            {
                "episode_id": f"synthetic_{episode_index}",
                "vlm_tokens": vlm_tokens,
                "states": states,
                "actions": actions,
                "frame_index": torch.arange(episode_length),
            }
        )
    source_path = root / "synthetic_source.pt"
    torch.save({"episodes": episodes}, source_path)
    synthetic_config = _deepcopy_config(config)
    synthetic_config["data"]["root"] = str(root)
    synthetic_config["data"]["input_paths"] = [str(source_path)]
    return build_planner_feature_cache(synthetic_config, output_root=root)


def _iter_episode_samples(
    episode: dict[str, Any], data_config: dict[str, Any], target_config: dict[str, Any]
) -> list[dict[str, Any]]:
    vlm_tokens = _as_numpy(episode["vlm_tokens"])
    states = _as_numpy(episode["states"])
    actions = _as_numpy(episode["actions"])
    if vlm_tokens.ndim != 3:
        raise ValueError(f"vlm_tokens must have shape [T, L, D], got {vlm_tokens.shape}")
    if states.ndim != 2:
        raise ValueError(f"states must have shape [T, S], got {states.shape}")
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [T, A], got {actions.shape}")
    if not (vlm_tokens.shape[0] == states.shape[0] == actions.shape[0]):
        raise ValueError("vlm_tokens, states, and actions must have matching time dimension")

    planning_horizon = int(target_config["planning_horizon"])
    include_tail = bool(data_config.get("include_tail", True))
    stride = int(data_config.get("stride", 1))
    if stride <= 0:
        raise ValueError("data.stride must be positive")

    samples = []
    frame_index = _frame_indices(episode, length=actions.shape[0])
    for timestep in range(0, actions.shape[0], stride):
        valid_count = min(planning_horizon, actions.shape[0] - timestep)
        if valid_count < planning_horizon and not include_tail:
            continue
        future = np.zeros((planning_horizon, actions.shape[1]), dtype=np.float32)
        future[:valid_count] = actions[timestep : timestep + valid_count]
        action_segments, action_segment_mask = build_action_segment_target(
            future,
            num_plan_steps=int(target_config["num_plan_steps"]),
            planning_horizon=planning_horizon,
            valid_action_count=valid_count,
        )
        samples.append(
            {
                "vlm_tokens": vlm_tokens[timestep],
                "state": states[timestep],
                "action_segments": action_segments,
                "action_segment_mask": action_segment_mask.astype(np.float32),
                "episode_id": str(episode["episode_id"]),
                "frame_index": int(frame_index[timestep]),
                "source_path": str(episode.get("source_path", "")),
            }
        )
    return samples


def _flush_shard(root: Path, split: str, shard_index: int, samples: list[dict[str, Any]]) -> dict[str, Any]:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    filename = f"planner_samples_{shard_index:05d}.pt"
    path = split_dir / filename
    payload = {
        "vlm_tokens": torch.tensor(np.stack([sample["vlm_tokens"] for sample in samples]), dtype=torch.float32),
        "state": torch.tensor(np.stack([sample["state"] for sample in samples]), dtype=torch.float32),
        "action_segments": torch.tensor(np.stack([sample["action_segments"] for sample in samples]), dtype=torch.float32),
        "action_segment_mask": torch.tensor(
            np.stack([sample["action_segment_mask"] for sample in samples]), dtype=torch.float32
        ),
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


def _load_episodes(path: Path, data_config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = sorted(path.glob("*.pt")) + sorted(path.glob("*.npz")) if path.is_dir() else [path]
    episodes: list[dict[str, Any]] = []
    for source_path in paths:
        if source_path.suffix == ".pt":
            raw = torch.load(source_path, map_location="cpu", weights_only=False)
        elif source_path.suffix == ".npz":
            raw = dict(np.load(source_path, allow_pickle=True))
        else:
            continue
        episodes.extend(_normalize_loaded_source(raw, source_path, data_config))
    return episodes


def _normalize_loaded_source(raw: Any, source_path: Path, data_config: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and "episodes" in raw:
        return [_normalize_episode(episode, source_path, data_config, fallback_id=index) for index, episode in enumerate(raw["episodes"])]
    if isinstance(raw, list):
        return [_normalize_episode(episode, source_path, data_config, fallback_id=index) for index, episode in enumerate(raw)]
    if isinstance(raw, dict):
        return [_normalize_episode(raw, source_path, data_config, fallback_id=0)]
    raise ValueError(f"unsupported planner feature source: {source_path}")


def _normalize_episode(
    raw: dict[str, Any], source_path: Path, data_config: dict[str, Any], *, fallback_id: int
) -> dict[str, Any]:
    vlm_key = str(data_config.get("vlm_token_key", "vlm_tokens"))
    state_key = str(data_config.get("state_key", "states"))
    action_key = str(data_config.get("action_key", "actions"))
    episode_key = str(data_config.get("episode_key", "episode_id"))
    frame_key = str(data_config.get("frame_key", "frame_index"))
    if vlm_key not in raw or state_key not in raw or action_key not in raw:
        raise KeyError(f"{source_path} must contain {vlm_key!r}, {state_key!r}, and {action_key!r}")
    episode_id = raw.get(episode_key, f"{source_path.stem}:{fallback_id}")
    return {
        "episode_id": str(episode_id),
        "vlm_tokens": raw[vlm_key],
        "states": raw[state_key],
        "actions": raw[action_key],
        "frame_index": raw.get(frame_key),
        "source_path": str(source_path),
    }


def _split_episode_ids(episode_ids: list[str], *, seed: int, val_fraction: float) -> tuple[set[str], set[str]]:
    unique_ids = sorted(set(episode_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    if len(unique_ids) <= 1:
        return set(unique_ids), set(unique_ids)
    val_count = max(1, int(round(len(unique_ids) * val_fraction)))
    val_count = min(val_count, len(unique_ids) - 1)
    eval_ids = set(unique_ids[:val_count])
    train_ids = set(unique_ids[val_count:])
    return train_ids, eval_ids


def _sample_split(sample_count: int, train_ids: set[str], config: dict[str, Any], data_config: dict[str, Any]) -> str:
    val_fraction = float(data_config.get("val_fraction", 0.1))
    rng = random.Random(int(config.get("seed", 42)) + sample_count)
    return str(data_config.get("eval_split", "eval")) if rng.random() < val_fraction else str(data_config.get("train_split", "train"))


def _frame_indices(episode: dict[str, Any], *, length: int) -> np.ndarray:
    frame_index = episode.get("frame_index")
    if frame_index is None:
        return np.arange(length, dtype=np.int64)
    frame_array = _as_numpy(frame_index).reshape(-1)
    if frame_array.shape[0] != length:
        raise ValueError(f"frame_index length {frame_array.shape[0]} does not match episode length {length}")
    return frame_array.astype(np.int64)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _deepcopy_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config))
