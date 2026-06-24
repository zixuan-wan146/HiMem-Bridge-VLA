from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from himem_bridge_vla.dataset.memory_replay_dataset import MemoryReplayFrameDataset
from himem_bridge_vla.model.himem.dual_fifo_memory import MemoryReadResult
from himem_bridge_vla.model.himem.dual_fifo_memory import SHORT_MEMORY
from himem_bridge_vla.model.himem.dual_fifo_memory import VisualMemoryEntry


MEMORY_TOKEN_CACHE_FORMAT = "memory_replay_visual_token_cache"
MEMORY_TOKEN_CACHE_VERSION = 1
DEFAULT_TOKEN_CACHE_SHARD_SIZE = 1024


class VisualTokenEncoder(Protocol):
    """Encode one RGB image into visual tokens with shape [num_tokens, hidden_dim]."""

    name: str
    hidden_dim: int
    tokens_per_view: int | None

    def encode_image(self, image: Image.Image) -> Any:
        ...


@dataclass(frozen=True)
class TokenCacheShard:
    path: Path
    sample_count: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class TokenCacheBuildResult:
    output_root: Path
    manifest_path: Path
    sample_count: int
    shards: tuple[TokenCacheShard, ...]


@dataclass(frozen=True)
class TokenCacheDatasetConfig:
    manifest_path: Path
    output_root: Path
    benchmark: str
    sample_count: int
    hidden_dim: int
    storage_dtype: str


class ImageStatsVisualTokenEncoder:
    """Small deterministic encoder used for tests and pipeline smoke checks.

    This encoder is intentionally not a training feature extractor. It lets the
    replay-cache IO path run without model downloads or GPU allocation.
    """

    name = "image_stats"

    def __init__(self, *, hidden_dim: int = 16, tokens_per_view: int = 1) -> None:
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(tokens_per_view) <= 0:
            raise ValueError("tokens_per_view must be positive")
        self.hidden_dim = int(hidden_dim)
        self.tokens_per_view = int(tokens_per_view)

    def encode_image(self, image: Image.Image) -> Any:
        torch = _require_torch()
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        flat = rgb.reshape(-1, 3)
        stats = np.concatenate(
            [
                flat.mean(axis=0),
                flat.std(axis=0),
                flat.min(axis=0),
                flat.max(axis=0),
            ],
            axis=0,
        )
        values = np.resize(stats, self.hidden_dim * self.tokens_per_view).reshape(
            self.tokens_per_view,
            self.hidden_dim,
        )
        return torch.tensor(values, dtype=torch.float32)


class InternVL3VisualTokenEncoder:
    """InternVL3 visual-tower encoder for replay visual token caches."""

    name = "internvl3"
    tokens_per_view = None

    def __init__(
        self,
        *,
        model_name: str = "OpenGVLab/InternVL3-1B",
        image_size: int = 448,
        device: str = "cuda",
        storage_dtype: str = "bfloat16",
    ) -> None:
        torch = _require_torch()
        from himem_bridge_vla.model.internvl3.internvl3_embedder import InternVL3Embedder

        self.embedder = InternVL3Embedder(model_name=model_name, image_size=image_size, device=device)
        self.embedder.eval()
        self.device = str(device)
        self.storage_dtype = resolve_torch_dtype(storage_dtype)
        self.hidden_dim = int(getattr(self.embedder.model, "llm_hidden_size", 0) or 0)
        if self.hidden_dim <= 0:
            with torch.no_grad():
                tokens = self.encode_image(Image.new("RGB", (image_size, image_size)))
            self.hidden_dim = int(tokens.shape[-1])

    def encode_image(self, image: Image.Image) -> Any:
        torch = _require_torch()
        with torch.no_grad():
            pixel_values, _num_tiles = self.embedder._preprocess_images([image])
            tokens = self.embedder.model.extract_feature(pixel_values)
        tokens = flatten_visual_tokens(tokens).to(dtype=self.storage_dtype).cpu()
        if tokens.ndim != 2:
            raise ValueError(
                f"InternVL3 visual tokens must be rank-2 after flattening, got {tuple(tokens.shape)}"
            )
        return tokens


def build_memory_replay_token_cache(
    *,
    benchmark: str,
    data_root: str | Path,
    index_path: str | Path,
    output_root: str | Path,
    encoder: VisualTokenEncoder,
    view_names: Sequence[str] | None = None,
    max_samples: int | None = None,
    max_samples_per_shard: int = DEFAULT_TOKEN_CACHE_SHARD_SIZE,
    storage_dtype: str = "bfloat16",
    manifest_extra: Mapping[str, Any] | None = None,
) -> TokenCacheBuildResult:
    max_samples_per_shard = int(max_samples_per_shard)
    if max_samples_per_shard <= 0:
        raise ValueError("max_samples_per_shard must be positive")

    dataset = MemoryReplayFrameDataset(
        benchmark=benchmark,
        data_root=data_root,
        index_path=index_path,
        view_names=view_names,
        max_samples=max_samples,
    )
    output_path = Path(output_root).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    target_dtype = resolve_torch_dtype(storage_dtype)
    pending: list[dict[str, Any]] = []
    shards: list[TokenCacheShard] = []
    sample_count = 0

    for dataset_index in range(len(dataset)):
        item = dataset[dataset_index]
        pending.append(
            encode_memory_replay_item(
                item,
                encoder=encoder,
                storage_dtype=target_dtype,
                sample_index=dataset_index,
            )
        )
        sample_count += 1
        if len(pending) >= max_samples_per_shard:
            shards.append(_write_token_cache_shard(shard_dir, pending, start_index=sample_count - len(pending)))
            pending = []

    if pending:
        shards.append(_write_token_cache_shard(shard_dir, pending, start_index=sample_count - len(pending)))

    manifest = build_token_cache_manifest(
        benchmark=benchmark,
        data_root=data_root,
        index_path=index_path,
        output_root=output_path,
        encoder=encoder,
        storage_dtype=storage_dtype,
        sample_count=sample_count,
        max_samples=max_samples,
        max_samples_per_shard=max_samples_per_shard,
        view_names=view_names,
        shards=shards,
        extra=manifest_extra,
    )
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return TokenCacheBuildResult(
        output_root=output_path,
        manifest_path=manifest_path,
        sample_count=sample_count,
        shards=tuple(shards),
    )


def encode_memory_replay_item(
    item: Mapping[str, Any],
    *,
    encoder: VisualTokenEncoder,
    storage_dtype: Any,
    sample_index: int,
) -> dict[str, Any]:
    torch = _require_torch()
    current_tokens = encode_images_by_view(item["current_images"], encoder=encoder, storage_dtype=storage_dtype)
    short_tokens = tuple(
        None
        if images_by_view is None
        else encode_images_by_view(images_by_view, encoder=encoder, storage_dtype=storage_dtype)
        for images_by_view in item["short_images"]
    )
    return {
        "sample_index": int(sample_index),
        "benchmark": str(item["benchmark"]),
        "episode_id": str(item["episode_id"]),
        "current_step": int(item["current_step"]),
        "current_tokens_by_view": current_tokens,
        "current_state": torch.as_tensor(item["current_state"], dtype=torch.float32).cpu(),
        "short_tokens_by_view": short_tokens,
        "short_steps": _short_steps_tensor(item, len(short_tokens)),
        "short_mask": torch.as_tensor(item["short_mask"], dtype=torch.bool).cpu(),
        "future_actions": torch.as_tensor(item["future_actions"], dtype=torch.float32).cpu(),
        "action_valid_count": int(item["action_valid_count"]),
    }


def encode_images_by_view(
    images_by_view: Mapping[str, Image.Image],
    *,
    encoder: VisualTokenEncoder,
    storage_dtype: Any,
) -> dict[str, Any]:
    tokens_by_view = {}
    for view_name, image in images_by_view.items():
        tokens = encoder.encode_image(image)
        tokens_by_view[str(view_name)] = ensure_rank2_tokens(tokens, storage_dtype=storage_dtype)
    return tokens_by_view


def ensure_rank2_tokens(tokens: Any, *, storage_dtype: Any) -> Any:
    torch = _require_torch()
    tensor = torch.as_tensor(tokens).detach().cpu()
    tensor = flatten_visual_tokens(tensor)
    if tensor.ndim != 2:
        raise ValueError(f"visual tokens must have shape [num_tokens, hidden_dim], got {tuple(tensor.shape)}")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"visual tokens must be non-empty, got {tuple(tensor.shape)}")
    return tensor.to(dtype=storage_dtype).contiguous()


def flatten_visual_tokens(tokens: Any) -> Any:
    torch = _require_torch()
    tensor = torch.as_tensor(tokens)
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
    if tensor.ndim == 4:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1] * tensor.shape[2], tensor.shape[3])
    raise ValueError(f"unsupported visual token tensor rank: {tensor.ndim}")


def build_token_cache_manifest(
    *,
    benchmark: str,
    data_root: str | Path,
    index_path: str | Path,
    output_root: str | Path,
    encoder: VisualTokenEncoder,
    storage_dtype: str,
    sample_count: int,
    max_samples: int | None,
    max_samples_per_shard: int,
    view_names: Sequence[str] | None,
    shards: Sequence[TokenCacheShard],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": MEMORY_TOKEN_CACHE_FORMAT,
        "version": MEMORY_TOKEN_CACHE_VERSION,
        "benchmark": str(benchmark).upper(),
        "data_root": str(Path(data_root).expanduser()),
        "index_path": str(Path(index_path).expanduser()),
        "output_root": str(Path(output_root).expanduser()),
        "encoder": encoder.name,
        "hidden_dim": int(encoder.hidden_dim),
        "tokens_per_view": None if encoder.tokens_per_view is None else int(encoder.tokens_per_view),
        "storage_dtype": str(storage_dtype),
        "sample_count": int(sample_count),
        "max_samples": None if max_samples is None else int(max_samples),
        "max_samples_per_shard": int(max_samples_per_shard),
        "view_names": None if view_names is None else [str(name) for name in view_names],
        "shards": [
            {
                "path": str(shard.path.relative_to(Path(output_root).expanduser())),
                "sample_count": shard.sample_count,
                "start_index": shard.start_index,
                "end_index": shard.end_index,
            }
            for shard in shards
        ],
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


class MemoryTokenCacheDataset:
    """PyTorch-compatible dataset over replay visual-token cache shards."""

    def __init__(self, manifest_path: str | Path, *, max_samples: int | None = None) -> None:
        self.manifest_path = resolve_token_cache_manifest_path(manifest_path)
        self.manifest = read_token_cache_manifest(self.manifest_path)
        _validate_token_cache_manifest(self.manifest, self.manifest_path)
        self.output_root = self.manifest_path.parent
        self.shards = tuple(_resolve_manifest_shards(self.manifest, self.output_root))
        self.shard_end_indices = tuple(shard.end_index for shard in self.shards)

        sample_count = int(self.manifest["sample_count"])
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive when provided")
            sample_count = min(sample_count, int(max_samples))
        self.sample_count = sample_count
        self.config = TokenCacheDatasetConfig(
            manifest_path=self.manifest_path,
            output_root=self.output_root,
            benchmark=str(self.manifest["benchmark"]),
            sample_count=self.sample_count,
            hidden_dim=int(self.manifest["hidden_dim"]),
            storage_dtype=str(self.manifest["storage_dtype"]),
        )
        self._loaded_shard_index: int | None = None
        self._loaded_shard_samples: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += self.sample_count
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)
        shard_index = bisect_right(self.shard_end_indices, index)
        if shard_index >= len(self.shards):
            raise IndexError(index)
        shard = self.shards[shard_index]
        samples = self._load_shard_samples(shard_index)
        local_index = index - shard.start_index
        return normalize_token_cache_sample(samples[local_index])

    def _load_shard_samples(self, shard_index: int) -> list[dict[str, Any]]:
        if self._loaded_shard_index == shard_index and self._loaded_shard_samples is not None:
            return self._loaded_shard_samples
        shard = self.shards[shard_index]
        payload = _torch_load(shard.path)
        if payload.get("format") != MEMORY_TOKEN_CACHE_FORMAT:
            raise ValueError(f"invalid token cache shard format in {shard.path}")
        samples = list(payload.get("samples", []))
        if len(samples) != shard.sample_count:
            raise ValueError(
                f"shard {shard.path} manifest sample_count={shard.sample_count} "
                f"but file has {len(samples)} samples"
            )
        self._loaded_shard_index = shard_index
        self._loaded_shard_samples = samples
        return samples


def collate_memory_token_cache_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must contain at least one item")
    torch = _require_torch()
    future_actions, action_mask = _pad_future_actions([sample["future_actions"] for sample in batch])
    return {
        "benchmark": [str(sample["benchmark"]) for sample in batch],
        "episode_id": [str(sample["episode_id"]) for sample in batch],
        "sample_index": torch.tensor([int(sample["sample_index"]) for sample in batch], dtype=torch.long),
        "current_step": torch.tensor([int(sample["current_step"]) for sample in batch], dtype=torch.long),
        "current_tokens_by_view": [sample["current_tokens_by_view"] for sample in batch],
        "current_state": torch.stack(
            [torch.as_tensor(sample["current_state"], dtype=torch.float32) for sample in batch]
        ),
        "short_tokens_by_view": [sample["short_tokens_by_view"] for sample in batch],
        "short_steps": torch.stack([torch.as_tensor(sample["short_steps"], dtype=torch.long) for sample in batch]),
        "short_mask": torch.stack([torch.as_tensor(sample["short_mask"], dtype=torch.bool) for sample in batch]),
        "short_memory": [memory_read_result_from_token_cache_sample(sample) for sample in batch],
        "future_actions": future_actions,
        "action_mask": action_mask,
        "action_valid_count": torch.tensor([int(sample["action_valid_count"]) for sample in batch], dtype=torch.long),
    }


def memory_read_result_from_token_cache_sample(
    sample: Mapping[str, Any],
    *,
    long_capacity: int = 0,
) -> MemoryReadResult:
    torch = _require_torch()
    short_tokens = tuple(sample["short_tokens_by_view"])
    short_mask = torch.as_tensor(sample["short_mask"], dtype=torch.bool).reshape(-1)
    short_steps = torch.as_tensor(sample["short_steps"], dtype=torch.long).reshape(-1)
    if len(short_tokens) != int(short_mask.numel()) or len(short_tokens) != int(short_steps.numel()):
        raise ValueError("short token, mask, and step lengths must match")

    entries: list[VisualMemoryEntry | None] = []
    for tokens_by_view, is_valid, tau in zip(short_tokens, short_mask.tolist(), short_steps.tolist(), strict=True):
        if not is_valid:
            entries.append(None)
            continue
        if tokens_by_view is None:
            raise ValueError("short_mask marks an empty token slot as valid")
        if int(tau) < 0:
            raise ValueError("valid short memory entry must have a non-negative tau")
        entries.append(VisualMemoryEntry(visual_tokens_by_view=tokens_by_view, tau=int(tau), eta=SHORT_MEMORY))

    if int(long_capacity) < 0:
        raise ValueError("long_capacity must be non-negative")
    entries.extend([None] * int(long_capacity))
    entry_mask = torch.tensor([entry is not None for entry in entries], dtype=torch.bool)
    return MemoryReadResult(
        entries=tuple(entries),
        entry_mask=entry_mask,
        short_capacity=len(short_tokens),
        long_capacity=int(long_capacity),
    )


def normalize_token_cache_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    torch = _require_torch()
    normalized = dict(sample)
    short_tokens = tuple(normalized.get("short_tokens_by_view", ()))
    normalized["short_tokens_by_view"] = short_tokens
    normalized["short_mask"] = torch.as_tensor(normalized["short_mask"], dtype=torch.bool).cpu()
    normalized["short_steps"] = _short_steps_tensor(normalized, len(short_tokens))
    normalized["current_state"] = torch.as_tensor(normalized["current_state"], dtype=torch.float32).cpu()
    normalized["future_actions"] = torch.as_tensor(normalized["future_actions"], dtype=torch.float32).cpu()
    normalized["action_valid_count"] = int(normalized["action_valid_count"])
    normalized["current_step"] = int(normalized["current_step"])
    normalized["sample_index"] = int(normalized.get("sample_index", -1))
    normalized["benchmark"] = str(normalized["benchmark"])
    normalized["episode_id"] = str(normalized["episode_id"])
    return normalized


def read_token_cache_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = resolve_token_cache_manifest_path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_token_cache_manifest_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_dir():
        resolved = resolved / "manifest.json"
    return resolved


def resolve_torch_dtype(name: str) -> Any:
    torch = _require_torch()
    normalized = str(name).lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported storage dtype: {name!r}")


def _write_token_cache_shard(shard_dir: Path, samples: Sequence[Mapping[str, Any]], *, start_index: int) -> TokenCacheShard:
    torch = _require_torch()
    if not samples:
        raise ValueError("cannot write an empty token cache shard")
    end_index = int(start_index) + len(samples)
    shard_path = shard_dir / f"shard_{int(start_index):09d}_{end_index:09d}.pt"
    torch.save(
        {"format": MEMORY_TOKEN_CACHE_FORMAT, "version": MEMORY_TOKEN_CACHE_VERSION, "samples": list(samples)},
        shard_path,
    )
    return TokenCacheShard(
        path=shard_path,
        sample_count=len(samples),
        start_index=int(start_index),
        end_index=end_index,
    )


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("memory token cache utilities require torch") from exc
    return torch


def _short_steps_tensor(item: Mapping[str, Any], short_count: int) -> Any:
    torch = _require_torch()
    if "short_steps" not in item:
        return torch.full((int(short_count),), -1, dtype=torch.long)
    raw_steps = item["short_steps"]
    if isinstance(raw_steps, torch.Tensor):
        steps = raw_steps.to(dtype=torch.long).reshape(-1).cpu()
    else:
        steps = torch.as_tensor(raw_steps, dtype=torch.long).reshape(-1).cpu()
    if steps.numel() != int(short_count):
        raise ValueError(f"short_steps has {steps.numel()} values for {short_count} short entries")
    return steps


def _validate_token_cache_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    if manifest.get("format") != MEMORY_TOKEN_CACHE_FORMAT:
        raise ValueError(f"invalid token cache format in {manifest_path}: {manifest.get('format')!r}")
    if int(manifest.get("version", -1)) != MEMORY_TOKEN_CACHE_VERSION:
        raise ValueError(f"unsupported token cache version in {manifest_path}: {manifest.get('version')!r}")
    if int(manifest.get("sample_count", -1)) < 0:
        raise ValueError(f"token cache manifest has invalid sample_count: {manifest.get('sample_count')!r}")
    if int(manifest.get("hidden_dim", 0)) <= 0:
        raise ValueError(f"token cache manifest has invalid hidden_dim: {manifest.get('hidden_dim')!r}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"token cache manifest has no shards: {manifest_path}")


def _resolve_manifest_shards(manifest: Mapping[str, Any], output_root: Path) -> list[TokenCacheShard]:
    shards: list[TokenCacheShard] = []
    expected_start = 0
    for raw_shard in manifest["shards"]:
        shard = TokenCacheShard(
            path=output_root / str(raw_shard["path"]),
            sample_count=int(raw_shard["sample_count"]),
            start_index=int(raw_shard["start_index"]),
            end_index=int(raw_shard["end_index"]),
        )
        if shard.start_index != expected_start:
            raise ValueError(
                f"non-contiguous token cache shard starts at {shard.start_index}, expected {expected_start}"
            )
        if shard.end_index - shard.start_index != shard.sample_count:
            raise ValueError(f"token cache shard has inconsistent index range: {shard}")
        if not shard.path.exists():
            raise FileNotFoundError(f"token cache shard does not exist: {shard.path}")
        shards.append(shard)
        expected_start = shard.end_index
    if expected_start != int(manifest["sample_count"]):
        raise ValueError(f"manifest sample_count={manifest['sample_count']} but shards end at {expected_start}")
    return shards


def _pad_future_actions(actions: Sequence[Any]) -> tuple[Any, Any]:
    torch = _require_torch()
    tensors = [torch.as_tensor(action, dtype=torch.float32).cpu() for action in actions]
    if not tensors:
        raise ValueError("actions must not be empty")
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("future_actions must have shape [T, A]")
    action_dim = int(tensors[0].shape[-1])
    if any(int(tensor.shape[-1]) != action_dim for tensor in tensors):
        raise ValueError("all future_actions tensors in a batch must share action dim")
    max_steps = max(int(tensor.shape[0]) for tensor in tensors)
    batch = torch.zeros(len(tensors), max_steps, action_dim, dtype=torch.float32)
    mask = torch.zeros(len(tensors), max_steps, dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        step_count = int(tensor.shape[0])
        batch[index, :step_count] = tensor
        mask[index, :step_count] = True
    return batch, mask


def _torch_load(path: str | Path) -> Any:
    torch = _require_torch()
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)
