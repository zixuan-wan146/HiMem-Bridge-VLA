from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from himem_bridge_vla.dataset.memory_replay_dataset import MemoryReplayFrameDataset
MEMORY_TOKEN_CACHE_FORMAT = "memory_replay_visual_token_cache"
MEMORY_TOKEN_CACHE_VERSION = 1
EPISODE_FEATURE_CACHE_FORMAT = "libero_episode_feature_cache"
EPISODE_FEATURE_CACHE_VERSION = 1
DEFAULT_TOKEN_CACHE_SHARD_SIZE = 1024


class VisualTokenEncoder(Protocol):
    """Encode one RGB image into visual tokens with shape [num_tokens, hidden_dim]."""

    name: str
    hidden_dim: int
    tokens_per_view: int | None

    def encode_image(self, image: Image.Image) -> Any:
        ...

    def encode_images(self, images: Sequence[Image.Image]) -> Sequence[Any]:
        ...


class VLMHiddenStateEncoder(Protocol):
    """Encode current observation and prompt into selected VLM hidden-state layers."""

    name: str
    hidden_dim: int
    selected_layers: tuple[int | str, ...]

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> tuple[Any, ...]:
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


@dataclass(frozen=True)
class VLMCurrentFeatures:
    hidden_states: tuple[Any, ...]
    planner_vl_summary: Any | None = None


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

    def encode_images(self, images: Sequence[Image.Image]) -> list[Any]:
        return [self.encode_image(image) for image in images]


class ImageStatsVLMHiddenStateEncoder:
    """Deterministic hidden-state stand-in for cache IO tests."""

    name = "image_stats_vlm_hidden_states"

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        tokens_per_view: int = 1,
        selected_layers: Sequence[int | str] = (3, 6, 9, 12),
    ) -> None:
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(tokens_per_view) <= 0:
            raise ValueError("tokens_per_view must be positive")
        self.hidden_dim = int(hidden_dim)
        self.tokens_per_view = int(tokens_per_view)
        self.selected_layers = tuple(selected_layers)
        self._visual = ImageStatsVisualTokenEncoder(hidden_dim=self.hidden_dim, tokens_per_view=self.tokens_per_view)

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> tuple[Any, ...]:
        return self.encode_current_features(images_by_view, prompt).hidden_states

    def encode_current_features(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> VLMCurrentFeatures:
        torch = _require_torch()
        base_tokens = torch.cat(
            [self._visual.encode_image(image) for _view_name, image in sorted(images_by_view.items())],
            dim=0,
        )
        prompt_offset = min(len(str(prompt)), 512) / 512.0
        hidden_states = tuple(
            base_tokens + (layer_index + 1) * 0.01 + prompt_offset
            for layer_index, _ in enumerate(self.selected_layers)
        )
        return VLMCurrentFeatures(
            hidden_states=hidden_states,
            planner_vl_summary=hidden_states[-1][-1].to(dtype=torch.float32),
        )


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

    def encode_images(self, images: Sequence[Image.Image]) -> list[Any]:
        torch = _require_torch()
        images = list(images)
        if not images:
            return []
        with torch.no_grad():
            pixel_values, num_tiles_list = self.embedder._preprocess_images(images)
            tokens = self.embedder.model.extract_feature(pixel_values)
        token_tensor = torch.as_tensor(tokens)
        encoded: list[Any] = []
        cursor = 0
        for tile_count in num_tiles_list:
            tile_count = int(tile_count)
            image_tokens = token_tensor[cursor : cursor + tile_count]
            encoded.append(flatten_visual_tokens(image_tokens).to(dtype=self.storage_dtype).cpu())
            cursor += tile_count
        if cursor != int(token_tensor.shape[0]):
            raise ValueError(
                f"InternVL3 visual batch split consumed {cursor} tiles, "
                f"but encoder returned {int(token_tensor.shape[0])}"
            )
        return encoded


class InternVL3VLMHiddenStateEncoder:
    """InternVL3 language-model hidden-state encoder for current replay observations."""

    name = "internvl3_vlm_hidden_states"

    def __init__(
        self,
        *,
        model_name: str = "OpenGVLab/InternVL3-1B",
        image_size: int = 448,
        device: str = "cuda",
        storage_dtype: str = "bfloat16",
        selected_layers: Sequence[int | str] = (3, 6, 9, 12),
        embedder: Any | None = None,
    ) -> None:
        from himem_bridge_vla.model.internvl3.internvl3_embedder import InternVL3Embedder

        self.embedder = embedder or InternVL3Embedder(model_name=model_name, image_size=image_size, device=device)
        self.embedder.eval()
        self.device = str(device)
        self.storage_dtype = resolve_torch_dtype(storage_dtype)
        self.selected_layers = tuple(selected_layers)
        self.hidden_dim = int(getattr(self.embedder.model, "llm_hidden_size", 0) or 0)

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> tuple[Any, ...]:
        return self.encode_current_features(images_by_view, prompt).hidden_states

    def encode_current_features(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> VLMCurrentFeatures:
        torch = _require_torch()
        images = list(images_by_view.values())
        if not images:
            raise ValueError("images_by_view must contain at least one image")
        image_mask = torch.ones(len(images), dtype=torch.bool)
        with torch.no_grad():
            output = self.embedder.get_fused_image_text_embedding_from_tensor_images(
                image_tensors=images,
                image_mask=image_mask,
                text_prompt=str(prompt),
                return_cls_only=False,
                return_hidden_states=True,
                selected_layers=self.selected_layers,
            )
        hidden_states = tuple(
            flatten_visual_tokens(hidden_state).to(dtype=self.storage_dtype).cpu()
            for hidden_state in output.hidden_states
        )
        if not hidden_states:
            raise ValueError("InternVL3 returned no hidden states")
        if self.hidden_dim <= 0:
            self.hidden_dim = int(hidden_states[0].shape[-1])
        planner_vl_summary = getattr(output, "planner_vl_summary", None)
        if planner_vl_summary is not None:
            planner_vl_summary = torch.as_tensor(planner_vl_summary).reshape(-1, self.hidden_dim)[0].to(
                dtype=self.storage_dtype
            ).cpu()
        return VLMCurrentFeatures(hidden_states=hidden_states, planner_vl_summary=planner_vl_summary)


def build_memory_replay_token_cache(
    *,
    benchmark: str,
    data_root: str | Path,
    index_path: str | Path,
    output_root: str | Path,
    encoder: VisualTokenEncoder,
    hidden_state_encoder: VLMHiddenStateEncoder | None = None,
    view_names: Sequence[str] | None = None,
    max_samples: int | None = None,
    max_samples_per_shard: int = DEFAULT_TOKEN_CACHE_SHARD_SIZE,
    storage_dtype: str = "bfloat16",
    manifest_extra: Mapping[str, Any] | None = None,
) -> TokenCacheBuildResult:
    if str(benchmark).upper() == "LIBERO":
        raise ValueError(
            "LIBERO Stage1 no longer uses memory_replay_visual_token_cache. "
            "Build the active cache with build_libero_episode_replay_index.py followed by "
            "build_libero_episode_feature_cache.py."
        )
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
    visual_token_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    hidden_state_cache: dict[tuple[str, str, int, str], VLMCurrentFeatures] = {}
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None

    for dataset_index in range(len(dataset)):
        item = dataset[dataset_index]
        action_min, action_max = _update_running_minmax(action_min, action_max, item["future_actions"], name="future_actions")
        state_min, state_max = _update_running_minmax(state_min, state_max, item["current_state"], name="current_state")
        pending.append(
            encode_memory_replay_item(
                item,
                encoder=encoder,
                hidden_state_encoder=hidden_state_encoder,
                storage_dtype=target_dtype,
                sample_index=dataset_index,
                visual_token_cache=visual_token_cache,
                hidden_state_cache=hidden_state_cache,
            )
        )
        sample_count += 1
        if len(pending) >= max_samples_per_shard:
            shards.append(_write_token_cache_shard(shard_dir, pending, start_index=sample_count - len(pending)))
            pending = []

    if pending:
        shards.append(_write_token_cache_shard(shard_dir, pending, start_index=sample_count - len(pending)))

    extra = dict(manifest_extra or {})
    extra.setdefault("builder_mode", "frame_token_dedup")
    extra["visual_token_cache_entries"] = len(visual_token_cache)
    if hidden_state_encoder is not None:
        extra["hidden_state_cache_entries"] = len(hidden_state_cache)
    if action_min is not None and action_max is not None and state_min is not None and state_max is not None:
        extra["normalization"] = _build_minmax_normalization_manifest(
            benchmark=benchmark,
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max,
        )
        extra["action_normalization"] = {
            "enabled": True,
            "type": "train_split_minmax_to_minus_one_one",
            "clip_after_normalization": True,
            "clip_range": [-1.0, 1.0],
            "statistics_from": "cache_build_rows",
        }

    manifest = build_token_cache_manifest(
        benchmark=benchmark,
        data_root=data_root,
        index_path=index_path,
        output_root=output_path,
        encoder=encoder,
        hidden_state_encoder=hidden_state_encoder,
        storage_dtype=storage_dtype,
        sample_count=sample_count,
        max_samples=max_samples,
        max_samples_per_shard=max_samples_per_shard,
        view_names=view_names,
        shards=shards,
        extra=extra,
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
    hidden_state_encoder: VLMHiddenStateEncoder | None = None,
    storage_dtype: Any,
    sample_index: int,
    visual_token_cache: dict[tuple[str, str, int], dict[str, Any]] | None = None,
    hidden_state_cache: dict[tuple[str, str, int, str], VLMCurrentFeatures] | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    benchmark = str(item["benchmark"])
    episode_id = str(item["episode_id"])
    current_step = int(item["current_step"])
    current_tokens = _get_or_encode_frame_tokens(
        item["current_images"],
        cache_key=(benchmark, episode_id, current_step),
        encoder=encoder,
        storage_dtype=storage_dtype,
        visual_token_cache=visual_token_cache,
    )
    short_entries = tuple(item["short_images"])
    short_steps = _short_steps_tensor(item, len(short_entries))
    short_tokens = tuple(
        None
        if images_by_view is None or int(short_steps[entry_index].item()) < 0
        else _get_or_encode_frame_tokens(
            images_by_view,
            cache_key=(benchmark, episode_id, int(short_steps[entry_index].item())),
            encoder=encoder,
            storage_dtype=storage_dtype,
            visual_token_cache=visual_token_cache,
        )
        for entry_index, images_by_view in enumerate(short_entries)
    )
    encoded = {
        "sample_index": int(sample_index),
        "benchmark": benchmark,
        "episode_id": episode_id,
        "prompt": str(item.get("prompt", "")),
        "current_step": current_step,
        "current_tokens_by_view": current_tokens,
        "current_state": torch.as_tensor(item["current_state"], dtype=torch.float32).cpu(),
        "short_tokens_by_view": short_tokens,
        "short_steps": short_steps,
        "short_mask": torch.as_tensor(item["short_mask"], dtype=torch.bool).cpu(),
        "executed_actions": torch.as_tensor(item["executed_actions"], dtype=torch.float32).cpu(),
        "executed_action_mask": torch.as_tensor(item["executed_action_mask"], dtype=torch.bool).cpu(),
        "future_actions": torch.as_tensor(item["future_actions"], dtype=torch.float32).cpu(),
        "action_valid_count": int(item["action_valid_count"]),
    }
    if hidden_state_encoder is not None:
        current_features = _get_or_encode_current_features(
            item["current_images"],
            prompt=str(item.get("prompt", "")),
            cache_key=(benchmark, episode_id, current_step, str(item.get("prompt", ""))),
            hidden_state_encoder=hidden_state_encoder,
            storage_dtype=storage_dtype,
            hidden_state_cache=hidden_state_cache,
        )
        encoded["current_hidden_states"] = current_features.hidden_states
        if current_features.planner_vl_summary is not None:
            encoded["planner_vl_summary"] = torch.as_tensor(
                current_features.planner_vl_summary,
                dtype=storage_dtype,
            ).cpu()
    return encoded


def _get_or_encode_frame_tokens(
    images_by_view: Mapping[str, Image.Image],
    *,
    cache_key: tuple[str, str, int],
    encoder: VisualTokenEncoder,
    storage_dtype: Any,
    visual_token_cache: dict[tuple[str, str, int], dict[str, Any]] | None,
) -> dict[str, Any]:
    if visual_token_cache is not None and cache_key in visual_token_cache:
        return visual_token_cache[cache_key]
    tokens = encode_images_by_view(images_by_view, encoder=encoder, storage_dtype=storage_dtype)
    if visual_token_cache is not None:
        visual_token_cache[cache_key] = tokens
    return tokens


def _get_or_encode_current_features(
    images_by_view: Mapping[str, Image.Image],
    *,
    prompt: str,
    cache_key: tuple[str, str, int, str],
    hidden_state_encoder: VLMHiddenStateEncoder,
    storage_dtype: Any,
    hidden_state_cache: dict[tuple[str, str, int, str], VLMCurrentFeatures] | None,
) -> VLMCurrentFeatures:
    if hidden_state_cache is not None and cache_key in hidden_state_cache:
        return hidden_state_cache[cache_key]
    if hasattr(hidden_state_encoder, "encode_current_features"):
        raw_features = hidden_state_encoder.encode_current_features(images_by_view, prompt)
        raw_hidden_states = raw_features.hidden_states
        raw_planner_vl_summary = raw_features.planner_vl_summary
    else:
        raw_hidden_states = hidden_state_encoder.encode_current(images_by_view, prompt)
        raw_planner_vl_summary = None
    hidden_states = tuple(
        ensure_rank2_tokens(hidden_state, storage_dtype=storage_dtype)
        for hidden_state in raw_hidden_states
    )
    planner_vl_summary = None
    if raw_planner_vl_summary is not None:
        torch = _require_torch()
        planner_vl_summary = torch.as_tensor(raw_planner_vl_summary).detach().cpu().reshape(-1).to(dtype=storage_dtype)
        if planner_vl_summary.numel() <= 0:
            raise ValueError("planner_vl_summary must not be empty")
    features = VLMCurrentFeatures(hidden_states=hidden_states, planner_vl_summary=planner_vl_summary)
    if hidden_state_cache is not None:
        hidden_state_cache[cache_key] = features
    return features


def encode_images_by_view(
    images_by_view: Mapping[str, Image.Image],
    *,
    encoder: VisualTokenEncoder,
    storage_dtype: Any,
) -> dict[str, Any]:
    tokens_by_view = {}
    items = list(images_by_view.items())
    if hasattr(encoder, "encode_images"):
        encoded_tokens = list(encoder.encode_images([image for _view_name, image in items]))
        if len(encoded_tokens) != len(items):
            raise ValueError(
                f"visual encoder returned {len(encoded_tokens)} images for {len(items)} input images"
            )
    else:
        encoded_tokens = [encoder.encode_image(image) for _view_name, image in items]
    for (view_name, _image), tokens in zip(items, encoded_tokens, strict=True):
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
    hidden_state_encoder: VLMHiddenStateEncoder | None = None,
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
        "hidden_state_encoder": None if hidden_state_encoder is None else hidden_state_encoder.name,
        "hidden_state_layers": None
        if hidden_state_encoder is None
        else [_serialize_layer_selector(layer) for layer in hidden_state_encoder.selected_layers],
        "planner_vl_summary": None
        if hidden_state_encoder is None
        else {
            "enabled": bool(hasattr(hidden_state_encoder, "encode_current_features")),
            "source": "vlm_last_valid_token",
            "encoder": hidden_state_encoder.name,
        },
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


def _update_running_minmax(
    current_min: np.ndarray | None,
    current_max: np.ndarray | None,
    values: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty when building token-cache normalization stats")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim >= 2:
        array = array.reshape(-1, array.shape[-1])
    else:
        raise ValueError(f"{name} must have at least one dimension")
    if array.shape[-1] <= 0:
        raise ValueError(f"{name} last dimension must be positive")
    finite = np.isfinite(array)
    if not bool(finite.all()):
        raise ValueError(f"{name} contains non-finite values")

    value_min = array.min(axis=0)
    value_max = array.max(axis=0)
    if current_min is None or current_max is None:
        return value_min, value_max
    if current_min.shape != value_min.shape or current_max.shape != value_max.shape:
        raise ValueError(
            f"{name} dimension changed while building normalization stats: "
            f"{current_min.shape} -> {value_min.shape}"
        )
    return np.minimum(current_min, value_min), np.maximum(current_max, value_max)


def _build_minmax_normalization_manifest(
    *,
    benchmark: str,
    action_min: np.ndarray,
    action_max: np.ndarray,
    state_min: np.ndarray,
    state_max: np.ndarray,
) -> dict[str, Any]:
    robot_key = _normalization_robot_key(benchmark)
    return {
        "enabled": True,
        "type": "train_split_minmax_to_minus_one_one",
        "statistics_from": "cache_build_rows",
        "clip_after_normalization": True,
        "clip_range": [-1.0, 1.0],
        "robot_key": robot_key,
        "stats": {
            robot_key: {
                "observation.state": {
                    "min": np.asarray(state_min, dtype=np.float32).astype(float).tolist(),
                    "max": np.asarray(state_max, dtype=np.float32).astype(float).tolist(),
                },
                "action": {
                    "min": np.asarray(action_min, dtype=np.float32).astype(float).tolist(),
                    "max": np.asarray(action_max, dtype=np.float32).astype(float).tolist(),
                },
            }
        },
    }


def _normalization_robot_key(benchmark: str) -> str:
    return str(benchmark).strip().lower() or "default"


def _parse_token_cache_normalization(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    normalization = manifest.get("normalization")
    if not isinstance(normalization, Mapping) or not bool(normalization.get("enabled", False)):
        return None
    if normalization.get("type") != "train_split_minmax_to_minus_one_one":
        raise ValueError(f"unsupported token-cache normalization type: {normalization.get('type')!r}")
    stats = normalization.get("stats")
    if not isinstance(stats, Mapping) or not stats:
        raise ValueError("token-cache normalization must contain non-empty stats")
    robot_key = str(normalization.get("robot_key") or next(iter(stats)))
    if robot_key not in stats:
        raise KeyError(f"normalization robot_key {robot_key!r} not found in stats")
    robot_stats = stats[robot_key]
    for group_name in ("observation.state", "action"):
        group = robot_stats.get(group_name)
        if not isinstance(group, Mapping) or "min" not in group or "max" not in group:
            raise ValueError(f"normalization stats missing {group_name}.min/max")
    return {
        "type": normalization["type"],
        "robot_key": robot_key,
        "stats": {str(key): value for key, value in stats.items()},
        "clip_after_normalization": bool(normalization.get("clip_after_normalization", True)),
    }


class MemoryTokenCacheDataset:
    """PyTorch-compatible dataset over replay visual-token cache shards."""

    def __init__(self, manifest_path: str | Path, *, max_samples: int | None = None) -> None:
        self.manifest_path = resolve_token_cache_manifest_path(manifest_path)
        self.manifest = read_token_cache_manifest(self.manifest_path)
        _validate_token_cache_manifest(self.manifest, self.manifest_path)
        self.output_root = self.manifest_path.parent
        self.shards = tuple(_resolve_manifest_shards(self.manifest, self.output_root))
        self.shard_end_indices = tuple(shard.end_index for shard in self.shards)
        self.normalization = _parse_token_cache_normalization(self.manifest)
        self.arm2stats_dict = None if self.normalization is None else dict(self.normalization["stats"])

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
        sample = normalize_token_cache_sample(samples[local_index])
        return _apply_token_cache_normalization(sample, self.normalization)

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


class MemoryTokenCacheTrajectoryDataset:
    """Trajectory-window view over a memory-token cache.

    The underlying cache is one row per replan point. This wrapper groups rows
    by episode and returns burn-in + loss windows so recurrent progress memory
    is updated chronologically instead of being reset for random frame batches.
    """

    INDEX_VERSION = 1

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        burnin_replan_steps: int = 8,
        loss_replan_steps: int = 8,
        allow_short_burnin: bool = True,
        action_horizon: int = 32,
        window_stride: int = 1,
        max_samples: int | None = None,
    ) -> None:
        burnin_replan_steps = int(burnin_replan_steps)
        loss_replan_steps = int(loss_replan_steps)
        action_horizon = int(action_horizon)
        window_stride = int(window_stride)
        if burnin_replan_steps < 0:
            raise ValueError("burnin_replan_steps must be non-negative")
        if loss_replan_steps <= 0:
            raise ValueError("loss_replan_steps must be positive")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")

        self.base = MemoryTokenCacheDataset(manifest_path, max_samples=max_samples)
        self.manifest_path = self.base.manifest_path
        self.manifest = self.base.manifest
        self.config = self.base.config
        self.normalization = self.base.normalization
        self.arm2stats_dict = self.base.arm2stats_dict
        self.burnin_replan_steps = burnin_replan_steps
        self.loss_replan_steps = loss_replan_steps
        self.allow_short_burnin = bool(allow_short_burnin)
        self.action_horizon = action_horizon
        self.window_stride = window_stride

        rows = self._load_or_build_trajectory_index()
        if self.base.sample_count < int(self.manifest["sample_count"]):
            rows = [row for row in rows if int(row["sample_index"]) < self.base.sample_count]
        self.windows = tuple(self._build_windows(rows))
        if not self.windows:
            raise ValueError(
                "trajectory token-cache dataset produced no windows; check burnin/loss length and action horizon"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[int(index)]
        return {
            "samples": [self.base[int(sample_index)] for sample_index in window["sample_indices"]],
            "loss_mask": list(window["loss_mask"]),
            "episode_id": str(window["episode_id"]),
            "start_step": int(window["start_step"]),
        }

    def _load_or_build_trajectory_index(self) -> list[dict[str, Any]]:
        index_path = self.manifest_path.parent / "trajectory_index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload.get("rows")
            if (
                payload.get("format") == "memory_token_cache_trajectory_index"
                and int(payload.get("version", -1)) == self.INDEX_VERSION
                and int(payload.get("sample_count", -1)) == int(self.manifest["sample_count"])
                and isinstance(rows, list)
            ):
                return rows

        rows: list[dict[str, Any]] = []
        for shard in self.base.shards:
            payload = _torch_load(shard.path)
            samples = list(payload.get("samples", []))
            if len(samples) != shard.sample_count:
                raise ValueError(f"invalid sample count in shard {shard.path}")
            for local_index, sample in enumerate(samples):
                rows.append(
                    {
                        "sample_index": int(sample.get("sample_index", shard.start_index + local_index)),
                        "benchmark": str(sample["benchmark"]),
                        "episode_id": str(sample["episode_id"]),
                        "current_step": int(sample["current_step"]),
                        "action_valid_count": int(sample["action_valid_count"]),
                    }
                )
        payload = {
            "format": "memory_token_cache_trajectory_index",
            "version": self.INDEX_VERSION,
            "sample_count": int(self.manifest["sample_count"]),
            "rows": rows,
        }
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
        return rows

    def _build_windows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        episodes: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            episodes[(str(row["benchmark"]), str(row["episode_id"]))].append(row)

        windows: list[dict[str, Any]] = []
        for (_benchmark, episode_id), episode_rows in episodes.items():
            ordered = sorted(episode_rows, key=lambda row: (int(row["current_step"]), int(row["sample_index"])))
            valid_loss_positions = {
                pos for pos, row in enumerate(ordered) if int(row["action_valid_count"]) >= self.action_horizon
            }
            max_start = len(ordered) - self.loss_replan_steps
            for start_pos in range(0, max_start + 1, self.window_stride):
                loss_positions = range(start_pos, start_pos + self.loss_replan_steps)
                if any(pos not in valid_loss_positions for pos in loss_positions):
                    continue
                if not self.allow_short_burnin and start_pos < self.burnin_replan_steps:
                    continue
                burnin_start = max(0, start_pos - self.burnin_replan_steps)
                selected = ordered[burnin_start : start_pos + self.loss_replan_steps]
                burnin_count = start_pos - burnin_start
                windows.append(
                    {
                        "episode_id": episode_id,
                        "start_step": int(ordered[start_pos]["current_step"]),
                        "sample_indices": [int(row["sample_index"]) for row in selected],
                        "loss_mask": [False] * burnin_count + [True] * self.loss_replan_steps,
                    }
                )
        return windows


class EpisodeFeatureCacheTrajectoryDataset:
    """Episode-level trajectory view over processed LIBERO feature cache shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        action_horizon: int = 32,
        max_episodes: int | None = None,
    ) -> None:
        self.manifest_path = resolve_token_cache_manifest_path(manifest_path)
        self.manifest = read_token_cache_manifest(self.manifest_path)
        _validate_episode_feature_cache_manifest(self.manifest, self.manifest_path)
        self.output_root = self.manifest_path.parent
        self.shards = tuple(_resolve_episode_feature_shards(self.manifest, self.output_root))
        self.shard_end_indices = tuple(shard.end_index for shard in self.shards)
        self.normalization = _parse_token_cache_normalization(self.manifest)
        self.arm2stats_dict = None if self.normalization is None else dict(self.normalization["stats"])
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        episode_count = int(self.manifest["episode_count"])
        if max_episodes is not None:
            if int(max_episodes) <= 0:
                raise ValueError("max_episodes must be positive when provided")
            episode_count = min(episode_count, int(max_episodes))
        self.episode_count = episode_count
        self.config = TokenCacheDatasetConfig(
            manifest_path=self.manifest_path,
            output_root=self.output_root,
            benchmark=str(self.manifest["benchmark"]),
            sample_count=self.episode_count,
            hidden_dim=int(self.manifest["hidden_dim"]),
            storage_dtype=str(self.manifest["storage_dtype"]),
        )
        self._loaded_shard_index: int | None = None
        self._loaded_shard_episodes: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return self.episode_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += self.episode_count
        if index < 0 or index >= self.episode_count:
            raise IndexError(index)
        shard_index = bisect_right(self.shard_end_indices, index)
        if shard_index >= len(self.shards):
            raise IndexError(index)
        shard = self.shards[shard_index]
        episodes = self._load_shard_episodes(shard_index)
        local_index = index - shard.start_index
        episode = episodes[local_index]
        samples = [
            self._node_to_sample(episode, node, sample_index=index * 100000 + node_index)
            for node_index, node in enumerate(episode["nodes"])
        ]
        loss_mask = [
            int(node["action_valid_count"]) >= self.action_horizon
            for node in episode["nodes"]
        ]
        if not any(loss_mask):
            raise ValueError(f"episode {episode.get('episode_id')} has no full-horizon Stage1 loss nodes")
        return {
            "samples": samples,
            "loss_mask": loss_mask,
            "episode_id": str(episode["episode_id"]),
            "start_step": int(episode["nodes"][0]["current_step"]),
        }

    def _load_shard_episodes(self, shard_index: int) -> list[dict[str, Any]]:
        if self._loaded_shard_index == shard_index and self._loaded_shard_episodes is not None:
            return self._loaded_shard_episodes
        shard = self.shards[shard_index]
        payload = _torch_load(shard.path)
        if payload.get("format") != EPISODE_FEATURE_CACHE_FORMAT:
            raise ValueError(f"invalid episode feature cache shard format in {shard.path}")
        episodes = list(payload.get("episodes", []))
        if len(episodes) != shard.sample_count:
            raise ValueError(
                f"shard {shard.path} manifest episode_count={shard.sample_count} "
                f"but file has {len(episodes)} episodes"
            )
        self._loaded_shard_index = shard_index
        self._loaded_shard_episodes = episodes
        return episodes

    def _node_to_sample(
        self,
        episode: Mapping[str, Any],
        node: Mapping[str, Any],
        *,
        sample_index: int,
    ) -> dict[str, Any]:
        torch = _require_torch()
        current_step = int(node["current_step"])
        actions = torch.as_tensor(episode["actions"], dtype=torch.float32).cpu()
        visual_tokens_by_step = episode["visual_tokens_by_step"]
        state_by_step = episode["state_by_step"]
        current_features_by_step = episode["current_features_by_step"]

        short_steps = [None if step is None else int(step) for step in node.get("short_visual_steps", [])]
        short_mask = [bool(value) for value in node.get("short_mask", [])]
        short_tokens = tuple(
            None
            if step is None or index >= len(short_mask) or not short_mask[index]
            else _mapping_get_step(visual_tokens_by_step, int(step), label="visual_tokens_by_step")
            for index, step in enumerate(short_steps)
        )
        executed_start, executed_end = [int(value) for value in node["executed_action_range"]]
        executed_actions, executed_action_mask = _pad_executed_actions(
            actions[executed_start:executed_end],
            valid_count=int(node["executed_action_valid_count"]),
            action_dim=int(actions.shape[-1]),
            target_length=max(1, executed_end - executed_start, int(self.manifest["source_executed_action_stride"])),
        )
        future_start, future_end = [int(value) for value in node["future_action_range"]]
        features = _mapping_get_step(current_features_by_step, current_step, label="current_features_by_step")
        sample = {
            "sample_index": int(sample_index),
            "benchmark": str(self.manifest.get("benchmark", "LIBERO")),
            "episode_id": str(episode["episode_id"]),
            "prompt": str(episode.get("prompt", "")),
            "current_step": current_step,
            "current_tokens_by_view": _mapping_get_step(
                visual_tokens_by_step,
                current_step,
                label="visual_tokens_by_step",
            ),
            "current_state": _mapping_get_step(state_by_step, current_step, label="state_by_step"),
            "short_tokens_by_view": short_tokens,
            "short_steps": [-1 if step is None else int(step) for step in short_steps],
            "short_mask": short_mask,
            "executed_actions": executed_actions,
            "executed_action_mask": executed_action_mask,
            "future_actions": actions[future_start:future_end].contiguous(),
            "action_valid_count": int(node["action_valid_count"]),
            "current_hidden_states": tuple(features["hidden_states"]),
            "planner_vl_summary": features["planner_vl_summary"],
        }
        return _apply_token_cache_normalization(normalize_token_cache_sample(sample), self.normalization)


def collate_direct_bridge_token_cache_windows(
    batch: Sequence[Mapping[str, Any]],
    *,
    memory_entry_tokens: int = 16,
    action_horizon: int | None = None,
    view_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collate episode/node sequences into per-timestep active mini-batches."""

    if not batch:
        raise ValueError("batch must contain at least one episode sequence")
    torch = _require_torch()
    max_length = max(len(window["samples"]) for window in batch)
    steps = []
    for step_index in range(max_length):
        active_samples = []
        batch_indices = []
        loss_mask = []
        for batch_index, window in enumerate(batch):
            samples = list(window["samples"])
            if step_index >= len(samples):
                continue
            active_samples.append(samples[step_index])
            batch_indices.append(batch_index)
            loss_mask.append(bool(window["loss_mask"][step_index]))
        if not active_samples:
            continue
        step_batch = collate_direct_bridge_token_cache_samples(
            active_samples,
            memory_entry_tokens=memory_entry_tokens,
            action_horizon=action_horizon,
            view_names=view_names,
        )
        step_batch["batch_indices"] = torch.tensor(batch_indices, dtype=torch.long)
        step_batch["loss_mask"] = torch.tensor(loss_mask, dtype=torch.bool)
        steps.append(step_batch)
    if not steps:
        raise ValueError("episode sequence batch contains no active steps")
    return {
        "trajectory_steps": steps,
        "batch_size": len(batch),
        "episode_id": [str(window["episode_id"]) for window in batch],
        "start_step": torch.tensor([int(window["start_step"]) for window in batch], dtype=torch.long),
    }


def collate_memory_token_cache_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must contain at least one item")
    torch = _require_torch()
    future_actions, action_mask = _pad_future_actions([sample["future_actions"] for sample in batch])
    output = {
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
        "future_actions": future_actions,
        "action_mask": action_mask,
        "action_valid_count": torch.tensor([int(sample["action_valid_count"]) for sample in batch], dtype=torch.long),
    }
    executed_actions = _stack_optional_rank2(batch, "executed_actions", dtype=torch.float32)
    executed_action_mask = _stack_optional_rank1(batch, "executed_action_mask", dtype=torch.bool)
    if executed_actions is not None:
        output["executed_actions"] = executed_actions
    if executed_action_mask is not None:
        output["executed_action_mask"] = executed_action_mask
    hidden_states = _stack_optional_hidden_states(batch, "current_hidden_states")
    if hidden_states is not None:
        output["vlm_hidden_states"] = hidden_states
    planner_vl_summary = _stack_optional_rank1(batch, "planner_vl_summary", dtype=torch.float32)
    if planner_vl_summary is not None:
        output["planner_vl_summary"] = planner_vl_summary
    return output


def collate_direct_bridge_token_cache_samples(
    batch: Sequence[Mapping[str, Any]],
    *,
    memory_entry_tokens: int = 16,
    action_horizon: int | None = None,
    view_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collate visual-token cache rows into the direct bridge-attn training contract.

    The cache stores raw visual tokens by view. This collate function builds:

    - ``fused_tokens`` from the current frame visual tokens.
    - ``memory_context`` from fixed-size short-memory entries.
    - ``short_memory_time_ids`` with one id per short-memory entry.
    - per-dimension ``action_mask`` matching ``future_actions``.
    """

    if not batch:
        raise ValueError("batch must contain at least one item")
    memory_entry_tokens = int(memory_entry_tokens)
    if memory_entry_tokens <= 0:
        raise ValueError("memory_entry_tokens must be positive")

    torch = _require_torch()
    current_tokens = []
    memory_context = []
    memory_context_mask = []
    short_time_ids = []

    for sample in batch:
        current = concat_tokens_by_view(sample["current_tokens_by_view"], view_names=view_names)
        current_tokens.append(current)

        sample_short_tokens = []
        sample_short_mask = []
        sample_time_ids = []
        short_entries = tuple(sample.get("short_tokens_by_view", ()))
        short_valid = torch.as_tensor(sample.get("short_mask", [entry is not None for entry in short_entries])).bool()
        if short_valid.numel() != len(short_entries):
            raise ValueError("short_mask length must match short_tokens_by_view length")

        for entry_index, entry in enumerate(short_entries):
            if entry is None or not bool(short_valid[entry_index].item()):
                packed = torch.zeros(memory_entry_tokens, current.shape[-1], dtype=current.dtype)
                packed_mask = torch.zeros(memory_entry_tokens, dtype=torch.bool)
            else:
                packed, packed_mask = pack_visual_tokens(
                    concat_tokens_by_view(entry, view_names=view_names),
                    target_tokens=memory_entry_tokens,
                )
            sample_short_tokens.append(packed)
            sample_short_mask.append(packed_mask)
            sample_time_ids.append(torch.full((memory_entry_tokens,), entry_index, dtype=torch.long))

        if sample_short_tokens:
            memory_context.append(torch.cat(sample_short_tokens, dim=0))
            memory_context_mask.append(torch.cat(sample_short_mask, dim=0))
            short_time_ids.append(torch.cat(sample_time_ids, dim=0))
        else:
            memory_context.append(torch.zeros(0, current.shape[-1], dtype=current.dtype))
            memory_context_mask.append(torch.zeros(0, dtype=torch.bool))
            short_time_ids.append(torch.zeros(0, dtype=torch.long))

    fused_tokens = _pad_token_sequences(current_tokens)
    future_actions, step_mask = _pad_future_actions([sample["future_actions"] for sample in batch])
    if action_horizon is not None:
        future_actions, step_mask = _resize_action_horizon(
            future_actions,
            step_mask,
            action_horizon=int(action_horizon),
        )
    action_mask = step_mask.unsqueeze(-1).expand_as(future_actions).clone()
    executed_actions = _stack_optional_rank2(batch, "executed_actions", dtype=torch.float32)
    executed_action_mask = _stack_optional_rank1(batch, "executed_action_mask", dtype=torch.bool)

    output = {
        "benchmark": [str(sample["benchmark"]) for sample in batch],
        "episode_id": [str(sample["episode_id"]) for sample in batch],
        "sample_index": torch.tensor([int(sample["sample_index"]) for sample in batch], dtype=torch.long),
        "current_step": torch.tensor([int(sample["current_step"]) for sample in batch], dtype=torch.long),
        "fused_tokens": fused_tokens,
        "states": torch.stack([torch.as_tensor(sample["current_state"], dtype=torch.float32) for sample in batch]),
        "actions": future_actions,
        "action_mask": action_mask,
        "memory_context": _pad_token_sequences(memory_context),
        "memory_context_mask": _pad_bool_sequences(memory_context_mask),
        "short_memory_time_ids": _pad_long_sequences(short_time_ids),
        "action_valid_count": torch.tensor([int(sample["action_valid_count"]) for sample in batch], dtype=torch.long),
    }
    if executed_actions is not None:
        output["executed_actions"] = executed_actions
    if executed_action_mask is not None:
        output["executed_action_mask"] = executed_action_mask
    hidden_states = _stack_optional_hidden_states(batch, "current_hidden_states")
    if hidden_states is not None:
        output["vlm_hidden_states"] = hidden_states
    planner_vl_summary = _stack_optional_rank1(batch, "planner_vl_summary", dtype=torch.float32)
    if planner_vl_summary is not None:
        output["planner_vl_summary"] = planner_vl_summary
    return output


def concat_tokens_by_view(tokens_by_view: Mapping[str, Any], *, view_names: Sequence[str] | None = None) -> Any:
    torch = _require_torch()
    if view_names is None:
        view_names = sorted(str(name) for name in tokens_by_view)
    tensors = []
    for view_name in view_names:
        if view_name not in tokens_by_view:
            raise KeyError(f"missing visual tokens for view {view_name!r}")
        tensors.append(flatten_visual_tokens(torch.as_tensor(tokens_by_view[view_name], dtype=torch.float32).cpu()))
    if not tensors:
        raise ValueError("tokens_by_view must contain at least one view")
    hidden_dim = int(tensors[0].shape[-1])
    if any(int(tensor.shape[-1]) != hidden_dim for tensor in tensors):
        raise ValueError("all view token tensors must share hidden dim")
    return torch.cat(tensors, dim=0)


def pack_visual_tokens(tokens: Any, *, target_tokens: int) -> tuple[Any, Any]:
    """Convert an arbitrary token sequence into a fixed-size token entry.

    Long sequences are reduced with deterministic mean pooling over equally
    spaced bins. Short sequences are zero-padded and expose a validity mask.
    """

    torch = _require_torch()
    tokens = flatten_visual_tokens(torch.as_tensor(tokens, dtype=torch.float32).cpu())
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    token_count, hidden_dim = int(tokens.shape[0]), int(tokens.shape[1])
    if token_count <= 0:
        raise ValueError("visual token sequence must be non-empty")

    if token_count == target_tokens:
        return tokens.contiguous(), torch.ones(target_tokens, dtype=torch.bool)
    if token_count < target_tokens:
        output = torch.zeros(target_tokens, hidden_dim, dtype=tokens.dtype)
        output[:token_count] = tokens
        mask = torch.zeros(target_tokens, dtype=torch.bool)
        mask[:token_count] = True
        return output, mask

    boundaries = torch.linspace(0, token_count, steps=target_tokens + 1).round().long()
    output = torch.zeros(target_tokens, hidden_dim, dtype=tokens.dtype)
    for index in range(target_tokens):
        start = int(boundaries[index].item())
        end = int(boundaries[index + 1].item())
        if end <= start:
            end = min(token_count, start + 1)
        output[index] = tokens[start:end].mean(dim=0)
    return output, torch.ones(target_tokens, dtype=torch.bool)


def normalize_token_cache_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    torch = _require_torch()
    normalized = dict(sample)
    short_tokens = tuple(normalized.get("short_tokens_by_view", ()))
    normalized["short_tokens_by_view"] = short_tokens
    normalized["short_mask"] = torch.as_tensor(normalized["short_mask"], dtype=torch.bool).cpu()
    normalized["short_steps"] = _short_steps_tensor(normalized, len(short_tokens))
    normalized["current_state"] = torch.as_tensor(normalized["current_state"], dtype=torch.float32).cpu()
    if "current_hidden_states" in normalized:
        normalized["current_hidden_states"] = tuple(
            flatten_visual_tokens(torch.as_tensor(hidden_state, dtype=torch.float32).cpu())
            for hidden_state in normalized["current_hidden_states"]
        )
    if "planner_vl_summary" in normalized:
        normalized["planner_vl_summary"] = torch.as_tensor(
            normalized["planner_vl_summary"],
            dtype=torch.float32,
        ).reshape(-1).cpu()
    normalized["future_actions"] = torch.as_tensor(normalized["future_actions"], dtype=torch.float32).cpu()
    if "executed_actions" in normalized:
        normalized["executed_actions"] = torch.as_tensor(normalized["executed_actions"], dtype=torch.float32).cpu()
    if "executed_action_mask" in normalized:
        normalized["executed_action_mask"] = torch.as_tensor(normalized["executed_action_mask"], dtype=torch.bool).cpu()
    normalized["action_valid_count"] = int(normalized["action_valid_count"])
    normalized["current_step"] = int(normalized["current_step"])
    normalized["sample_index"] = int(normalized.get("sample_index", -1))
    normalized["benchmark"] = str(normalized["benchmark"])
    normalized["episode_id"] = str(normalized["episode_id"])
    normalized["prompt"] = str(normalized.get("prompt", ""))
    return normalized


def _apply_token_cache_normalization(
    sample: dict[str, Any],
    normalization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if normalization is None:
        return sample

    torch = _require_torch()
    robot_key = str(normalization["robot_key"])
    stats = normalization["stats"][robot_key]
    clip = bool(normalization.get("clip_after_normalization", True))

    state_min = torch.as_tensor(stats["observation.state"]["min"], dtype=torch.float32)
    state_max = torch.as_tensor(stats["observation.state"]["max"], dtype=torch.float32)
    action_min = torch.as_tensor(stats["action"]["min"], dtype=torch.float32)
    action_max = torch.as_tensor(stats["action"]["max"], dtype=torch.float32)

    sample["current_state"] = _minmax_normalize_tensor(
        torch.as_tensor(sample["current_state"], dtype=torch.float32).cpu(),
        state_min,
        state_max,
        clip=clip,
        name="current_state",
    )
    sample["future_actions"] = _minmax_normalize_tensor(
        torch.as_tensor(sample["future_actions"], dtype=torch.float32).cpu(),
        action_min,
        action_max,
        clip=clip,
        name="future_actions",
    )
    if "executed_actions" in sample:
        executed = _minmax_normalize_tensor(
            torch.as_tensor(sample["executed_actions"], dtype=torch.float32).cpu(),
            action_min,
            action_max,
            clip=clip,
            name="executed_actions",
        )
        if "executed_action_mask" in sample:
            mask = torch.as_tensor(sample["executed_action_mask"], dtype=torch.bool).cpu()
            if mask.shape != executed.shape[:1]:
                raise ValueError(f"executed_action_mask shape {tuple(mask.shape)} does not match executed_actions")
            executed = executed * mask.unsqueeze(-1).to(dtype=executed.dtype)
        sample["executed_actions"] = executed
    return sample


def _minmax_normalize_tensor(
    value: Any,
    min_value: Any,
    max_value: Any,
    *,
    clip: bool,
    name: str,
) -> Any:
    torch = _require_torch()
    tensor = torch.as_tensor(value, dtype=torch.float32).cpu()
    min_tensor = torch.as_tensor(min_value, dtype=torch.float32).cpu()
    max_tensor = torch.as_tensor(max_value, dtype=torch.float32).cpu()
    dim = int(tensor.shape[-1]) if tensor.ndim > 0 else 1
    if dim > int(min_tensor.shape[0]) or dim > int(max_tensor.shape[0]):
        raise ValueError(
            f"{name} dim {dim} exceeds normalization stats dims "
            f"{tuple(min_tensor.shape)}, {tuple(max_tensor.shape)}"
        )
    min_tensor = min_tensor[:dim].to(dtype=tensor.dtype)
    max_tensor = max_tensor[:dim].to(dtype=tensor.dtype)
    normalized = 2.0 * (tensor - min_tensor) / (max_tensor - min_tensor + 1e-8) - 1.0
    if clip:
        normalized = torch.clamp(normalized, -1.0, 1.0)
    return normalized.contiguous()


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


def _serialize_layer_selector(layer: int | str) -> int | str:
    if isinstance(layer, (int, np.integer)):
        return int(layer)
    text = str(layer)
    try:
        return int(text)
    except ValueError:
        return text


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


def _validate_episode_feature_cache_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    if manifest.get("format") != EPISODE_FEATURE_CACHE_FORMAT:
        raise ValueError(f"invalid episode feature cache format in {manifest_path}: {manifest.get('format')!r}")
    if int(manifest.get("version", -1)) != EPISODE_FEATURE_CACHE_VERSION:
        raise ValueError(f"unsupported episode feature cache version in {manifest_path}: {manifest.get('version')!r}")
    if not str(manifest.get("benchmark", "")).strip():
        raise ValueError(f"episode feature cache manifest must include benchmark: {manifest_path}")
    if int(manifest.get("episode_count", -1)) < 0:
        raise ValueError(f"episode feature cache manifest has invalid episode_count: {manifest.get('episode_count')!r}")
    if int(manifest.get("node_count", -1)) < 0:
        raise ValueError(f"episode feature cache manifest has invalid node_count: {manifest.get('node_count')!r}")
    if int(manifest.get("hidden_dim", 0)) <= 0:
        raise ValueError(f"episode feature cache manifest has invalid hidden_dim: {manifest.get('hidden_dim')!r}")
    if int(manifest.get("source_executed_action_stride", 0)) <= 0:
        raise ValueError("episode feature cache manifest must include source_executed_action_stride")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"episode feature cache manifest has no shards: {manifest_path}")


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


def _resolve_episode_feature_shards(manifest: Mapping[str, Any], output_root: Path) -> list[TokenCacheShard]:
    shards: list[TokenCacheShard] = []
    expected_start = 0
    for raw_shard in manifest["shards"]:
        shard = TokenCacheShard(
            path=output_root / str(raw_shard["path"]),
            sample_count=int(raw_shard["episode_count"]),
            start_index=int(raw_shard["start_index"]),
            end_index=int(raw_shard["end_index"]),
        )
        if shard.start_index != expected_start:
            raise ValueError(
                f"non-contiguous episode feature shard starts at {shard.start_index}, expected {expected_start}"
            )
        if shard.end_index - shard.start_index != shard.sample_count:
            raise ValueError(f"episode feature shard has inconsistent index range: {shard}")
        if not shard.path.exists():
            raise FileNotFoundError(f"episode feature shard does not exist: {shard.path}")
        shards.append(shard)
        expected_start = shard.end_index
    if expected_start != int(manifest["episode_count"]):
        raise ValueError(f"manifest episode_count={manifest['episode_count']} but shards end at {expected_start}")
    return shards


def _mapping_get_step(mapping: Mapping[Any, Any], step: int, *, label: str) -> Any:
    if step in mapping:
        return mapping[step]
    text_step = str(step)
    if text_step in mapping:
        return mapping[text_step]
    raise KeyError(f"{label} missing step {step}")


def _pad_executed_actions(
    actions: Any,
    *,
    valid_count: int,
    action_dim: int,
    target_length: int,
) -> tuple[Any, Any]:
    torch = _require_torch()
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    action_dim = int(action_dim)
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    tensor = torch.as_tensor(actions, dtype=torch.float32).reshape(-1, action_dim).cpu()
    valid = min(max(int(valid_count), 0), int(tensor.shape[0]), target_length)
    output = torch.zeros(target_length, action_dim, dtype=torch.float32)
    mask = torch.zeros(target_length, dtype=torch.bool)
    if valid > 0:
        output[-valid:] = tensor[-valid:]
        mask[-valid:] = True
    return output, mask


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


def _stack_optional_rank2(batch: Sequence[Mapping[str, Any]], key: str, *, dtype: Any) -> Any | None:
    if not all(key in sample for sample in batch):
        return None
    torch = _require_torch()
    tensors = [torch.as_tensor(sample[key], dtype=dtype).cpu() for sample in batch]
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError(f"{key} must have shape [T, D]")
    expected_shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != expected_shape for tensor in tensors):
        raise ValueError(f"all {key} tensors in a batch must share shape")
    return torch.stack(tensors, dim=0)


def _stack_optional_rank1(batch: Sequence[Mapping[str, Any]], key: str, *, dtype: Any) -> Any | None:
    if not all(key in sample for sample in batch):
        return None
    torch = _require_torch()
    tensors = [torch.as_tensor(sample[key], dtype=dtype).reshape(-1).cpu() for sample in batch]
    expected_shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != expected_shape for tensor in tensors):
        raise ValueError(f"all {key} tensors in a batch must share shape")
    return torch.stack(tensors, dim=0)


def _stack_optional_hidden_states(batch: Sequence[Mapping[str, Any]], key: str) -> list[Any] | None:
    if not all(key in sample for sample in batch):
        return None
    torch = _require_torch()
    per_sample = [tuple(sample[key]) for sample in batch]
    layer_count = len(per_sample[0])
    if layer_count <= 0:
        raise ValueError(f"{key} must contain at least one hidden-state layer")
    if any(len(sample_layers) != layer_count for sample_layers in per_sample):
        raise ValueError(f"all samples must provide the same number of {key} layers")
    hidden_states = []
    for layer_index in range(layer_count):
        hidden_states.append(
            _pad_token_sequences(
                [
                    flatten_visual_tokens(torch.as_tensor(sample_layers[layer_index], dtype=torch.float32).cpu())
                    for sample_layers in per_sample
                ]
            )
        )
    return hidden_states


def _resize_action_horizon(actions: Any, step_mask: Any, *, action_horizon: int) -> tuple[Any, Any]:
    torch = _require_torch()
    action_horizon = int(action_horizon)
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if actions.shape[1] == action_horizon:
        return actions, step_mask
    if actions.shape[1] > action_horizon:
        return actions[:, :action_horizon].contiguous(), step_mask[:, :action_horizon].contiguous()
    padded_actions = torch.zeros(actions.shape[0], action_horizon, actions.shape[2], dtype=actions.dtype)
    padded_mask = torch.zeros(step_mask.shape[0], action_horizon, dtype=step_mask.dtype)
    padded_actions[:, : actions.shape[1]] = actions
    padded_mask[:, : step_mask.shape[1]] = step_mask
    return padded_actions, padded_mask


def _pad_token_sequences(sequences: Sequence[Any]) -> Any:
    torch = _require_torch()
    tensors = [flatten_visual_tokens(torch.as_tensor(sequence, dtype=torch.float32).cpu()) for sequence in sequences]
    if not tensors:
        raise ValueError("sequences must not be empty")
    hidden_dim = int(tensors[0].shape[-1])
    if any(int(tensor.shape[-1]) != hidden_dim for tensor in tensors):
        raise ValueError("all token sequences in a batch must share hidden dim")
    max_tokens = max(int(tensor.shape[0]) for tensor in tensors)
    output = torch.zeros(len(tensors), max_tokens, hidden_dim, dtype=torch.float32)
    for index, tensor in enumerate(tensors):
        output[index, : tensor.shape[0]] = tensor
    return output


def _pad_bool_sequences(sequences: Sequence[Any]) -> Any:
    torch = _require_torch()
    tensors = [torch.as_tensor(sequence, dtype=torch.bool).reshape(-1).cpu() for sequence in sequences]
    if not tensors:
        raise ValueError("sequences must not be empty")
    max_tokens = max(int(tensor.numel()) for tensor in tensors)
    output = torch.zeros(len(tensors), max_tokens, dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        output[index, : tensor.numel()] = tensor
    return output


def _pad_long_sequences(sequences: Sequence[Any]) -> Any:
    torch = _require_torch()
    tensors = [torch.as_tensor(sequence, dtype=torch.long).reshape(-1).cpu() for sequence in sequences]
    if not tensors:
        raise ValueError("sequences must not be empty")
    max_tokens = max(int(tensor.numel()) for tensor in tensors)
    output = torch.zeros(len(tensors), max_tokens, dtype=torch.long)
    for index, tensor in enumerate(tensors):
        output[index, : tensor.numel()] = tensor
    return output


def _torch_load(path: str | Path) -> Any:
    torch = _require_torch()
    try:
        return torch.load(path, weights_only=True)
    except (TypeError, pickle.UnpicklingError, RuntimeError):
        return torch.load(path, weights_only=False)
