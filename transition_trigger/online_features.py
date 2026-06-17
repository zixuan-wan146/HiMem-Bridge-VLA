from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


FrameMapping = Mapping[str, Any]


@dataclass(frozen=True)
class OnlineFeatureSpec:
    dataset_name: str | None
    window_size: int
    input_dim: int
    block_names: tuple[str, ...]


class CanonicalFeatureBuilder:
    """Build runtime features with the same layout used during training."""

    def __init__(self, config: Mapping[str, Any], *, dataset_name: str | None = None) -> None:
        self.config = dict(config)
        self.feature_config = resolve_online_feature_config(self.config, dataset_name=dataset_name)
        self.dataset_name = self.feature_config.get("dataset_name")
        self.window_size = int(self.config.get("data", {}).get("window_size", 32))
        if self.window_size <= 0:
            raise ValueError("data.window_size must be positive")
        if str(self.feature_config.get("mode", "flat")) != "canonical_blocks":
            raise ValueError("online runtime feature building requires features.mode='canonical_blocks'")
        self.blocks = [dict(block) for block in self.feature_config.get("blocks", [])]
        if not self.blocks:
            raise ValueError("features.blocks must be non-empty for online feature building")
        self.input_dim = self._resolve_input_dim()

    @property
    def spec(self) -> OnlineFeatureSpec:
        return OnlineFeatureSpec(
            dataset_name=None if self.dataset_name is None else str(self.dataset_name),
            window_size=self.window_size,
            input_dim=self.input_dim,
            block_names=tuple(str(block.get("name", index)) for index, block in enumerate(self.blocks)),
        )

    def build_window(self, frames: Sequence[FrameMapping]) -> np.ndarray:
        if not frames:
            raise ValueError("frames must be non-empty")
        values = np.stack([self._build_value_row(frame) for frame in frames]).astype(np.float32)
        parts = [values]
        if bool(self.feature_config.get("include_deltas", True)):
            parts.append(_delta(values))
        if bool(self.feature_config.get("include_value_mask", True)):
            masks = [_block_mask(len(frames), block) for block in self.blocks]
            parts.append(np.concatenate(masks, axis=1).astype(np.float32))
        source = self.feature_config.get("source_one_hot") or {}
        if bool(source.get("enabled", False)):
            names = [str(name) for name in source.get("names", [])]
            dataset_name = str(self.dataset_name or "")
            if dataset_name not in names:
                raise ValueError(f"dataset_name {dataset_name!r} is not listed in features.source_one_hot.names")
            one_hot = np.zeros((len(frames), len(names)), dtype=np.float32)
            one_hot[:, names.index(dataset_name)] = 1.0
            parts.append(one_hot)
        features = np.concatenate(parts, axis=1).astype(np.float32)
        if features.shape[1] != self.input_dim:
            raise ValueError(f"built input_dim={features.shape[1]} does not match expected input_dim={self.input_dim}")
        return features

    def _resolve_input_dim(self) -> int:
        value_dim = sum(int(block["dim"]) for block in self.blocks)
        input_dim = value_dim
        if bool(self.feature_config.get("include_deltas", True)):
            input_dim += value_dim
        if bool(self.feature_config.get("include_value_mask", True)):
            input_dim += value_dim
        source = self.feature_config.get("source_one_hot") or {}
        if bool(source.get("enabled", False)):
            input_dim += len(source.get("names", []))
        expected = self.feature_config.get("expected_input_dim")
        if expected is not None and int(expected) != input_dim:
            raise ValueError(f"features.expected_input_dim={expected} does not match online input_dim={input_dim}")
        return input_dim

    def _build_value_row(self, frame: FrameMapping) -> np.ndarray:
        values = [self._extract_block(frame, block) for block in self.blocks]
        return np.concatenate(values).astype(np.float32)

    def _extract_block(self, frame: FrameMapping, block: Mapping[str, Any]) -> np.ndarray:
        dim = int(block["dim"])
        if dim <= 0:
            raise ValueError("feature block dim must be positive")
        if "constant" in block:
            return _fit_dim(np.asarray(block["constant"], dtype=np.float32).reshape(-1), dim)
        sources = block.get("sources")
        if sources is None:
            if "key" not in block:
                return np.zeros(dim, dtype=np.float32)
            sources = [{"key": block["key"], "start": block.get("start", 0), "length": block.get("length")}]
        arrays = [self._extract_source(frame, source) for source in sources]
        if not arrays:
            return np.zeros(dim, dtype=np.float32)
        return _fit_dim(np.concatenate(arrays), dim)

    @staticmethod
    def _extract_source(frame: FrameMapping, source: Mapping[str, Any]) -> np.ndarray:
        key = str(source["key"])
        if key not in frame:
            raise KeyError(f"feature source key not found in online frame: {key}")
        values = np.asarray(frame[key], dtype=np.float32).reshape(-1)
        start = int(source.get("start", 0))
        length = source.get("length")
        if length is None:
            return values[start:]
        return values[start : start + int(length)]


class OnlineTransitionFeatureBuffer:
    """Causal fixed-window feature buffer for transition trigger deployment."""

    def __init__(self, config: Mapping[str, Any], *, dataset_name: str | None = None) -> None:
        self.builder = CanonicalFeatureBuilder(config, dataset_name=dataset_name)
        self._frames: deque[FrameMapping] = deque(maxlen=self.builder.window_size)

    @property
    def spec(self) -> OnlineFeatureSpec:
        return self.builder.spec

    @property
    def ready(self) -> bool:
        return len(self._frames) == self.builder.window_size

    def reset(self) -> None:
        self._frames.clear()

    def append(self, frame: FrameMapping) -> None:
        self._frames.append(dict(frame))

    def append_and_build(self, frame: FrameMapping) -> torch.Tensor | None:
        self.append(frame)
        if not self.ready:
            return None
        return self.window_tensor()

    def window_features(self) -> np.ndarray:
        if not self.ready:
            raise ValueError(f"feature buffer needs {self.builder.window_size} frames, has {len(self._frames)}")
        return self.builder.build_window(list(self._frames))

    def window_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.window_features())


def resolve_online_feature_config(config: Mapping[str, Any], *, dataset_name: str | None = None) -> dict[str, Any]:
    base = dict(config.get("features", {}))
    datasets = list(config.get("data", {}).get("datasets") or [])
    if not datasets:
        if dataset_name is not None:
            base["dataset_name"] = str(dataset_name)
        return base

    if dataset_name is None:
        if len(datasets) != 1:
            names = [str(item.get("name", "")) for item in datasets]
            raise ValueError(f"dataset_name is required when config contains multiple datasets: {names}")
        dataset = datasets[0]
    else:
        dataset = None
        for item in datasets:
            if str(item.get("name")) == str(dataset_name):
                dataset = item
                break
        if dataset is None:
            names = [str(item.get("name", "")) for item in datasets]
            raise ValueError(f"unknown dataset_name={dataset_name!r}; available datasets={names}")

    merged = _merge_feature_config(base, dataset.get("features", {}))
    merged["dataset_name"] = str(dataset.get("name", dataset_name or ""))
    return merged


def _merge_feature_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            child = dict(merged[key])
            child.update(value)
            merged[key] = child
        else:
            merged[key] = value
    return merged


def _fit_dim(values: np.ndarray, dim: int) -> np.ndarray:
    values = values.astype(np.float32).reshape(-1)
    if values.shape[0] == dim:
        return values
    if values.shape[0] > dim:
        return values[:dim]
    padded = np.zeros(dim, dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded


def _block_mask(length: int, block: Mapping[str, Any]) -> np.ndarray:
    dim = int(block["dim"])
    valid = float(block.get("valid", 1.0 if ("key" in block or "sources" in block or "constant" in block) else 0.0))
    return np.full((length, dim), valid, dtype=np.float32)


def _delta(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    out[1:] = values[1:] - values[:-1]
    return out
