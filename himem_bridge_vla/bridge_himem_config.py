from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


BRIDGE_VARIANTS = {"crosskv", "mixed_latent"}
CONTEXT_MODES = {"fused_only", "bridge_clean", "bridge_residual", "bridge_gated_residual"}
MEMORY_KINDS = {"dual_fifo_visual"}
ACTION_QUERY_SOURCES = {"learned_bridge"}
COARSE_PLANNER_TYPES = {"query_cross_attention"}
COARSE_PLANNER_LOSSES = {"intent_latent"}
COARSE_PLANNER_PLACEMENTS = {"bridge_crosskv"}


@dataclass(frozen=True)
class VLMConfig:
    hidden_dim: int = 896
    raw_dim: int | None = None
    raw_layers: tuple[int | str, ...] = (3, 7, 11, 14)
    freeze: bool = True
    allow_image_token_truncation: bool = False


@dataclass(frozen=True)
class ActionQueryConfig:
    source: str = "learned_bridge"
    num_tokens: int = 64


@dataclass(frozen=True)
class BridgeConfig:
    enabled: bool = False
    variant: str = "crosskv"
    num_layers: int = 4
    num_heads: int = 8
    num_action_tokens: int = 16
    dropout: float = 0.0
    raw_gate_init: float = 0.0
    ffn_mult: int = 4


@dataclass(frozen=True)
class ContextConfig:
    mode: str = "fused_only"
    fused_gate_init: float = 0.0


@dataclass(frozen=True)
class ShortMemoryConfig:
    capacity: int = 2
    offsets: tuple[int, ...] = (32, 16)


@dataclass(frozen=True)
class LongMemoryConfig:
    capacity: int = 4


@dataclass(frozen=True)
class MemoryCompressionConfig:
    entry_tokens: int = 1
    num_heads: int = 8
    dropout: float = 0.0
    max_age_steps: int = 512


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = False
    kind: str = "dual_fifo_visual"
    hidden_dim: int = 896
    views: tuple[str, ...] = ("base", "wrist")
    short: ShortMemoryConfig = field(default_factory=ShortMemoryConfig)
    long: LongMemoryConfig = field(default_factory=LongMemoryConfig)
    compression: MemoryCompressionConfig = field(default_factory=MemoryCompressionConfig)


@dataclass(frozen=True)
class SkillConfig:
    enabled: bool = False
    num_tokens: int = 4


@dataclass(frozen=True)
class CoarsePlannerConfig:
    enabled: bool = False
    type: str = "query_cross_attention"
    hidden_dim: int = 896
    num_layers: int = 4
    num_heads: int = 8
    num_plan_steps: int = 1
    planning_horizon: int = 32
    latent_dim: int = 128
    latent_head_hidden_dim: int = 512
    segment_action_dim: int = 7
    dropout: float = 0.05
    loss: str = "intent_latent"
    loss_weight: float = 0.2
    latent_loss_weight: float = 1.0
    chunk_loss_weight: float = 0.25
    gripper_loss_weight: float = 2.0
    segment_autoencoder_checkpoint: str | None = None
    input_memory: bool = False
    placement: str = "bridge_crosskv"
    gripper_indices: tuple[int, ...] = (-1,)


@dataclass(frozen=True)
class ActionHeadConfig:
    kind: str = "flowmatching"
    use_existing_checkpoint_config: bool = False
    horizon: int | None = 32
    per_action_dim: int | None = None


@dataclass(frozen=True)
class BridgeHiMemConfig:
    experiment_name: str = "bridge_himem"
    seed: int = 42
    vlm: VLMConfig = field(default_factory=VLMConfig)
    action_query: ActionQueryConfig = field(default_factory=ActionQueryConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    coarse_planner: CoarsePlannerConfig = field(default_factory=CoarsePlannerConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BridgeHiMemConfig":
        if "bridge_himem" in mapping:
            mapping = _expect_mapping(mapping["bridge_himem"], "bridge_himem")
        _reject_unknown(mapping, _field_names(cls), "bridge_himem")
        config = cls(
            experiment_name=str(mapping.get("experiment_name", cls.experiment_name)),
            seed=_int(mapping.get("seed", cls.seed), "seed"),
            vlm=_build_dataclass(VLMConfig, mapping.get("vlm", {}), "vlm"),
            action_query=_build_dataclass(ActionQueryConfig, mapping.get("action_query", {}), "action_query"),
            bridge=_build_dataclass(BridgeConfig, mapping.get("bridge", {}), "bridge"),
            context=_build_dataclass(ContextConfig, mapping.get("context", {}), "context"),
            memory=_build_memory_config(mapping.get("memory", {})),
            skill=_build_dataclass(SkillConfig, mapping.get("skill", {}), "skill"),
            coarse_planner=_build_dataclass(
                CoarsePlannerConfig,
                mapping.get("coarse_planner", {}),
                "coarse_planner",
            ),
            action_head=_build_dataclass(ActionHeadConfig, mapping.get("action_head", {}), "action_head"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        _positive_int(self.vlm.hidden_dim, "vlm.hidden_dim")
        if self.vlm.raw_dim is not None:
            _positive_int(self.vlm.raw_dim, "vlm.raw_dim")
        if len(self.vlm.raw_layers) == 0:
            raise ValueError("vlm.raw_layers must select at least one hidden-state layer")

        if self.action_query.source not in ACTION_QUERY_SOURCES:
            raise ValueError(f"action_query.source must be one of {sorted(ACTION_QUERY_SOURCES)}")
        _positive_int(self.action_query.num_tokens, "action_query.num_tokens")

        if self.bridge.variant not in BRIDGE_VARIANTS:
            raise ValueError(f"bridge.variant must be one of {sorted(BRIDGE_VARIANTS)}")
        _positive_int(self.bridge.num_layers, "bridge.num_layers")
        _positive_int(self.bridge.num_heads, "bridge.num_heads")
        _positive_int(self.bridge.num_action_tokens, "bridge.num_action_tokens")
        _positive_int(self.bridge.ffn_mult, "bridge.ffn_mult")
        _non_negative_float(self.bridge.dropout, "bridge.dropout")
        if self.vlm.hidden_dim % self.bridge.num_heads != 0:
            raise ValueError("vlm.hidden_dim must be divisible by bridge.num_heads")

        if self.context.mode not in CONTEXT_MODES:
            raise ValueError(f"context.mode must be one of {sorted(CONTEXT_MODES)}")

        if self.memory.kind not in MEMORY_KINDS:
            raise ValueError(f"memory.kind must be one of {sorted(MEMORY_KINDS)}")
        _positive_int(self.memory.hidden_dim, "memory.hidden_dim")
        if len(self.memory.views) == 0:
            raise ValueError("memory.views must contain at least one view")
        _positive_int(self.memory.short.capacity, "memory.short.capacity")
        if len(self.memory.short.offsets) != self.memory.short.capacity:
            raise ValueError("memory.short.offsets length must match memory.short.capacity")
        for offset in self.memory.short.offsets:
            _positive_int(offset, "memory.short.offsets")
        _non_negative_int(self.memory.long.capacity, "memory.long.capacity")
        _positive_int(self.memory.compression.entry_tokens, "memory.compression.entry_tokens")
        _positive_int(self.memory.compression.num_heads, "memory.compression.num_heads")
        _non_negative_float(self.memory.compression.dropout, "memory.compression.dropout")
        _non_negative_int(self.memory.compression.max_age_steps, "memory.compression.max_age_steps")
        if self.memory.hidden_dim % self.memory.compression.num_heads != 0:
            raise ValueError("memory.hidden_dim must be divisible by memory.compression.num_heads")

        _positive_int(self.skill.num_tokens, "skill.num_tokens")

        if self.memory.enabled and self.memory.hidden_dim != self.vlm.hidden_dim:
            raise ValueError("memory.hidden_dim must match vlm.hidden_dim")
        if self.skill.enabled and not self.bridge.enabled:
            raise ValueError("skill.enabled=true requires bridge.enabled=true")
        if self.skill.enabled and self.bridge.variant != "mixed_latent":
            raise ValueError("skill tokens are implemented for mixed_latent experiments only")
        if self.context.mode == "fused_only" and self.memory.enabled:
            raise ValueError("context.mode=fused_only cannot expose memory to the action head")

        if self.coarse_planner.type not in COARSE_PLANNER_TYPES:
            raise ValueError(f"coarse_planner.type must be one of {sorted(COARSE_PLANNER_TYPES)}")
        if self.coarse_planner.loss not in COARSE_PLANNER_LOSSES:
            raise ValueError(f"coarse_planner.loss must be one of {sorted(COARSE_PLANNER_LOSSES)}")
        if self.coarse_planner.placement not in COARSE_PLANNER_PLACEMENTS:
            raise ValueError(f"coarse_planner.placement must be one of {sorted(COARSE_PLANNER_PLACEMENTS)}")
        _positive_int(self.coarse_planner.hidden_dim, "coarse_planner.hidden_dim")
        _positive_int(self.coarse_planner.num_layers, "coarse_planner.num_layers")
        _positive_int(self.coarse_planner.num_heads, "coarse_planner.num_heads")
        _positive_int(self.coarse_planner.num_plan_steps, "coarse_planner.num_plan_steps")
        _positive_int(self.coarse_planner.planning_horizon, "coarse_planner.planning_horizon")
        _positive_int(self.coarse_planner.latent_dim, "coarse_planner.latent_dim")
        _positive_int(self.coarse_planner.latent_head_hidden_dim, "coarse_planner.latent_head_hidden_dim")
        _positive_int(self.coarse_planner.segment_action_dim, "coarse_planner.segment_action_dim")
        _non_negative_float(self.coarse_planner.dropout, "coarse_planner.dropout")
        _non_negative_float(self.coarse_planner.loss_weight, "coarse_planner.loss_weight")
        _non_negative_float(self.coarse_planner.latent_loss_weight, "coarse_planner.latent_loss_weight")
        _non_negative_float(self.coarse_planner.chunk_loss_weight, "coarse_planner.chunk_loss_weight")
        _non_negative_float(self.coarse_planner.gripper_loss_weight, "coarse_planner.gripper_loss_weight")
        _validate_action_indices(
            self.coarse_planner.gripper_indices,
            self.coarse_planner.segment_action_dim,
            "coarse_planner.gripper_indices",
        )
        if self.coarse_planner.enabled:
            if not self.bridge.enabled:
                raise ValueError("coarse_planner.enabled=true requires bridge.enabled=true")
            if self.coarse_planner.hidden_dim != self.vlm.hidden_dim:
                raise ValueError("coarse_planner.hidden_dim must match vlm.hidden_dim")
            if self.coarse_planner.hidden_dim % self.coarse_planner.num_heads != 0:
                raise ValueError("coarse_planner.hidden_dim must be divisible by coarse_planner.num_heads")
            if self.coarse_planner.num_layers < 3:
                raise ValueError("coarse_planner.num_layers must be at least 3")
            if self.coarse_planner.input_memory:
                raise ValueError("coarse_planner.input_memory must remain false in the first version")
            if self.coarse_planner.planning_horizon % self.coarse_planner.num_plan_steps != 0:
                raise ValueError("coarse_planner.planning_horizon must be divisible by num_plan_steps")
        if self.action_head.kind != "flowmatching":
            raise ValueError("action_head.kind must be 'flowmatching'")
        if self.action_head.horizon is not None:
            _positive_int(self.action_head.horizon, "action_head.horizon")
        if self.action_head.per_action_dim is not None:
            _positive_int(self.action_head.per_action_dim, "action_head.per_action_dim")

    def to_legacy_model_config(self) -> dict[str, Any]:
        legacy: dict[str, Any] = {
            "use_bridge": self.bridge.enabled,
            "use_memory": self.memory.enabled,
            "bridge_variant": self.bridge.variant,
            "bridge_context_mode": self.context.mode,
            "bridge_fused_gate_init": self.context.fused_gate_init,
            "bridge_hidden_dim": self.vlm.hidden_dim,
            "bridge_raw_dim": self.vlm.raw_dim or self.vlm.hidden_dim,
            "bridge_raw_layers": list(self.vlm.raw_layers),
            "allow_image_token_truncation": self.vlm.allow_image_token_truncation,
            "bridge_num_layers": self.bridge.num_layers,
            "bridge_num_heads": self.bridge.num_heads,
            "bridge_num_tokens": self.bridge.num_action_tokens,
            "bridge_num_action_queries": self.action_query.num_tokens,
            "bridge_dropout": self.bridge.dropout,
            "bridge_raw_gate_init": self.bridge.raw_gate_init,
            "bridge_ffn_mult": self.bridge.ffn_mult,
            "memory_kind": self.memory.kind,
            "memory_hidden_dim": self.memory.hidden_dim,
            "memory_views": list(self.memory.views),
            "memory_short_capacity": self.memory.short.capacity,
            "memory_short_offsets": list(self.memory.short.offsets),
            "memory_long_capacity": self.memory.long.capacity,
            "memory_entry_tokens": self.memory.compression.entry_tokens,
            "memory_compression_num_heads": self.memory.compression.num_heads,
            "memory_compression_dropout": self.memory.compression.dropout,
            "memory_max_age_steps": self.memory.compression.max_age_steps,
            "skill_tokens_enabled": self.skill.enabled,
            "skill_num_tokens": self.skill.num_tokens,
            "coarse_planner_enabled": self.coarse_planner.enabled,
            "coarse_planner_type": self.coarse_planner.type,
            "coarse_planner_hidden_dim": self.coarse_planner.hidden_dim,
            "coarse_planner_num_layers": self.coarse_planner.num_layers,
            "coarse_planner_num_heads": self.coarse_planner.num_heads,
            "coarse_planner_num_plan_steps": self.coarse_planner.num_plan_steps,
            "coarse_planner_planning_horizon": self.coarse_planner.planning_horizon,
            "coarse_planner_latent_dim": self.coarse_planner.latent_dim,
            "coarse_planner_latent_head_hidden_dim": self.coarse_planner.latent_head_hidden_dim,
            "coarse_planner_action_dim": self.coarse_planner.segment_action_dim,
            "coarse_planner_dropout": self.coarse_planner.dropout,
            "coarse_planner_loss": self.coarse_planner.loss,
            "coarse_planner_loss_weight": self.coarse_planner.loss_weight,
            "coarse_planner_latent_loss_weight": self.coarse_planner.latent_loss_weight,
            "coarse_planner_chunk_loss_weight": self.coarse_planner.chunk_loss_weight,
            "coarse_planner_gripper_loss_weight": self.coarse_planner.gripper_loss_weight,
            "coarse_planner_segment_autoencoder_checkpoint": self.coarse_planner.segment_autoencoder_checkpoint,
            "coarse_planner_input_memory": self.coarse_planner.input_memory,
            "coarse_planner_placement": self.coarse_planner.placement,
            "coarse_planner_gripper_indices": list(self.coarse_planner.gripper_indices),
        }
        if not self.action_head.use_existing_checkpoint_config:
            if self.action_head.horizon is not None:
                legacy["horizon"] = self.action_head.horizon
            if self.action_head.per_action_dim is not None:
                legacy["per_action_dim"] = self.action_head.per_action_dim
        return legacy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bridge_himem_config(source: str | Path | Mapping[str, Any] | BridgeHiMemConfig) -> BridgeHiMemConfig:
    if isinstance(source, BridgeHiMemConfig):
        return source
    if isinstance(source, Mapping):
        return BridgeHiMemConfig.from_mapping(source)

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Bridge-HiMem config file not found: {path}")
    data = load_bridge_himem_config_mapping(path)
    return BridgeHiMemConfig.from_mapping(data)


def load_bridge_himem_config_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Bridge-HiMem config file not found: {path}")
    return _load_yaml_with_extends(path.resolve(), seen=set())


def _load_yaml_with_extends(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"Circular Bridge-HiMem config extends chain: {chain}")
    seen.add(path)

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to load Bridge-HiMem YAML configs") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Bridge-HiMem config must be a mapping: {path}")

    data = dict(data)
    extends = data.pop("extends", None)
    if extends is None:
        seen.remove(path)
        return data

    merged: dict[str, Any] = {}
    extends_list = [extends] if isinstance(extends, (str, Path)) else list(extends)
    for parent in extends_list:
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent_data = _load_yaml_with_extends(parent_path.resolve(), seen=seen)
        merged = _deep_merge(merged, parent_data)

    seen.remove(path)
    return _deep_merge(merged, data)


def _build_memory_config(value: Any) -> MemoryConfig:
    mapping = _expect_mapping(value, "memory")
    _reject_unknown(mapping, _field_names(MemoryConfig), "memory")
    return MemoryConfig(
        enabled=_bool(mapping.get("enabled", MemoryConfig.enabled), "memory.enabled"),
        kind=str(mapping.get("kind", MemoryConfig.kind)),
        hidden_dim=_int(mapping.get("hidden_dim", MemoryConfig.hidden_dim), "memory.hidden_dim"),
        views=_coerce_str_tuple(mapping.get("views", MemoryConfig.views), "memory.views"),
        short=_build_dataclass(ShortMemoryConfig, mapping.get("short", {}), "memory.short"),
        long=_build_dataclass(LongMemoryConfig, mapping.get("long", {}), "memory.long"),
        compression=_build_dataclass(
            MemoryCompressionConfig,
            mapping.get("compression", {}),
            "memory.compression",
        ),
    )


def _build_dataclass(cls: type, value: Any, label: str):
    mapping = _expect_mapping(value, label)
    _reject_unknown(mapping, _field_names(cls), label)
    kwargs = {}
    for name in cls.__dataclass_fields__:
        if name not in mapping:
            continue
        kwargs[name] = _coerce_field_value(name, mapping[name], f"{label}.{name}")
    return cls(**kwargs)


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")


def _field_names(cls: type) -> set[str]:
    return set(cls.__dataclass_fields__)


def _positive_int(value: Any, label: str) -> None:
    if _int(value, label) <= 0:
        raise ValueError(f"{label} must be positive")


def _non_negative_int(value: Any, label: str) -> None:
    if _int(value, label) < 0:
        raise ValueError(f"{label} must be non-negative")


def _non_negative_float(value: Any, label: str) -> None:
    if _float(value, label) < 0.0:
        raise ValueError(f"{label} must be non-negative")


def _int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc


def _float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got {value!r}") from exc


def _coerce_field_value(name: str, value: Any, label: str) -> Any:
    if value is None:
        return None
    if name in {
        "hidden_dim",
        "raw_dim",
        "num_tokens",
        "num_layers",
        "num_heads",
        "num_action_tokens",
        "ffn_mult",
        "horizon",
        "per_action_dim",
        "action_dim",
        "latent_dim",
        "latent_head_hidden_dim",
        "segment_action_dim",
        "num_plan_steps",
        "planning_horizon",
        "capacity",
        "entry_tokens",
        "max_age_steps",
    }:
        return _int(value, label)
    if name in {
        "dropout",
        "raw_gate_init",
        "fused_gate_init",
        "ema_decay",
        "loss_weight",
        "latent_loss_weight",
        "chunk_loss_weight",
        "gripper_loss_weight",
    }:
        return _float(value, label)
    if name in {"enabled", "freeze", "use_existing_checkpoint_config", "input_memory"}:
        return _bool(value, label)
    if name == "raw_layers":
        return _coerce_raw_layers(value, label)
    if name in {
        "source",
        "variant",
        "mode",
        "accumulator",
        "write_policy",
        "kind",
        "type",
        "loss",
        "segment_autoencoder_checkpoint",
    }:
        return str(value)
    if name in {"gripper_indices"}:
        return _coerce_int_tuple(value, label)
    if name == "offsets":
        return _coerce_int_tuple(value, label)
    if name == "views":
        return _coerce_str_tuple(value, label)
    return value


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, int):
        return bool(value)
    raise ValueError(f"{label} must be a boolean, got {value!r}")


def _coerce_raw_layers(value: Any, label: str) -> tuple[int | str, ...]:
    if isinstance(value, (int, str)):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{label} must be a sequence of layer selectors") from exc
    return tuple(_coerce_layer_selector(layer, label) for layer in values)


def _coerce_layer_selector(value: Any, label: str) -> int | str:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lstrip("-").isdigit():
            return int(normalized)
        return normalized
    raise ValueError(f"{label} values must be integers or named selectors, got {value!r}")


def _coerce_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{label} must be a sequence of integers") from exc
    return tuple(_int(item, label) for item in values)


def _coerce_str_tuple(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{label} must be a sequence of strings") from exc
    return tuple(str(item) for item in values)


def _validate_action_indices(indices: tuple[int, ...], action_dim: int, label: str) -> None:
    for index in indices:
        value = int(index)
        if value < 0:
            value += action_dim
        if value < 0 or value >= action_dim:
            raise ValueError(f"{label} index {index} is out of range for action_dim {action_dim}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
