from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    from .bridge_himem_config import BridgeHiMemConfig, load_bridge_himem_config
except ImportError:
    from bridge_himem_config import BridgeHiMemConfig, load_bridge_himem_config


def resolve_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the single resolved config dictionary used by training and model init."""

    if bool(config.get("experiment_config_resolved", False)):
        return dict(config)

    resolved = dict(config)
    explicit_keys = {
        str(key)
        for key in resolved.get("_explicit_config_keys", ())
        if key not in {"bridge_himem_config", "bridge_himem", "bridge_himem_config_path"}
    }
    explicit_values = {key: resolved[key] for key in explicit_keys if key in resolved and resolved[key] is not None}
    bridge_spec = resolved.get("bridge_himem_config")
    if bridge_spec is None:
        bridge_spec = resolved.get("bridge_himem")

    if bridge_spec is not None:
        bridge_config = load_bridge_himem_config(bridge_spec)
        resolved.update(bridge_config.to_legacy_model_config())
        resolved["bridge_himem"] = bridge_config.to_dict()
        if isinstance(bridge_spec, (str, Path)):
            resolved["bridge_himem_config_path"] = str(bridge_spec)
        elif isinstance(bridge_spec, BridgeHiMemConfig):
            resolved["bridge_himem_config_path"] = bridge_config.experiment_name
        if resolved.get("seed") is None:
            resolved["seed"] = bridge_config.seed

    # Bridge YAML defines the model family. Training profiles and CLI flags are
    # still allowed to override fields such as the W4 planner checkpoint.
    resolved.update(explicit_values)

    if resolved.get("seed") is None:
        resolved["seed"] = 42

    resolved["experiment_config_resolved"] = True
    return resolved
