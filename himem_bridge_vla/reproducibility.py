from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ModuleNotFoundError:
        pass


def write_experiment_snapshot(save_dir: str | Path, config: Mapping[str, Any]) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    _write_json(save_path / "resolved_config.json", config)
    _write_json(save_path / "reproducibility.json", build_reproducibility_metadata(config))


def build_reproducibility_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": config.get("seed"),
        "deterministic": bool(config.get("deterministic", False)),
        "git": _git_metadata(),
        "bridge_himem_config_path": config.get("bridge_himem_config_path"),
        "experiment_name": _experiment_name(config),
    }


def _experiment_name(config: Mapping[str, Any]) -> str | None:
    bridge_config = config.get("bridge_himem")
    if isinstance(bridge_config, Mapping):
        experiment_name = bridge_config.get("experiment_name")
        if experiment_name is not None:
            return str(experiment_name)
    return None


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _git_dirty(),
    }


def _git_dirty() -> bool | None:
    status = _run_git(["status", "--porcelain"])
    if status is None:
        return None
    return bool(status.strip())


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
