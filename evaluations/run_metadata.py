from __future__ import annotations

from datetime import datetime, timezone
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


def build_run_metadata(
    *,
    repo_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    repo_path = Path(repo_root).expanduser().resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    environ = os.environ if environ is None else environ
    argv = sys.argv if argv is None else argv
    created_at_utc = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "created_at_utc": created_at_utc,
        "cwd": str(Path.cwd()),
        "argv": [str(item) for item in argv],
        "command": " ".join(str(item) for item in argv),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git": {
            "repo_root": str(repo_path),
            "commit": _git_output(repo_path, "rev-parse", "HEAD"),
            "branch": _git_output(repo_path, "rev-parse", "--abbrev-ref", "HEAD"),
            "is_dirty": _git_is_dirty(repo_path),
        },
        "environment": _safe_environment(environ),
    }


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_is_dirty(repo_root: Path) -> bool | None:
    status = _git_output(repo_root, "status", "--porcelain")
    if status is None:
        return None
    return bool(status)


def _safe_environment(environ: Mapping[str, str]) -> dict[str, str]:
    allowed_exact = {
        "CALVIN_ROOT",
        "HF_ENDPOINT",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "LIBERO_DATASETS_DIR",
        "LIBERO_ENV_PREFIX",
        "LIBERO_PYTHON",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
    }
    allowed_prefixes = ("HIMEM_",)
    blocked_fragments = ("TOKEN", "SECRET", "PASSWORD", "KEY")

    safe_items = {}
    for key, value in environ.items():
        if any(fragment in key.upper() for fragment in blocked_fragments):
            continue
        if key in allowed_exact or any(key.startswith(prefix) for prefix in allowed_prefixes):
            safe_items[key] = str(value)
    return dict(sorted(safe_items.items()))
