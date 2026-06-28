#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "src" / "himem_bridge_vla").is_dir():
            return candidate
    raise RuntimeError("Could not locate HiMem-Bridge-VLA repository root")


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from himem_bridge_vla.cli.cache.build_libero_episode_feature_cache import main


if __name__ == "__main__":
    raise SystemExit(main())
