#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.bridge_himem_config import load_bridge_himem_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Bridge-HiMem YAML configs.")
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="Specific YAML configs to validate. Defaults to configs/bridge_himem/**/*.yaml.",
    )
    args = parser.parse_args()

    paths = args.configs or sorted((REPO_ROOT / "configs" / "bridge_himem").glob("**/*.yaml"))
    if not paths:
        raise FileNotFoundError("No Bridge-HiMem YAML configs found")

    for path in paths:
        config = load_bridge_himem_config(path)
        legacy = config.to_legacy_model_config()
        print(
            f"{path}: experiment={config.experiment_name} "
            f"bridge={legacy['use_bridge']} memory={legacy['use_himem']} "
            f"variant={legacy['bridge_variant']} context={legacy['bridge_context_mode']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
