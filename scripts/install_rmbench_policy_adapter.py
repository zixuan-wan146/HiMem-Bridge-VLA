#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_NAME = "HiMemBridgeVLA"
SOURCE_POLICY_DIR = REPO_ROOT / "evaluations" / "rmbench" / "policy" / POLICY_NAME


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the HiMem-Bridge-VLA policy adapter into RMBench.")
    parser.add_argument("--rmbench-root", required=True, help="Path to the RMBench repository root.")
    parser.add_argument("--force", action="store_true", help="Replace an existing policy adapter directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destination = install_policy_adapter(args.rmbench_root, force=bool(args.force))
    print(destination)
    return 0


def install_policy_adapter(rmbench_root: str | Path, *, force: bool = False) -> Path:
    source = SOURCE_POLICY_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"source policy adapter is missing: {source}")

    root = Path(rmbench_root).expanduser()
    policy_root = root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RMBench policy directory is missing: {policy_root}")

    destination = policy_root / POLICY_NAME
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists; use --force to replace it")
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


if __name__ == "__main__":
    raise SystemExit(main())

