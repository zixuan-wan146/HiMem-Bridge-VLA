#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluations.libero.libero_eval_summary import build_run_metadata  # noqa: E402


TRACKED_ENV_KEYS = (
    "HIMEM_LIBERO_RUN_DIR",
    "HIMEM_LIBERO_LOG_DIR",
    "HIMEM_LIBERO_VIDEO_DIR",
    "HIMEM_LIBERO_LOG_FILE",
    "HIMEM_LIBERO_RESULT_FILE",
    "HIMEM_LIBERO_MANIFEST_FILE",
    "HIMEM_LIBERO_CKPT_NAME",
    "HIMEM_LIBERO_TASK_SUITES",
    "HIMEM_LIBERO_TASK_LIMIT",
    "HIMEM_LIBERO_EPISODES",
    "HIMEM_LIBERO_MAX_STEPS",
    "HIMEM_LIBERO_HORIZON",
    "HIMEM_LIBERO_SEED",
    "HIMEM_SERVER_URI",
    "HIMEM_MUJOCO_GL",
    "LIBERO_PYTHON",
    "PYOPENGL_PLATFORM",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a manifest before a LIBERO smoke/eval run.")
    parser.add_argument("--output", required=True, help="Manifest JSON output path.")
    parser.add_argument(
        "--run-kind",
        required=True,
        choices=("smoke", "eval"),
        help="LIBERO run type that produced this manifest.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root used for git metadata.",
    )
    return parser.parse_args(argv)


def build_manifest(
    *,
    run_kind: str,
    repo_root: str | Path,
    environ: Mapping[str, str] | None = None,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    environ = os.environ if environ is None else environ
    metadata = build_run_metadata(repo_root=repo_root, environ=environ, argv=argv)
    return {
        "schema_version": 1,
        "run_kind": run_kind,
        "metadata": metadata,
        "libero": {
            key: str(environ[key])
            for key in TRACKED_ENV_KEYS
            if key in environ and environ[key] != ""
        },
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    manifest_path = Path(path).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        run_kind=args.run_kind,
        repo_root=args.repo_root,
        argv=sys.argv if argv is None else ["write_libero_run_manifest.py", *argv],
    )
    output_path = write_manifest(args.output, manifest)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
