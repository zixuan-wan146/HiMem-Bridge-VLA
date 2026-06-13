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

from evaluations.run_metadata import build_run_metadata  # noqa: E402


TRACKED_ENV_KEYS = (
    "HIMEM_CALVIN_RUN_DIR",
    "HIMEM_CALVIN_LOG_DIR",
    "HIMEM_CALVIN_VIDEO_DIR",
    "HIMEM_CALVIN_LOG_FILE",
    "HIMEM_CALVIN_RESULT_FILE",
    "HIMEM_CALVIN_MANIFEST_FILE",
    "HIMEM_CALVIN_CKPT_NAME",
    "HIMEM_CALVIN_ROOT",
    "HIMEM_CALVIN_DATASET_PATH",
    "HIMEM_CALVIN_ANNOTATIONS_PATH",
    "HIMEM_CALVIN_NUM_SEQUENCES",
    "HIMEM_CALVIN_SEQUENCE_OFFSET",
    "HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK",
    "HIMEM_CALVIN_HORIZON",
    "HIMEM_CALVIN_SEED",
    "HIMEM_CALVIN_SAVE_VIDEO",
    "HIMEM_CALVIN_VIDEO_FPS",
    "HIMEM_CALVIN_GRIPPER_MODE",
    "HIMEM_CALVIN_RESET_MEMORY_SCOPE",
    "HIMEM_CALVIN_SHOW_GUI",
    "HIMEM_SERVER_URI",
    "HIMEM_MUJOCO_GL",
    "CALVIN_ROOT",
    "CALVIN_PYTHON",
    "PYOPENGL_PLATFORM",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a manifest before a CALVIN smoke/eval run.")
    parser.add_argument("--output", required=True, help="Manifest JSON output path.")
    parser.add_argument(
        "--run-kind",
        required=True,
        choices=("smoke", "eval"),
        help="CALVIN run type that produced this manifest.",
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
        "calvin": {
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
        argv=sys.argv if argv is None else ["write_calvin_run_manifest.py", *argv],
    )
    output_path = write_manifest(args.output, manifest)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
