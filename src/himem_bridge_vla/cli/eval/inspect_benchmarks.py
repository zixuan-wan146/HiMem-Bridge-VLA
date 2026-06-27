#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_RMBENCH_TASKS = (
    "observe_and_pickup",
    "rearrange_blocks",
    "put_back_block",
    "swap_blocks",
    "swap_T",
    "blocks_ranking_try",
    "press_button",
    "cover_blocks",
    "battery_try",
)

LIBERO_PLUS_EXACT_DIR_NAMES = (
    "libero_plus",
    "libero-plus",
    "LIBERO-Plus",
    "LIBERO_PLUS",
)

LIBERO_PLUS_RELATED_BUT_DIFFERENT_NAMES = (
    "libero+",
    "LIBERO+",
    "libero_pro",
    "libero-pro",
    "LIBERO-PRO",
    "libero_para",
    "libero-para",
    "LIBERO-Para",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local benchmark assets for reproducible runs.")
    parser.add_argument("--data-root", default=os.environ.get("AUTODL_TMP", str(Path.home() / "autodl-tmp")))
    parser.add_argument("--libero-root", default=None, help="Defaults to <data-root>/libero/datasets.")
    parser.add_argument("--libero-plus-root", default=None, help="Defaults to <data-root>/libero_plus.")
    parser.add_argument("--rmbench-root", default=None, help="Defaults to <data-root>/benchmarks/RMBench.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--allow-missing", action="store_true", help="Return success even when a benchmark is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    inventory = build_inventory(
        data_root=data_root,
        libero_root=Path(args.libero_root).expanduser().resolve() if args.libero_root else data_root / "libero" / "datasets",
        libero_plus_root=(
            Path(args.libero_plus_root).expanduser().resolve() if args.libero_plus_root else data_root / "libero_plus"
        ),
        rmbench_root=Path(args.rmbench_root).expanduser().resolve() if args.rmbench_root else data_root / "benchmarks" / "RMBench",
    )
    payload = json.dumps(inventory, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    if args.allow_missing:
        return 0
    missing = [name for name, item in inventory["benchmarks"].items() if not item["exists"]]
    return 1 if missing else 0


def build_inventory(
    *,
    data_root: Path,
    libero_root: Path,
    libero_plus_root: Path,
    rmbench_root: Path,
) -> dict[str, Any]:
    return {
        "data_root": str(data_root),
        "benchmarks": {
            "libero": inspect_libero(libero_root),
            "libero_plus": inspect_libero_plus(libero_plus_root, data_root=data_root),
            "rmbench": inspect_rmbench(rmbench_root),
        },
    }


def inspect_libero(root: Path) -> dict[str, Any]:
    suites = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "libero_100"):
        suite_root = root / suite
        demo_files = sorted(suite_root.glob("*_demo.hdf5")) if suite_root.exists() else []
        suites[suite] = {
            "exists": suite_root.exists(),
            "path": str(suite_root),
            "demo_files": len(demo_files),
            "sample_files": [path.name for path in demo_files[:3]],
        }
    return {
        "exists": root.exists(),
        "path": str(root),
        "suites": suites,
        "total_demo_files": sum(item["demo_files"] for item in suites.values()),
    }


def inspect_libero_plus(root: Path, *, data_root: Path | None = None) -> dict[str, Any]:
    candidate_files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    related_candidates = _find_libero_plus_related_candidates(data_root or root.parent, exact_root=root)
    status = "available" if root.exists() else "missing"
    notes = [
        "This field tracks the exact LIBERO-Plus benchmark target.",
        "Name-similar resources such as LIBERO+, LIBERO-PRO, or LIBERO-Para are reported as related candidates only.",
    ]
    if not root.exists() and related_candidates:
        notes.append("Related candidates exist locally, but they are not treated as LIBERO-Plus by default.")
    return {
        "exists": root.exists(),
        "path": str(root),
        "status": status,
        "file_count": len(candidate_files),
        "sample_files": [str(path.relative_to(root)) for path in candidate_files[:10]] if root.exists() else [],
        "related_candidates": related_candidates,
        "notes": notes,
    }


def inspect_rmbench(root: Path) -> dict[str, Any]:
    manifest_path = root / "data" / "rmbench_9tasks_manifest.json"
    manifest = _read_json(manifest_path)
    task_names = tuple(manifest.get("tasks") or DEFAULT_RMBENCH_TASKS) if isinstance(manifest, dict) else DEFAULT_RMBENCH_TASKS
    tasks = {}
    for task in task_names:
        task_root = root / "data" / task / "demo_clean"
        tasks[task] = {
            "exists": task_root.exists(),
            "path": str(task_root),
            "hdf5_files": _count(task_root / "data", "*.hdf5"),
            "traj_files": _count(task_root / "_traj_data", "*"),
            "instruction_files": _count(task_root / "instructions", "*.json"),
            "video_files": _count(task_root / "video", "*"),
        }
    return {
        "exists": root.exists(),
        "path": str(root),
        "manifest": {
            "exists": manifest_path.exists(),
            "path": str(manifest_path),
            "repo_id": manifest.get("repo_id") if isinstance(manifest, dict) else None,
            "file_count": manifest.get("file_count") if isinstance(manifest, dict) else None,
            "skip_video": manifest.get("skip_video") if isinstance(manifest, dict) else None,
        },
        "tasks": tasks,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _count(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.glob(pattern) if path.is_file())


def _find_libero_plus_related_candidates(data_root: Path, *, exact_root: Path) -> list[dict[str, Any]]:
    if not data_root.exists():
        return []
    names = set(LIBERO_PLUS_EXACT_DIR_NAMES) | set(LIBERO_PLUS_RELATED_BUT_DIFFERENT_NAMES)
    candidates = []
    search_roots = [data_root, data_root / "benchmarks", data_root / "libero"]
    seen: set[Path] = set()
    for parent in search_roots:
        if not parent.exists() or not parent.is_dir():
            continue
        for child in sorted(parent.iterdir()):
            if child.name not in names or child in seen:
                continue
            seen.add(child)
            relation = "exact-name-candidate" if child.name in LIBERO_PLUS_EXACT_DIR_NAMES else "name-similar-not-equivalent"
            candidates.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "exists": child.exists(),
                    "relation": relation,
                    "is_selected_root": child.resolve() == exact_root.resolve(),
                }
            )
    return candidates


if __name__ == "__main__":
    raise SystemExit(main())
