from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config_utils import iter_dataset_entries, resolve_dataset_config_paths, validate_dataset_config_structure


DEFAULT_VIEW_MAP = {
    "image_1": "observation.images.image_1",
    "image_2": "observation.images.image_2",
    "image_3": "observation.images.image_3",
}


@dataclass(frozen=True)
class DatasetValidationIssue:
    level: str
    path: str
    message: str


def validate_configured_datasets(
    config: Mapping[str, Any],
    base_dir: str | Path,
    *,
    require_videos: bool = True,
) -> list[DatasetValidationIssue]:
    validate_dataset_config_structure(config)
    resolved_config = resolve_dataset_config_paths(config, base_dir)
    issues: list[DatasetValidationIssue] = []

    max_state_dim = int(resolved_config["max_state_dim"])
    max_action_dim = int(resolved_config["max_action_dim"])
    for group_name, dataset_name, dataset_config in iter_dataset_entries(resolved_config):
        dataset_label = f"{group_name}/{dataset_name}"
        dataset_path = Path(str(dataset_config["path"]))
        view_map = dataset_config.get("view_map") or DEFAULT_VIEW_MAP
        if not isinstance(view_map, Mapping) or not view_map:
            issues.append(DatasetValidationIssue("FAIL", str(dataset_path), f"{dataset_label} view_map must be a mapping"))
            continue
        issues.extend(
            validate_dataset_path(
                dataset_path,
                dataset_label=dataset_label,
                view_map=view_map,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                require_videos=require_videos,
                state_stat_keys=dataset_config.get("state_stat_keys", ("observation.state", "state")),
                action_stat_keys=dataset_config.get("action_stat_keys", ("action", "actions")),
            )
        )
    return issues


def validate_dataset_path(
    dataset_path: Path,
    *,
    dataset_label: str,
    view_map: Mapping[str, Any],
    max_state_dim: int,
    max_action_dim: int,
    require_videos: bool,
    state_stat_keys: Any = ("observation.state",),
    action_stat_keys: Any = ("action",),
) -> list[DatasetValidationIssue]:
    issues: list[DatasetValidationIssue] = []
    if not dataset_path.exists():
        return [DatasetValidationIssue("FAIL", str(dataset_path), f"{dataset_label} path does not exist")]
    if not dataset_path.is_dir():
        return [DatasetValidationIssue("FAIL", str(dataset_path), f"{dataset_label} path is not a directory")]

    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    stats_json_path = dataset_path / "meta" / "stats.json"
    episodes_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))

    issues.extend(validate_tasks_jsonl(tasks_path, dataset_label))
    issues.extend(validate_jsonl_objects(episodes_path, dataset_label, "episodes"))
    issues.extend(
        validate_stats_files(
            stats_json_path,
            episodes_stats_path,
            dataset_label,
            max_state_dim,
            max_action_dim,
            state_stat_keys=state_stat_keys,
            action_stat_keys=action_stat_keys,
        )
    )

    if not parquet_files:
        issues.append(DatasetValidationIssue("FAIL", str(dataset_path / "data/*/*.parquet"), f"{dataset_label} has no parquet files"))
    elif require_videos:
        issues.extend(validate_video_paths(dataset_path, parquet_files, view_map, dataset_label))

    return issues


def validate_tasks_jsonl(path: Path, dataset_label: str) -> list[DatasetValidationIssue]:
    issues, rows = read_jsonl_objects(path, dataset_label, "tasks")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row.get("task_index"), int) or isinstance(row.get("task_index"), bool):
            issues.append(DatasetValidationIssue("FAIL", str(path), f"{dataset_label} tasks line {index} has invalid task_index"))
        if not isinstance(row.get("task"), str) or not row.get("task"):
            issues.append(DatasetValidationIssue("FAIL", str(path), f"{dataset_label} tasks line {index} has invalid task"))
    return issues


def validate_jsonl_objects(path: Path, dataset_label: str, label: str) -> list[DatasetValidationIssue]:
    issues, _rows = read_jsonl_objects(path, dataset_label, label)
    return issues


def read_jsonl_objects(path: Path, dataset_label: str, label: str) -> tuple[list[DatasetValidationIssue], list[dict[str, Any]]]:
    issues: list[DatasetValidationIssue] = []
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return [DatasetValidationIssue("FAIL", str(path), f"{dataset_label} missing {label} file")], rows

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            issues.append(DatasetValidationIssue("FAIL", str(path), f"{label} line {line_number} is invalid JSON: {exc}"))
            continue
        if not isinstance(row, dict):
            issues.append(DatasetValidationIssue("FAIL", str(path), f"{label} line {line_number} must be an object"))
            continue
        rows.append(row)

    if not rows:
        issues.append(DatasetValidationIssue("FAIL", str(path), f"{dataset_label} {label} file has no records"))
    return issues, rows


def validate_stats_files(
    stats_json_path: Path,
    episodes_stats_path: Path,
    dataset_label: str,
    max_state_dim: int,
    max_action_dim: int,
    state_stat_keys: Any = ("observation.state",),
    action_stat_keys: Any = ("action",),
) -> list[DatasetValidationIssue]:
    if stats_json_path.exists():
        try:
            payload = json.loads(stats_json_path.read_text())
        except json.JSONDecodeError as exc:
            return [DatasetValidationIssue("FAIL", str(stats_json_path), f"stats.json is invalid JSON: {exc}")]
        return _validate_stats_payload(
            payload,
            str(stats_json_path),
            dataset_label,
            max_state_dim,
            max_action_dim,
            state_stat_keys=state_stat_keys,
            action_stat_keys=action_stat_keys,
        )

    if not episodes_stats_path.exists():
        return [
            DatasetValidationIssue(
                "FAIL",
                str(episodes_stats_path),
                f"{dataset_label} missing stats.json or episodes_stats.jsonl",
            )
        ]

    issues, rows = read_jsonl_objects(episodes_stats_path, dataset_label, "episodes_stats")
    for index, row in enumerate(rows, start=1):
        stats = row.get("stats")
        if not isinstance(stats, dict):
            issues.append(DatasetValidationIssue("FAIL", str(episodes_stats_path), f"episodes_stats line {index} missing stats object"))
            continue
        issues.extend(
            _validate_stats_payload(
                stats,
                str(episodes_stats_path),
                f"{dataset_label} line {index}",
                max_state_dim,
                max_action_dim,
                state_stat_keys=state_stat_keys,
                action_stat_keys=action_stat_keys,
            )
        )
    return issues


def _validate_stats_payload(
    stats: Any,
    path_label: str,
    dataset_label: str,
    max_state_dim: int,
    max_action_dim: int,
    state_stat_keys: Any = ("observation.state",),
    action_stat_keys: Any = ("action",),
) -> list[DatasetValidationIssue]:
    if not isinstance(stats, dict):
        return [DatasetValidationIssue("FAIL", path_label, f"{dataset_label} stats must be an object")]

    issues: list[DatasetValidationIssue] = []
    issues.extend(validate_minmax_stat(stats, state_stat_keys, max_state_dim, path_label, dataset_label))
    issues.extend(validate_minmax_stat(stats, action_stat_keys, max_action_dim, path_label, dataset_label))
    return issues


def validate_minmax_stat(
    stats: Mapping[str, Any],
    stat_name: str | list[str] | tuple[str, ...],
    max_dim: int,
    path_label: str,
    dataset_label: str,
) -> list[DatasetValidationIssue]:
    stat_names = [stat_name] if isinstance(stat_name, str) else [str(name) for name in stat_name]
    stat = None
    selected_stat_name = stat_names[0]
    for candidate in stat_names:
        candidate_stat = stats.get(candidate)
        if isinstance(candidate_stat, Mapping):
            stat = candidate_stat
            selected_stat_name = candidate
            break
    if not isinstance(stat, Mapping):
        return [
            DatasetValidationIssue(
                "FAIL",
                path_label,
                f"{dataset_label} stats missing one of {', '.join(stat_names)} min/max object",
            )
        ]

    mins = stat.get("min")
    maxs = stat.get("max")
    issues: list[DatasetValidationIssue] = []
    issues.extend(validate_numeric_vector(mins, f"{selected_stat_name}.min", max_dim, path_label, dataset_label))
    issues.extend(validate_numeric_vector(maxs, f"{selected_stat_name}.max", max_dim, path_label, dataset_label))
    if issues:
        return issues

    if len(mins) != len(maxs):
        return [
            DatasetValidationIssue(
                "FAIL",
                path_label,
                f"{dataset_label} {selected_stat_name}.min and max must have the same length",
            )
        ]
    for index, (min_value, max_value) in enumerate(zip(mins, maxs)):
        if float(min_value) > float(max_value):
            issues.append(
                DatasetValidationIssue(
                    "FAIL",
                    path_label,
                    f"{dataset_label} {selected_stat_name}.min[{index}] must be <= max[{index}]",
                )
            )
    return issues


def validate_numeric_vector(
    value: Any,
    label: str,
    max_dim: int,
    path_label: str,
    dataset_label: str,
) -> list[DatasetValidationIssue]:
    if not isinstance(value, list) or not value:
        return [DatasetValidationIssue("FAIL", path_label, f"{dataset_label} {label} must be a non-empty list")]
    if len(value) > max_dim:
        return [DatasetValidationIssue("FAIL", path_label, f"{dataset_label} {label} length {len(value)} exceeds max_dim {max_dim}")]
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)):
            return [DatasetValidationIssue("FAIL", path_label, f"{dataset_label} {label}[{index}] must be a finite number")]
    return []


def validate_video_paths(
    dataset_path: Path,
    parquet_files: list[Path],
    view_map: Mapping[str, Any],
    dataset_label: str,
) -> list[DatasetValidationIssue]:
    issues: list[DatasetValidationIssue] = []
    for view_key, view_value in view_map.items():
        view_folders = _normalize_view_candidates(view_value)
        if not view_folders:
            issues.append(DatasetValidationIssue("FAIL", str(dataset_path), f"{dataset_label} view {view_key!r} has invalid folder"))
            continue
        for parquet_path in parquet_files:
            candidates = [
                dataset_path / "videos" / parquet_path.parent.name / view_folder / f"{parquet_path.stem}.mp4"
                for view_folder in view_folders
            ]
            if not any(video_path.exists() for video_path in candidates):
                joined = ", ".join(str(video_path) for video_path in candidates)
                issues.append(DatasetValidationIssue("FAIL", joined, f"{dataset_label} missing video for {parquet_path}"))
    return issues


def _normalize_view_candidates(view_value: Any) -> list[str]:
    if isinstance(view_value, str):
        return [view_value] if view_value else []
    if isinstance(view_value, list):
        return [str(item) for item in view_value if str(item)]
    return []
