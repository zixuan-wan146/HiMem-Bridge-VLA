#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from himem_bridge_vla.path_utils import find_repo_root

REPO_ROOT_FOR_IMPORTS = find_repo_root(__file__)
SRC_ROOT_FOR_IMPORTS = REPO_ROOT_FOR_IMPORTS / "src"
for import_root in (REPO_ROOT_FOR_IMPORTS, SRC_ROOT_FOR_IMPORTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from himem_bridge_vla.dataset.config_utils import (  # noqa: E402
    iter_dataset_entries,
    resolve_dataset_config_paths,
    validate_dataset_config_structure,
)
from himem_bridge_vla.dataset.validation import validate_configured_datasets  # noqa: E402
from himem_bridge_vla.path_utils import display_project_path, project_path  # noqa: E402
from himem_bridge_vla.runtime_config import TARGET_STATE_DIM  # noqa: E402


REQUIRED_REPO_FILES = (
    "README.md",
    "requirements-policy.json",
    "requirements-libero.txt",
    "requirements.txt",
    "src/himem_bridge_vla/training_config.py",
    "src/himem_bridge_vla/server_protocol.py",
    "src/himem_bridge_vla/runtime/contract.py",
    "src/himem_bridge_vla/runtime/feature_extractor.py",
    "src/himem_bridge_vla/runtime/inference_engine.py",
    "src/himem_bridge_vla/runtime/memory_builder.py",
    "src/himem_bridge_vla/runtime/websocket_server.py",
    "src/himem_bridge_vla/benchmarks/base.py",
    "src/himem_bridge_vla/benchmarks/libero/action_protocol.py",
    "src/himem_bridge_vla/benchmarks/libero/config.py",
    "src/himem_bridge_vla/benchmarks/libero/eval_summary.py",
    "src/himem_bridge_vla/benchmarks/libero/history.py",
    "src/himem_bridge_vla/benchmarks/libero/observation.py",
    "src/himem_bridge_vla/benchmarks/libero/request_builder.py",
    "src/himem_bridge_vla/benchmarks/libero/runner.py",
    "src/himem_bridge_vla/benchmarks/libero/spec.py",
    "src/himem_bridge_vla/benchmarks/rmbench/action_protocol.py",
    "src/himem_bridge_vla/benchmarks/rmbench/observation.py",
    "src/himem_bridge_vla/benchmarks/rmbench/request_builder.py",
    "src/himem_bridge_vla/benchmarks/rmbench/eval_client.py",
    "src/himem_bridge_vla/benchmarks/rmbench/policy_adapter.py",
    "src/himem_bridge_vla/benchmarks/rmbench/runner.py",
    "src/himem_bridge_vla/benchmarks/rmbench/spec.py",
    "src/himem_bridge_vla/model/himem_bridge_vla.py",
    "scripts/serve/serve_policy.py",
    "scripts/eval/eval_libero.py",
    "scripts/eval/eval_rmbench.py",
    "scripts/train/stage1/libero.py",
    "configs/datasets/simulation.yaml",
    "src/himem_bridge_vla/dataset/config_utils.py",
    "src/himem_bridge_vla/dataset/validation.py",
    "evaluations/legacy/libero/libero_action_protocol.py",
    "evaluations/legacy/libero/libero_client_config.py",
    "evaluations/legacy/libero/libero_eval_summary.py",
    "evaluations/legacy/libero/libero_client_4tasks.py",
    "scripts/quality/preflight.py",
    "scripts/quality/audit_requirements.py",
    "scripts/quality/check_runtime_environment.py",
    "scripts/quality/check_repo.sh",
    "scripts/quality/validate_training_configs.py",
    "scripts/eval/libero_profile.sh",
    "scripts/setup/download_libero_checkpoint.sh",
    "scripts/quality/validate_training_dataset.py",
    "scripts/setup/setup_libero_env.sh",
    "scripts/serve/start_himem_server.sh",
    "scripts/maintenance/export_unpushed_commits.sh",
    "scripts/eval/run_libero_smoke.sh",
    "scripts/eval/run_libero_eval.sh",
    "scripts/report/write_libero_run_manifest.py",
    "scripts/report/summarize_libero_results.py",
    "scripts/report/check_libero_metrics.py",
    "scripts/report/report_libero_runs.py",
    "scripts/eval/plan_libero_run.py",
    "scripts/eval/init_libero_experiment.py",
    "scripts/eval/inspect_benchmarks.py",
    "scripts/setup/download_rmbench_tasks.py",
    "scripts/cache/build_libero_memory_replay_index.py",
    "scripts/cache/build_rmbench_norm_stats.py",
    "scripts/cache/build_rmbench_memory_replay_index.py",
    "scripts/setup/install_rmbench_policy_adapter.py",
    "scripts/eval/plan_rmbench_eval.py",
    "scripts/eval/run_rmbench_eval.sh",
    "scripts/report/write_rmbench_run_manifest.py",
    "evaluations/legacy/rmbench/policy/HiMemBridgeVLA/deploy_policy.py",
    "evaluations/legacy/rmbench/policy/HiMemBridgeVLA/deploy_policy.yml",
    "configs/runtime/libero_profiles/smoke.env",
    "configs/runtime/libero_profiles/full_eval.env",
)

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "norm_stats.json",
    "model.pt",
)

HIMEM_RUNTIME_IMPORTS = (
    "torch",
    "websockets",
    "cv2",
    "PIL",
    "torchvision",
)

LIBERO_RUNTIME_IMPORTS = (
    "libero",
    "robosuite",
    "mujoco",
    "websockets",
    "imageio",
)

LIBERO_SUMMARY_COUNT_FIELDS = (
    "total_episodes",
    "successful_episodes",
    "failed_episodes",
)

LIBERO_SUMMARY_FLOAT_FIELDS = (
    "success_rate",
    "average_decision_steps",
    "average_control_steps",
    "average_success_decision_steps",
)

LIBERO_EPISODE_REQUIRED_FIELDS = (
    "task_suite",
    "task_id",
    "episode_id",
    "task_description",
    "success",
    "decision_steps",
    "control_steps",
)

LIBERO_MANIFEST_REQUIRED_ENV = (
    "HIMEM_LIBERO_CKPT_NAME",
    "HIMEM_LIBERO_LOG_DIR",
    "HIMEM_LIBERO_VIDEO_DIR",
    "HIMEM_LIBERO_LOG_FILE",
    "HIMEM_LIBERO_RESULT_FILE",
    "HIMEM_LIBERO_MANIFEST_FILE",
    "HIMEM_LIBERO_TASK_SUITES",
    "HIMEM_LIBERO_TASK_LIMIT",
    "HIMEM_LIBERO_EPISODES",
    "HIMEM_LIBERO_MAX_STEPS",
    "HIMEM_LIBERO_HORIZON",
    "HIMEM_SERVER_URI",
    "HIMEM_MUJOCO_GL",
)

SENSITIVE_ENV_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "KEY")


@dataclass(frozen=True)
class CheckResult:
    level: str
    name: str
    message: str


class Report:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def ok(self, name: str, message: str) -> None:
        self.results.append(CheckResult("OK", name, message))

    def warn(self, name: str, message: str) -> None:
        self.results.append(CheckResult("WARN", name, message))

    def fail(self, name: str, message: str) -> None:
        self.results.append(CheckResult("FAIL", name, message))

    @property
    def has_failures(self) -> bool:
        return any(result.level == "FAIL" for result in self.results)

    def print(self) -> None:
        for result in self.results:
            print(f"[{result.level}] {result.name}: {result.message}")


def repo_root_from_script() -> Path:
    return find_repo_root(__file__)


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    return project_path(value, repo_root)


def check_repo_layout(repo_root: Path, report: Report) -> None:
    if not repo_root.exists():
        report.fail("repo", "repository root does not exist: .")
        return

    missing = [path for path in REQUIRED_REPO_FILES if not (repo_root / path).exists()]
    if missing:
        report.fail("repo", f"missing required files: {', '.join(missing)}")
    else:
        report.ok("repo", "required files present under .")

    git_dir = repo_root / ".git"
    if git_dir.exists():
        report.ok("git", ".git directory present")
    else:
        report.warn("git", "repository root has no .git directory")


def check_script_permissions(repo_root: Path, report: Report) -> None:
    scripts_dir = repo_root / "scripts"
    scripts = sorted(scripts_dir.rglob("*.sh"))
    if not scripts:
        report.fail("scripts", f"no shell scripts found under {display_project_path(scripts_dir, repo_root)}")
        return

    non_executable = [str(path.relative_to(repo_root)) for path in scripts if not os.access(path, os.X_OK)]
    if non_executable:
        report.fail("scripts", f"not executable: {', '.join(non_executable)}")
    else:
        report.ok("scripts", f"{len(scripts)} shell scripts are executable")


def check_shell_syntax(repo_root: Path, report: Report) -> None:
    scripts = sorted((repo_root / "scripts").rglob("*.sh"))
    if not scripts:
        return

    bash = importlib.util.find_spec("subprocess")
    if bash is None:
        report.warn("shell", "subprocess module is unavailable; skipped bash -n")
        return

    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        rel = script.relative_to(repo_root)
        if result.returncode == 0:
            report.ok("shell", f"{rel} syntax OK")
        else:
            stderr = result.stderr.strip() or result.stdout.strip()
            report.fail("shell", f"{rel} syntax error: {stderr}")


def check_imports(import_names: tuple[str, ...], report: Report, group_name: str) -> None:
    missing = [name for name in import_names if importlib.util.find_spec(name) is None]
    if missing:
        report.fail(group_name, f"missing Python packages: {', '.join(missing)}")
    else:
        report.ok(group_name, f"required Python packages importable: {', '.join(import_names)}")


def check_checkpoint_dir(ckpt_dir: Path, report: Report) -> None:
    if not ckpt_dir.exists():
        report.fail("checkpoint", f"directory does not exist: {ckpt_dir}")
        return
    if not ckpt_dir.is_dir():
        report.fail("checkpoint", f"path is not a directory: {ckpt_dir}")
        return

    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (ckpt_dir / name).exists()]
    if missing:
        report.fail("checkpoint", f"missing required files in {ckpt_dir}: {', '.join(missing)}")
        return

    payloads = {}
    for json_name in ("config.json", "norm_stats.json"):
        path = ckpt_dir / json_name
        try:
            with path.open("r") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            report.fail("checkpoint", f"{json_name} is not valid JSON: {exc}")
            return
        if not isinstance(payload, dict):
            report.fail("checkpoint", f"{json_name} must contain a JSON object")
            return
        payloads[json_name] = payload

    config_error = validate_checkpoint_config(payloads["config.json"])
    if config_error:
        report.fail("checkpoint", f"config.json: {config_error}")
        return

    stats_error = validate_norm_stats(
        payloads["norm_stats.json"],
        state_dim=payloads["config.json"].get("state_dim"),
        action_dim=payloads["config.json"].get("per_action_dim"),
    )
    if stats_error:
        report.fail("checkpoint", f"norm_stats.json: {stats_error}")
        return

    weight_path = ckpt_dir / "model.pt"
    if weight_path.stat().st_size == 0:
        report.fail("checkpoint", f"checkpoint weight file is empty: {weight_path}")
        return

    report.ok("checkpoint", f"required checkpoint files are present in {ckpt_dir}")


def validate_checkpoint_config(config: dict[str, Any], target_dim: int = TARGET_STATE_DIM) -> str | None:
    positive_ints = {}
    for key in ("horizon", "per_action_dim", "state_dim"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return f"{key} must be a positive integer when present"
        positive_ints[key] = value

    per_action_dim = positive_ints.get("per_action_dim")
    if per_action_dim is not None and per_action_dim > target_dim:
        return f"per_action_dim {per_action_dim} exceeds server target dimension {target_dim}"

    state_dim = positive_ints.get("state_dim")
    if state_dim is not None and state_dim > target_dim:
        return f"state_dim {state_dim} exceeds server target dimension {target_dim}"

    action_dim = config.get("action_dim")
    if action_dim is not None:
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim <= 0:
            return "action_dim must be a positive integer when present"
        horizon = positive_ints.get("horizon")
        if horizon is not None and per_action_dim is not None and action_dim != horizon * per_action_dim:
            return (
                f"action_dim {action_dim} must equal horizon {horizon} "
                f"* per_action_dim {per_action_dim}"
            )

    return None


def validate_norm_stats(
    stats: dict[str, Any],
    target_dim: int = TARGET_STATE_DIM,
    *,
    state_dim: Any = None,
    action_dim: Any = None,
) -> str | None:
    if not stats:
        return "expected at least one robot stats entry"

    stat_dims = {
        "observation.state": _configured_dim_or_target(state_dim, target_dim),
        "action": _configured_dim_or_target(action_dim, target_dim),
    }
    for robot_name, robot_stats in stats.items():
        if not isinstance(robot_stats, dict):
            return f"{robot_name} stats must be an object"
        for stat_name, max_dim in stat_dims.items():
            stat_error = _validate_minmax_stat(robot_stats, stat_name, max_dim)
            if stat_error:
                return f"{robot_name}.{stat_error}"

    return None


def _configured_dim_or_target(value: Any, target_dim: int) -> int:
    if value is None:
        return target_dim
    try:
        configured_dim = int(value)
    except (TypeError, ValueError):
        return target_dim
    if configured_dim <= 0:
        return target_dim
    return min(configured_dim, target_dim)


def _validate_minmax_stat(robot_stats: dict[str, Any], stat_name: str, target_dim: int) -> str | None:
    stat = robot_stats.get(stat_name)
    if not isinstance(stat, dict):
        return f"{stat_name} must be an object with min/max"

    mins = stat.get("min")
    maxs = stat.get("max")
    min_error = _validate_numeric_vector(mins, f"{stat_name}.min", target_dim)
    if min_error:
        return min_error
    max_error = _validate_numeric_vector(maxs, f"{stat_name}.max", target_dim)
    if max_error:
        return max_error
    if len(mins) != len(maxs):
        return f"{stat_name}.min and max must have the same length"
    for index, (min_value, max_value) in enumerate(zip(mins, maxs)):
        if float(min_value) > float(max_value):
            return f"{stat_name}.min[{index}] must be <= max[{index}]"
    return None


def _validate_numeric_vector(value: Any, label: str, target_dim: int) -> str | None:
    if not isinstance(value, list) or not value:
        return f"{label} must be a non-empty list"
    if len(value) > target_dim:
        return f"{label} length {len(value)} exceeds server target dimension {target_dim}"
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return f"{label}[{index}] must be numeric"
        if not math.isfinite(float(item)):
            return f"{label}[{index}] must be finite"
    return None


def load_yaml_if_available(path: Path, repo_root: Path, report: Report) -> dict[str, Any] | None:
    spec = importlib.util.find_spec("yaml")
    if spec is None:
        report.warn("dataset", "PyYAML is not installed; skipped structured dataset config validation")
        return None

    import yaml  # type: ignore[import-not-found]

    try:
        with path.open("r") as f:
            loaded = yaml.safe_load(f)
    except Exception as exc:
        report.fail("dataset", f"failed to parse dataset config {display_project_path(path, repo_root)}: {exc}")
        return None

    if not isinstance(loaded, dict):
        report.fail("dataset", f"dataset config must be a mapping: {display_project_path(path, repo_root)}")
        return None
    return loaded


def check_dataset_config(config_path: Path, base_dir: Path, strict_data: bool, report: Report, repo_root: Path) -> None:
    if not config_path.exists():
        report.fail("dataset", f"dataset config does not exist: {display_project_path(config_path, repo_root)}")
        return

    config = load_yaml_if_available(config_path, repo_root, report)
    if config is None:
        return

    try:
        dataset_count = validate_dataset_config_structure(config)
        resolved_config = resolve_dataset_config_paths(config, base_dir)
    except (TypeError, ValueError) as exc:
        report.fail("dataset", str(exc))
        return

    if strict_data:
        issues = validate_configured_datasets(config, base_dir, require_videos=True)
        if issues:
            for issue in issues:
                issue_path = display_project_path(issue.path, repo_root)
                if issue.level == "FAIL":
                    report.fail("dataset", f"{issue_path}: {issue.message}")
                else:
                    report.warn("dataset", f"{issue_path}: {issue.message}")
            return
        report.ok("dataset", f"dataset config describes {dataset_count} dataset(s) with valid training data")
        return

    missing_paths: list[str] = []
    for group_name, dataset_name, dataset_config in iter_dataset_entries(resolved_config):
        dataset_path = Path(str(dataset_config["path"]))
        if not dataset_path.exists():
            missing_paths.append(f"{group_name}/{dataset_name}: {display_project_path(dataset_path, repo_root)}")
            continue

    if missing_paths:
        message = "configured dataset paths do not exist: " + "; ".join(missing_paths)
        report.warn("dataset", message)
    else:
        report.ok("dataset", f"dataset config describes {dataset_count} dataset(s)")


def resolve_libero_result_paths(raw_inputs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for raw_input in raw_inputs:
        matches = glob.glob(raw_input, recursive=True)
        candidate_paths = matches if matches else [raw_input]
        for candidate in candidate_paths:
            path = Path(candidate).expanduser()
            if path.is_dir():
                paths.update(path.rglob("*_results.json"))
            elif path.is_file():
                paths.add(path)
            else:
                raise FileNotFoundError(f"LIBERO result path not found: {raw_input}")
    return sorted(path.resolve() for path in paths)


def resolve_libero_manifest_paths(raw_inputs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for raw_input in raw_inputs:
        matches = glob.glob(raw_input, recursive=True)
        candidate_paths = matches if matches else [raw_input]
        for candidate in candidate_paths:
            path = Path(candidate).expanduser()
            if path.is_dir():
                paths.update(path.rglob("run_manifest.json"))
                paths.update(path.rglob("*_run_manifest.json"))
            elif path.is_file():
                paths.add(path)
            else:
                raise FileNotFoundError(f"LIBERO run manifest path not found: {raw_input}")
    return sorted(path.resolve() for path in paths)


def resolve_libero_run_dirs(raw_inputs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for raw_input in raw_inputs:
        matches = glob.glob(raw_input, recursive=True)
        candidate_paths = matches if matches else [raw_input]
        for candidate in candidate_paths:
            path = Path(candidate).expanduser()
            if path.is_dir():
                paths.add(path)
            elif path.exists():
                raise NotADirectoryError(f"LIBERO run path is not a directory: {candidate}")
            else:
                raise FileNotFoundError(f"LIBERO run directory not found: {raw_input}")
    return sorted(path.resolve() for path in paths)


def check_libero_result_file(result_path: Path, report: Report) -> None:
    if not result_path.exists():
        report.fail("libero-result", f"result file does not exist: {result_path}")
        return
    if not result_path.is_file():
        report.fail("libero-result", f"result path is not a file: {result_path}")
        return

    try:
        with result_path.open("r") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        report.fail("libero-result", f"{result_path} is not valid JSON: {exc}")
        return

    if not isinstance(payload, dict):
        report.fail("libero-result", f"{result_path} must contain a JSON object")
        return

    config = payload.get("config")
    if config is not None and not isinstance(config, dict):
        report.fail("libero-result", f"{result_path} config must be an object when present")
        return

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        report.fail("libero-result", f"{result_path} metadata must be an object when present")
        return

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        report.fail("libero-result", f"{result_path} has no summary object")
        return
    summary_error = _validate_libero_summary(summary, "summary")
    if summary_error:
        report.fail("libero-result", f"{result_path}: {summary_error}")
        return

    suites = summary.get("suites", {})
    if suites is not None and not isinstance(suites, dict):
        report.fail("libero-result", f"{result_path} summary.suites must be an object")
        return
    for suite_name, suite_summary in (suites or {}).items():
        if not isinstance(suite_summary, dict):
            report.fail("libero-result", f"{result_path} suite {suite_name!r} summary must be an object")
            return
        suite_error = _validate_libero_summary(suite_summary, f"summary.suites.{suite_name}")
        if suite_error:
            report.fail("libero-result", f"{result_path}: {suite_error}")
            return

    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        report.fail("libero-result", f"{result_path} episodes must be a list")
        return
    for index, episode in enumerate(episodes):
        episode_error = _validate_libero_episode(episode, index)
        if episode_error:
            report.fail("libero-result", f"{result_path}: {episode_error}")
            return

    summary_total = int(summary["total_episodes"])
    if summary_total != len(episodes):
        report.fail(
            "libero-result",
            f"{result_path} summary total_episodes={summary_total} does not match episodes length={len(episodes)}",
        )
        return
    consistency_error = _validate_libero_summary_matches_episodes(summary, episodes, "summary")
    if consistency_error:
        report.fail("libero-result", f"{result_path}: {consistency_error}")
        return

    episode_suite_names = {episode["task_suite"] for episode in episodes}
    summary_suite_names = set((suites or {}).keys())
    if episode_suite_names != summary_suite_names:
        report.fail(
            "libero-result",
            f"{result_path} summary.suites keys {sorted(summary_suite_names)} "
            f"do not match episode task suites {sorted(episode_suite_names)}",
        )
        return
    for suite_name in sorted(episode_suite_names):
        suite_episodes = [episode for episode in episodes if episode["task_suite"] == suite_name]
        suite_consistency_error = _validate_libero_summary_matches_episodes(
            suites[suite_name],
            suite_episodes,
            f"summary.suites.{suite_name}",
        )
        if suite_consistency_error:
            report.fail("libero-result", f"{result_path}: {suite_consistency_error}")
            return

    report.ok("libero-result", f"{result_path} describes {len(episodes)} episode(s)")


def check_libero_manifest_file(manifest_path: Path, report: Report) -> None:
    if not manifest_path.exists():
        report.fail("libero-manifest", f"manifest file does not exist: {manifest_path}")
        return
    if not manifest_path.is_file():
        report.fail("libero-manifest", f"manifest path is not a file: {manifest_path}")
        return

    try:
        with manifest_path.open("r") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        report.fail("libero-manifest", f"{manifest_path} is not valid JSON: {exc}")
        return

    if not isinstance(payload, dict):
        report.fail("libero-manifest", f"{manifest_path} must contain a JSON object")
        return

    manifest_error = _validate_libero_manifest(payload)
    if manifest_error:
        report.fail("libero-manifest", f"{manifest_path}: {manifest_error}")
        return

    run_kind = payload["run_kind"]
    ckpt_name = payload["libero"]["HIMEM_LIBERO_CKPT_NAME"]
    report.ok("libero-manifest", f"{manifest_path} describes {run_kind} run {ckpt_name!r}")


def check_libero_run_dir(run_dir: Path, report: Report) -> None:
    if not run_dir.exists():
        report.fail("libero-run", f"run directory does not exist: {run_dir}")
        return
    if not run_dir.is_dir():
        report.fail("libero-run", f"run path is not a directory: {run_dir}")
        return

    manifest_path = _discover_single_run_manifest(run_dir)
    if manifest_path is None:
        report.fail("libero-run", f"no run manifest found directly under {run_dir}")
        return
    if isinstance(manifest_path, list):
        rendered = ", ".join(str(path.name) for path in manifest_path)
        report.fail("libero-run", f"multiple run manifests found directly under {run_dir}: {rendered}")
        return

    failure_count = _failure_count(report)
    check_libero_manifest_file(manifest_path, report)
    if _failure_count(report) > failure_count:
        return

    manifest = _load_json_object(manifest_path)
    if manifest is None:
        report.fail("libero-run", f"{manifest_path} could not be reloaded after manifest validation")
        return

    libero = manifest["libero"]
    manifest_reference = _resolve_artifact_reference(run_dir, str(libero["HIMEM_LIBERO_MANIFEST_FILE"]))
    if manifest_reference != manifest_path.resolve():
        report.fail(
            "libero-run",
            f"{manifest_path} records HIMEM_LIBERO_MANIFEST_FILE={manifest_reference}, expected {manifest_path.resolve()}",
        )
        return

    result_path = _resolve_artifact_reference(run_dir, str(libero["HIMEM_LIBERO_RESULT_FILE"]))
    if not result_path.exists():
        report.fail("libero-run", f"referenced result file does not exist: {result_path}")
        return

    failure_count = _failure_count(report)
    check_libero_result_file(result_path, report)
    if _failure_count(report) > failure_count:
        return

    result_payload = _load_json_object(result_path)
    if result_payload is None:
        report.fail("libero-run", f"{result_path} could not be reloaded after result validation")
        return

    consistency_error = _validate_libero_run_consistency(manifest, result_payload)
    if consistency_error:
        report.fail("libero-run", f"{run_dir}: {consistency_error}")
        return

    report.ok("libero-run", f"{run_dir} has consistent manifest and result artifacts")


def _validate_libero_summary(summary: dict[str, Any], label: str) -> str | None:
    required_fields = (*LIBERO_SUMMARY_COUNT_FIELDS, *LIBERO_SUMMARY_FLOAT_FIELDS)
    missing = [field for field in required_fields if field not in summary]
    if missing:
        return f"{label} missing fields: {', '.join(missing)}"
    for field in LIBERO_SUMMARY_COUNT_FIELDS:
        value = summary[field]
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{label}.{field} must be an integer"
        if value < 0:
            return f"{label}.{field} must be non-negative"
    for field in LIBERO_SUMMARY_FLOAT_FIELDS:
        value = summary[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{label}.{field} must be numeric"
        if value < 0:
            return f"{label}.{field} must be non-negative"
    if not 0 <= float(summary["success_rate"]) <= 1:
        return f"{label}.success_rate must be between 0 and 1"
    total = int(summary["total_episodes"])
    successful = int(summary["successful_episodes"])
    failed = int(summary["failed_episodes"])
    if successful + failed != total:
        return f"{label} successful_episodes + failed_episodes must equal total_episodes"
    return None


def _validate_libero_episode(episode: Any, index: int) -> str | None:
    if not isinstance(episode, dict):
        return f"episodes[{index}] must be an object"
    missing = [field for field in LIBERO_EPISODE_REQUIRED_FIELDS if field not in episode]
    if missing:
        return f"episodes[{index}] missing fields: {', '.join(missing)}"
    if not isinstance(episode["task_suite"], str) or not episode["task_suite"]:
        return f"episodes[{index}].task_suite must be a non-empty string"
    if not isinstance(episode["task_description"], str):
        return f"episodes[{index}].task_description must be a string"
    if not isinstance(episode["success"], bool):
        return f"episodes[{index}].success must be a boolean"
    for field in ("task_id", "episode_id", "decision_steps", "control_steps"):
        value = episode[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return f"episodes[{index}].{field} must be a non-negative integer"
    if not episode["success"] and not str(episode.get("failure_reason", "")).strip():
        return f"episodes[{index}].failure_reason is required for failed episodes"
    return None


def _validate_libero_manifest(payload: dict[str, Any]) -> str | None:
    if payload.get("schema_version") != 1:
        return "schema_version must be 1"
    if payload.get("run_kind") not in {"smoke", "eval"}:
        return "run_kind must be 'smoke' or 'eval'"

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return "metadata must be an object"
    metadata_error = _validate_run_metadata(metadata)
    if metadata_error:
        return f"metadata.{metadata_error}"

    libero = payload.get("libero")
    if not isinstance(libero, dict):
        return "libero must be an object"
    libero_error = _validate_manifest_libero_env(libero)
    if libero_error:
        return f"libero.{libero_error}"

    return None


def _validate_run_metadata(metadata: dict[str, Any]) -> str | None:
    for field in ("created_at_utc", "cwd", "command", "platform", "hostname"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            return f"{field} must be a non-empty string"
    if not isinstance(metadata.get("argv"), list) or not all(isinstance(item, str) for item in metadata["argv"]):
        return "argv must be a list of strings"

    python_info = metadata.get("python")
    if not isinstance(python_info, dict):
        return "python must be an object"
    for field in ("executable", "version"):
        if not isinstance(python_info.get(field), str) or not python_info[field]:
            return f"python.{field} must be a non-empty string"

    git_info = metadata.get("git")
    if not isinstance(git_info, dict):
        return "git must be an object"
    for field in ("repo_root", "commit", "branch"):
        if not isinstance(git_info.get(field), str) or not git_info[field]:
            return f"git.{field} must be a non-empty string"
    if not isinstance(git_info.get("is_dirty"), bool):
        return "git.is_dirty must be a boolean"

    environment = metadata.get("environment")
    if not isinstance(environment, dict):
        return "environment must be an object"
    env_error = _validate_no_sensitive_environment(environment)
    if env_error:
        return f"environment.{env_error}"
    return None


def _validate_manifest_libero_env(libero: dict[str, Any]) -> str | None:
    missing = [field for field in LIBERO_MANIFEST_REQUIRED_ENV if field not in libero]
    if missing:
        return f"missing fields: {', '.join(missing)}"
    for key, value in libero.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return "all keys and values must be strings"
        if not value:
            return f"{key} must be non-empty"
    env_error = _validate_no_sensitive_environment(libero)
    if env_error:
        return env_error

    for field in ("HIMEM_LIBERO_EPISODES", "HIMEM_LIBERO_HORIZON"):
        if _parse_positive_int(libero[field]) is None:
            return f"{field} must be a positive integer string"
    if _parse_non_negative_int(libero["HIMEM_LIBERO_TASK_LIMIT"]) is None:
        return "HIMEM_LIBERO_TASK_LIMIT must be a non-negative integer string"
    max_steps = [item.strip() for item in libero["HIMEM_LIBERO_MAX_STEPS"].split(",")]
    if not max_steps or any(_parse_positive_int(item) is None for item in max_steps):
        return "HIMEM_LIBERO_MAX_STEPS must be a comma-separated list of positive integers"
    task_suites = [item.strip() for item in libero["HIMEM_LIBERO_TASK_SUITES"].split(",")]
    if not task_suites or any(not item for item in task_suites):
        return "HIMEM_LIBERO_TASK_SUITES must be a comma-separated list of suite names"
    server_uri = libero["HIMEM_SERVER_URI"]
    if not (server_uri.startswith("ws://") or server_uri.startswith("wss://")):
        return "HIMEM_SERVER_URI must start with ws:// or wss://"
    return None


def _discover_single_run_manifest(run_dir: Path) -> Path | list[Path] | None:
    candidates = []
    direct_manifest = run_dir / "run_manifest.json"
    if direct_manifest.exists():
        candidates.append(direct_manifest)
    candidates.extend(sorted(run_dir.glob("*_run_manifest.json")))
    unique_candidates = sorted({path.resolve() for path in candidates})
    if not unique_candidates:
        return None
    if len(unique_candidates) > 1:
        return unique_candidates
    return unique_candidates[0]


def _resolve_artifact_reference(run_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _failure_count(report: Report) -> int:
    return sum(1 for result in report.results if result.level == "FAIL")


def _validate_libero_run_consistency(
    manifest: dict[str, Any],
    result_payload: dict[str, Any],
) -> str | None:
    manifest_libero = manifest["libero"]
    manifest_metadata = manifest["metadata"]
    result_config = result_payload.get("config", {})
    result_metadata = result_payload.get("metadata", {})

    if isinstance(result_config, dict):
        result_ckpt_name = result_config.get("ckpt_name")
        manifest_ckpt_name = manifest_libero["HIMEM_LIBERO_CKPT_NAME"]
        if result_ckpt_name and str(result_ckpt_name) != manifest_ckpt_name:
            return (
                f"result config ckpt_name={result_ckpt_name!r} does not match "
                f"manifest HIMEM_LIBERO_CKPT_NAME={manifest_ckpt_name!r}"
            )

    manifest_git = manifest_metadata.get("git", {}) if isinstance(manifest_metadata, dict) else {}
    result_git = result_metadata.get("git", {}) if isinstance(result_metadata, dict) else {}
    if isinstance(manifest_git, dict) and isinstance(result_git, dict):
        for field in ("commit", "is_dirty"):
            manifest_value = manifest_git.get(field)
            result_value = result_git.get(field)
            if manifest_value is not None and result_value is not None and manifest_value != result_value:
                return f"result metadata git.{field}={result_value!r} does not match manifest git.{field}={manifest_value!r}"

    return None


def _validate_no_sensitive_environment(environment: dict[str, Any]) -> str | None:
    for key in environment:
        if any(fragment in key.upper() for fragment in SENSITIVE_ENV_FRAGMENTS):
            return f"{key} must not be recorded"
    return None


def _parse_positive_int(value: str) -> int | None:
    parsed = _parse_non_negative_int(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _parse_non_negative_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    if str(parsed) != value or parsed < 0:
        return None
    return parsed


def _validate_libero_summary_matches_episodes(
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    label: str,
) -> str | None:
    expected = _compute_episode_summary(episodes)
    for field in LIBERO_SUMMARY_COUNT_FIELDS:
        if int(summary[field]) != expected[field]:
            return f"{label}.{field}={summary[field]} does not match episode-derived value {expected[field]}"
    for field in LIBERO_SUMMARY_FLOAT_FIELDS:
        if not math.isclose(float(summary[field]), expected[field], rel_tol=1e-9, abs_tol=1e-9):
            return f"{label}.{field}={summary[field]} does not match episode-derived value {expected[field]}"
    return None


def _compute_episode_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(episodes)
    successful = sum(1 for episode in episodes if episode["success"])
    failed = total - successful
    decision_steps = [int(episode["decision_steps"]) for episode in episodes]
    control_steps = [int(episode["control_steps"]) for episode in episodes]
    success_decision_steps = [
        int(episode["decision_steps"]) for episode in episodes if episode["success"]
    ]
    return {
        "total_episodes": total,
        "successful_episodes": successful,
        "failed_episodes": failed,
        "success_rate": successful / total if total else 0.0,
        "average_decision_steps": _mean(decision_steps),
        "average_control_steps": _mean(control_steps),
        "average_success_decision_steps": _mean(success_decision_steps),
    }


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight HiMem-Bridge-VLA repository preflight checks.")
    parser.add_argument("--repo-root", default=".", help="Repository root to check.")
    parser.add_argument("--checkpoint", help="Optional HiMem-Bridge-VLA checkpoint directory to validate.")
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets/simulation.yaml",
        help="Dataset config to validate. Set to empty string to skip.",
    )
    parser.add_argument(
        "--dataset-base-dir",
        default=".",
        help="Base directory for relative dataset paths in the dataset config.",
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Fail when configured dataset paths or required dataset files are missing.",
    )
    parser.add_argument(
        "--check-imports",
        choices=("none", "himem", "libero", "all"),
        default="none",
        help="Optionally require runtime packages to be importable.",
    )
    parser.add_argument(
        "--libero-result",
        action="append",
        default=[],
        help="Optional LIBERO result JSON file, directory, or glob to validate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--libero-manifest",
        action="append",
        default=[],
        help="Optional LIBERO run manifest file, directory, or glob to validate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--libero-run-dir",
        action="append",
        default=[],
        help="Optional LIBERO run directory or glob to validate as a complete run artifact. Can be passed multiple times.",
    )
    parser.add_argument("--skip-shell-syntax", action="store_true", help="Skip bash -n checks for scripts/**/*.sh.")
    return parser.parse_args(argv)


def run_preflight(args: argparse.Namespace) -> Report:
    report = Report()
    repo_root = repo_root_from_script() if args.repo_root == "." else project_path(args.repo_root, repo_root_from_script())

    check_repo_layout(repo_root, report)
    check_script_permissions(repo_root, report)
    if not args.skip_shell_syntax:
        check_shell_syntax(repo_root, report)

    if args.dataset_config:
        dataset_config = resolve_repo_path(repo_root, args.dataset_config)
        dataset_base_dir = resolve_repo_path(repo_root, args.dataset_base_dir)
        check_dataset_config(dataset_config, dataset_base_dir, bool(args.strict_data), report, repo_root)

    if args.checkpoint:
        check_checkpoint_dir(resolve_repo_path(repo_root, args.checkpoint), report)

    libero_result_inputs = getattr(args, "libero_result", [])
    if libero_result_inputs:
        try:
            libero_result_paths = resolve_libero_result_paths(libero_result_inputs)
        except FileNotFoundError as exc:
            report.fail("libero-result", str(exc))
        else:
            if not libero_result_paths:
                report.fail("libero-result", "no LIBERO result files matched")
            for result_path in libero_result_paths:
                check_libero_result_file(result_path, report)

    libero_manifest_inputs = getattr(args, "libero_manifest", [])
    if libero_manifest_inputs:
        try:
            libero_manifest_paths = resolve_libero_manifest_paths(libero_manifest_inputs)
        except FileNotFoundError as exc:
            report.fail("libero-manifest", str(exc))
        else:
            if not libero_manifest_paths:
                report.fail("libero-manifest", "no LIBERO run manifest files matched")
            for manifest_path in libero_manifest_paths:
                check_libero_manifest_file(manifest_path, report)

    libero_run_dir_inputs = getattr(args, "libero_run_dir", [])
    if libero_run_dir_inputs:
        try:
            libero_run_dirs = resolve_libero_run_dirs(libero_run_dir_inputs)
        except (FileNotFoundError, NotADirectoryError) as exc:
            report.fail("libero-run", str(exc))
        else:
            if not libero_run_dirs:
                report.fail("libero-run", "no LIBERO run directories matched")
            for run_dir in libero_run_dirs:
                check_libero_run_dir(run_dir, report)

    if args.check_imports in ("himem", "all"):
        check_imports(HIMEM_RUNTIME_IMPORTS, report, "himem-imports")
    if args.check_imports in ("libero", "all"):
        check_imports(LIBERO_RUNTIME_IMPORTS, report, "libero-imports")

    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_preflight(args)
    report.print()
    return 1 if report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
