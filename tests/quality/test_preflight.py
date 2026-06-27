from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
from himem_bridge_vla.cli.quality import preflight
import argparse
import json
from pathlib import Path

import pytest


def load_preflight_module():
    return preflight


def result_levels(report):
    return [result.level for result in report.results]


def valid_libero_result_payload() -> dict:
    return {
        "config": {"ckpt_name": "smoke"},
        "metadata": {"git": {"commit": "abc123", "is_dirty": False}},
        "summary": {
            "total_episodes": 2,
            "successful_episodes": 1,
            "failed_episodes": 1,
            "success_rate": 0.5,
            "average_decision_steps": 2.5,
            "average_control_steps": 10.0,
            "average_success_decision_steps": 2.0,
            "suites": {
                "libero_spatial": {
                    "total_episodes": 2,
                    "successful_episodes": 1,
                    "failed_episodes": 1,
                    "success_rate": 0.5,
                    "average_decision_steps": 2.5,
                    "average_control_steps": 10.0,
                    "average_success_decision_steps": 2.0,
                }
            },
        },
        "episodes": [
            {
                "task_suite": "libero_spatial",
                "task_id": 0,
                "episode_id": 0,
                "task_description": "pick up the object",
                "success": True,
                "decision_steps": 2,
                "control_steps": 14,
                "failure_reason": "",
                "video_path": "task1_episode1.mp4",
            },
            {
                "task_suite": "libero_spatial",
                "task_id": 0,
                "episode_id": 1,
                "task_description": "pick up the object",
                "success": False,
                "decision_steps": 3,
                "control_steps": 6,
                "failure_reason": "max_steps_exhausted",
                "video_path": "task1_episode2.mp4",
            },
        ],
    }


def valid_libero_manifest_payload() -> dict:
    return {
        "schema_version": 1,
        "run_kind": "smoke",
        "metadata": {
            "created_at_utc": "2026-06-10T00:00:00Z",
            "cwd": "run_outputs/himem",
            "argv": ["scripts/report/write_libero_run_manifest.py"],
            "command": "scripts/report/write_libero_run_manifest.py",
            "python": {
                "executable": "python3",
                "version": "3.10.0",
            },
            "platform": "Linux",
            "hostname": "host",
            "git": {
                "repo_root": "run_outputs/himem",
                "commit": "abc123",
                "branch": "main",
                "is_dirty": False,
            },
            "environment": {
                "HIMEM_LIBERO_CKPT_NAME": "smoke",
            },
        },
        "libero": {
            "HIMEM_LIBERO_CKPT_NAME": "smoke",
            "HIMEM_LIBERO_LOG_DIR": "run_outputs/himem/run/logs",
            "HIMEM_LIBERO_VIDEO_DIR": "run_outputs/himem/run/videos",
            "HIMEM_LIBERO_LOG_FILE": "run_outputs/himem/run/logs/smoke.txt",
            "HIMEM_LIBERO_RESULT_FILE": "run_outputs/himem/run/results/smoke_results.json",
            "HIMEM_LIBERO_MANIFEST_FILE": "run_outputs/himem/run/run_manifest.json",
            "HIMEM_LIBERO_TASK_SUITES": "libero_spatial",
            "HIMEM_LIBERO_TASK_LIMIT": "1",
            "HIMEM_LIBERO_EPISODES": "1",
            "HIMEM_LIBERO_MAX_STEPS": "1",
            "HIMEM_LIBERO_HORIZON": "1",
            "HIMEM_SERVER_URI": "ws://127.0.0.1:9000",
            "HIMEM_MUJOCO_GL": "osmesa",
        },
    }


def valid_checkpoint_config() -> dict:
    return {
        "horizon": 14,
        "per_action_dim": 7,
        "state_dim": 7,
        "action_dim": 98,
    }


def valid_norm_stats() -> dict:
    return {
        "libero": {
            "observation.state": {
                "min": [0.0, -1.0, -2.0],
                "max": [1.0, 2.0, 3.0],
            },
            "action": {
                "min": [-1.0] * 7,
                "max": [1.0] * 7,
            },
        }
    }


def write_valid_libero_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "libero_run"
    result_file = run_dir / "results" / "smoke_results.json"
    manifest_file = run_dir / "run_manifest.json"
    result_file.parent.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "videos").mkdir()

    result_payload = valid_libero_result_payload()
    manifest_payload = valid_libero_manifest_payload()
    manifest_payload["libero"].update(
        {
            "HIMEM_LIBERO_LOG_DIR": str(run_dir / "logs"),
            "HIMEM_LIBERO_VIDEO_DIR": str(run_dir / "videos"),
            "HIMEM_LIBERO_LOG_FILE": str(run_dir / "logs" / "smoke.txt"),
            "HIMEM_LIBERO_RESULT_FILE": str(result_file),
            "HIMEM_LIBERO_MANIFEST_FILE": str(manifest_file),
        }
    )
    result_file.write_text(json.dumps(result_payload))
    manifest_file.write_text(json.dumps(manifest_payload))
    return run_dir


def test_checkpoint_validation_accepts_required_files(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps(valid_checkpoint_config()))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(valid_norm_stats()))
    (ckpt_dir / "model.pt").write_bytes(b"checkpoint")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert result_levels(report) == ["OK"]


def test_checkpoint_validation_accepts_multi_robot_norm_stats(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    stats = valid_norm_stats()
    stats["other_robot"] = {
        "observation.state": {"min": [0.0] * 7, "max": [1.0] * 7},
        "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
    }
    (ckpt_dir / "config.json").write_text(json.dumps(valid_checkpoint_config()))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(stats))
    (ckpt_dir / "model.pt").write_bytes(b"checkpoint")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert result_levels(report) == ["OK"]


def test_checkpoint_validation_rejects_missing_weight_file(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text("{}")
    (ckpt_dir / "norm_stats.json").write_text("{}")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert report.has_failures
    assert "model.pt" in report.results[-1].message


def test_checkpoint_validation_rejects_inconsistent_action_dim(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    config = valid_checkpoint_config()
    config["action_dim"] = 99
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(valid_norm_stats()))
    (ckpt_dir / "model.pt").write_bytes(b"checkpoint")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert report.has_failures
    assert "action_dim" in report.results[-1].message


def test_checkpoint_validation_rejects_invalid_norm_stats(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    stats = valid_norm_stats()
    stats["libero"]["action"]["max"] = [1.0, 2.0]
    (ckpt_dir / "config.json").write_text(json.dumps(valid_checkpoint_config()))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(stats))
    (ckpt_dir / "model.pt").write_bytes(b"checkpoint")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert report.has_failures
    assert "same length" in report.results[-1].message


def test_checkpoint_validation_rejects_stats_longer_than_configured_action_dim(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    stats = valid_norm_stats()
    stats["libero"]["action"]["min"] = [-1.0] * 8
    stats["libero"]["action"]["max"] = [1.0] * 8
    (ckpt_dir / "config.json").write_text(json.dumps(valid_checkpoint_config()))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(stats))
    (ckpt_dir / "model.pt").write_bytes(b"checkpoint")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert report.has_failures
    assert "exceeds server target dimension 7" in report.results[-1].message


def test_libero_result_validation_accepts_valid_result_file(tmp_path):
    preflight = load_preflight_module()
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(valid_libero_result_payload()))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert result_levels(report) == ["OK"]


def test_libero_result_validation_requires_failure_reason_for_failed_episode(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_result_payload()
    payload["episodes"][1]["failure_reason"] = ""
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert report.has_failures
    assert "failure_reason" in report.results[-1].message


def test_libero_result_validation_rejects_summary_episode_count_mismatch(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_result_payload()
    payload["summary"]["total_episodes"] = 3
    payload["summary"]["successful_episodes"] = 2
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert report.has_failures
    assert "does not match episodes length" in report.results[-1].message


def test_libero_result_validation_rejects_summary_metric_mismatch(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_result_payload()
    payload["summary"]["success_rate"] = 1.0
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert report.has_failures
    assert "summary.success_rate" in report.results[-1].message


def test_libero_result_validation_rejects_suite_keys_mismatch(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_result_payload()
    payload["summary"]["suites"] = {}
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert report.has_failures
    assert "summary.suites keys" in report.results[-1].message


def test_run_preflight_accepts_libero_result_directory(tmp_path):
    preflight = load_preflight_module()
    repo_root = find_repo_root(__file__)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "smoke_results.json").write_text(json.dumps(valid_libero_result_payload()))
    args = argparse.Namespace(
        repo_root=str(repo_root),
        checkpoint=None,
        dataset_config="",
        dataset_base_dir="himem_bridge_vla",
        strict_data=False,
        check_imports="none",
        libero_result=[str(result_dir)],
        libero_manifest=[],
        libero_run_dir=[],
        skip_shell_syntax=True,
    )

    report = preflight.run_preflight(args)

    assert not report.has_failures
    assert any(result.name == "libero-result" and result.level == "OK" for result in report.results)


def test_libero_manifest_validation_accepts_valid_manifest(tmp_path):
    preflight = load_preflight_module()
    manifest_file = tmp_path / "run_manifest.json"
    manifest_file.write_text(json.dumps(valid_libero_manifest_payload()))

    report = preflight.Report()
    preflight.check_libero_manifest_file(manifest_file, report)

    assert result_levels(report) == ["OK"]


def test_libero_manifest_validation_rejects_missing_libero_field(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_manifest_payload()
    del payload["libero"]["HIMEM_LIBERO_RESULT_FILE"]
    manifest_file = tmp_path / "run_manifest.json"
    manifest_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_manifest_file(manifest_file, report)

    assert report.has_failures
    assert "missing fields" in report.results[-1].message


def test_libero_manifest_validation_rejects_secret_environment(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_manifest_payload()
    payload["metadata"]["environment"]["HIMEM_TOKEN"] = "secret"
    manifest_file = tmp_path / "run_manifest.json"
    manifest_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_manifest_file(manifest_file, report)

    assert report.has_failures
    assert "must not be recorded" in report.results[-1].message


def test_libero_manifest_validation_rejects_invalid_numeric_config(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_manifest_payload()
    payload["libero"]["HIMEM_LIBERO_HORIZON"] = "0"
    manifest_file = tmp_path / "run_manifest.json"
    manifest_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_manifest_file(manifest_file, report)

    assert report.has_failures
    assert "HIMEM_LIBERO_HORIZON" in report.results[-1].message


def test_run_preflight_accepts_libero_manifest_directory(tmp_path):
    preflight = load_preflight_module()
    repo_root = find_repo_root(__file__)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps(valid_libero_manifest_payload()))
    args = argparse.Namespace(
        repo_root=str(repo_root),
        checkpoint=None,
        dataset_config="",
        dataset_base_dir="himem_bridge_vla",
        strict_data=False,
        check_imports="none",
        libero_result=[],
        libero_manifest=[str(run_dir)],
        libero_run_dir=[],
        skip_shell_syntax=True,
    )

    report = preflight.run_preflight(args)

    assert not report.has_failures
    assert any(result.name == "libero-manifest" and result.level == "OK" for result in report.results)


def test_libero_run_dir_validation_accepts_consistent_run_dir(tmp_path):
    preflight = load_preflight_module()
    run_dir = write_valid_libero_run_dir(tmp_path)

    report = preflight.Report()
    preflight.check_libero_run_dir(run_dir, report)

    assert not report.has_failures
    assert any(result.name == "libero-run" and result.level == "OK" for result in report.results)


def test_libero_run_dir_validation_rejects_missing_result_file(tmp_path):
    preflight = load_preflight_module()
    run_dir = write_valid_libero_run_dir(tmp_path)
    (run_dir / "results" / "smoke_results.json").unlink()

    report = preflight.Report()
    preflight.check_libero_run_dir(run_dir, report)

    assert report.has_failures
    assert "referenced result file does not exist" in report.results[-1].message


def test_libero_run_dir_validation_rejects_manifest_path_mismatch(tmp_path):
    preflight = load_preflight_module()
    run_dir = write_valid_libero_run_dir(tmp_path)
    manifest_file = run_dir / "run_manifest.json"
    payload = json.loads(manifest_file.read_text())
    payload["libero"]["HIMEM_LIBERO_MANIFEST_FILE"] = str(run_dir / "other_manifest.json")
    manifest_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_run_dir(run_dir, report)

    assert report.has_failures
    assert "HIMEM_LIBERO_MANIFEST_FILE" in report.results[-1].message


def test_libero_run_dir_validation_rejects_git_mismatch(tmp_path):
    preflight = load_preflight_module()
    run_dir = write_valid_libero_run_dir(tmp_path)
    result_file = run_dir / "results" / "smoke_results.json"
    payload = json.loads(result_file.read_text())
    payload["metadata"]["git"]["commit"] = "different"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_run_dir(run_dir, report)

    assert report.has_failures
    assert "git.commit" in report.results[-1].message


def test_run_preflight_accepts_libero_run_dir(tmp_path):
    preflight = load_preflight_module()
    repo_root = find_repo_root(__file__)
    run_dir = write_valid_libero_run_dir(tmp_path)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        checkpoint=None,
        dataset_config="",
        dataset_base_dir="himem_bridge_vla",
        strict_data=False,
        check_imports="none",
        libero_result=[],
        libero_manifest=[],
        libero_run_dir=[str(run_dir)],
        skip_shell_syntax=True,
    )

    report = preflight.run_preflight(args)

    assert not report.has_failures
    assert any(result.name == "libero-run" and result.level == "OK" for result in report.results)


def test_dataset_config_validation_accepts_repo_default_without_strict_data():
    preflight = load_preflight_module()
    repo_root = find_repo_root(__file__)
    report = preflight.Report()

    preflight.check_dataset_config(
        repo_root / "configs" / "datasets" / "simulation.yaml",
        repo_root,
        strict_data=False,
        report=report,
        repo_root=repo_root,
    )

    assert not report.has_failures


def test_dataset_config_strict_data_rejects_missing_dataset_path(tmp_path):
    pytest.importorskip("yaml")
    preflight = load_preflight_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "max_action_dim: 4",
                "max_state_dim: 4",
                "max_views: 1",
                "data_groups:",
                "  test_arm:",
                "    missing_dataset:",
                "      path: missing",
                "",
            ]
        )
    )
    report = preflight.Report()

    preflight.check_dataset_config(config_path, tmp_path, strict_data=True, report=report, repo_root=tmp_path)

    assert report.has_failures
    assert "path does not exist" in report.results[-1].message


def test_run_preflight_default_has_no_failures():
    preflight = load_preflight_module()
    repo_root = find_repo_root(__file__)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        checkpoint=None,
        dataset_config="configs/datasets/simulation.yaml",
        dataset_base_dir=".",
        strict_data=False,
        check_imports="none",
        libero_result=[],
        libero_manifest=[],
        libero_run_dir=[],
        skip_shell_syntax=False,
    )

    report = preflight.run_preflight(args)

    assert not report.has_failures
