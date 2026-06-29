from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "automation" / "watch_stage2_train_eval_shutdown.py"


def _load_watch_module():
    spec = importlib.util.spec_from_file_location("watch_stage2_train_eval_shutdown", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_orchestration_defaults_do_not_shutdown_or_retry_same_failure():
    module = _load_watch_module()

    args = module.parse_args([])

    assert args.shutdown is False
    assert args.max_attempts == 1


def test_eval_checkpoint_resolution_requires_step_best_model(tmp_path):
    module = _load_watch_module()
    save_dir = tmp_path / "save"
    step_final = save_dir / "step_final"
    step_final.mkdir(parents=True)
    (step_final / "model.pt").write_bytes(b"final")
    args = argparse.Namespace(stage2_save_dir=str(save_dir), eval_checkpoint_tag="step_best")

    with pytest.raises(FileNotFoundError, match="required best checkpoint"):
        module.resolve_eval_checkpoint_dir(args)

    step_best = save_dir / "step_best"
    step_best.mkdir()
    (step_best / "model.pt").write_bytes(b"best")

    assert module.resolve_eval_checkpoint_dir(args) == step_best


def test_shutdown_invokes_remote_shutdown_script_through_bash(tmp_path, monkeypatch):
    module = _load_watch_module()
    calls = []

    class FakeState:
        repo_root = tmp_path

        def record(self, *args, **kwargs):
            calls.append((args, kwargs))

    def fake_popen(command, *, cwd):
        calls.append((command, cwd))
        return object()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    module.shutdown(FakeState(), reason="test")

    assert (["bash", "/usr/bin/shutdown"], tmp_path) in calls
