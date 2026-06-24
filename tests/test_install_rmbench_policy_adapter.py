from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "install_rmbench_policy_adapter.py"


def load_module():
    spec = importlib.util.spec_from_file_location("install_rmbench_policy_adapter", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_install_policy_adapter_copies_official_policy_package(tmp_path):
    module = load_module()
    rmbench_root = tmp_path / "RMBench"
    (rmbench_root / "policy").mkdir(parents=True)

    destination = module.install_policy_adapter(rmbench_root)

    assert destination == rmbench_root / "policy" / "HiMemBridgeVLA"
    assert (destination / "__init__.py").exists()
    assert (destination / "deploy_policy.py").exists()
    assert (destination / "deploy_policy.yml").exists()


def test_install_policy_adapter_refuses_existing_destination_without_force(tmp_path):
    module = load_module()
    rmbench_root = tmp_path / "RMBench"
    destination = rmbench_root / "policy" / "HiMemBridgeVLA"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="use --force"):
        module.install_policy_adapter(rmbench_root)

