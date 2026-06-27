from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
from himem_bridge_vla.cli.setup import install_rmbench_policy_adapter

import pytest


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "setup" / "install_rmbench_policy_adapter.py"


def load_module():
    return install_rmbench_policy_adapter


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
