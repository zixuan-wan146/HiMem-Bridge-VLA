from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_requirements.py"
    spec = importlib.util.spec_from_file_location("audit_requirements", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_policy(path: Path, policy: dict) -> Path:
    path.write_text(json.dumps(policy))
    return path


def test_audit_accepts_pinned_requirements(tmp_path: Path):
    module = load_module()
    (tmp_path / "requirements-dev.txt").write_text("pytest==6.2.5\nnumpy==1.26.4\n")
    policy_path = write_policy(
        tmp_path / "requirements-policy.json",
        {"files": {"requirements-dev.txt": {"allow_unpinned": {}}}},
    )

    report = module.audit_requirements(tmp_path, policy_path)

    assert not report.has_failures
    assert any(message.level == "OK" and "exact pins" in message.message for message in report.messages)


def test_audit_rejects_unallowlisted_unpinned_requirements(tmp_path: Path):
    module = load_module()
    (tmp_path / "requirements-dev.txt").write_text("pytest\n")
    policy_path = write_policy(
        tmp_path / "requirements-policy.json",
        {"files": {"requirements-dev.txt": {"allow_unpinned": {}}}},
    )

    report = module.audit_requirements(tmp_path, policy_path)

    assert report.has_failures
    assert any("not allowlisted" in message.message for message in report.messages)


def test_audit_allows_documented_unpinned_requirements(tmp_path: Path):
    module = load_module()
    (tmp_path / "requirements-dev.txt").write_text("pytest\n")
    policy_path = write_policy(
        tmp_path / "requirements-policy.json",
        {"files": {"requirements-dev.txt": {"allow_unpinned": {"pytest": "kept floating for local smoke"}}}},
    )

    report = module.audit_requirements(tmp_path, policy_path)

    assert not report.has_failures
    assert any(message.level == "WARN" and "intentionally unpinned" in message.message for message in report.messages)


def test_audit_rejects_stale_allowlist_entries(tmp_path: Path):
    module = load_module()
    (tmp_path / "requirements-dev.txt").write_text("pytest==6.2.5\n")
    policy_path = write_policy(
        tmp_path / "requirements-policy.json",
        {"files": {"requirements-dev.txt": {"allow_unpinned": {"pytest": "old debt"}}}},
    )

    report = module.audit_requirements(tmp_path, policy_path)

    assert report.has_failures
    assert any("stale unpinned dependency" in message.message for message in report.messages)


def test_audit_rejects_requirements_files_missing_from_policy(tmp_path: Path):
    module = load_module()
    (tmp_path / "requirements-dev.txt").write_text("pytest==6.2.5\n")
    write_policy(
        tmp_path / "requirements-policy.json",
        {"files": {}},
    )

    report = module.audit_requirements(tmp_path, tmp_path / "requirements-policy.json")

    assert report.has_failures
    assert any("not covered" in message.message for message in report.messages)
