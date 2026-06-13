from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_calvin_run_manifest.py"


def load_manifest_module():
    spec = importlib.util.spec_from_file_location("write_calvin_run_manifest", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_calvin_manifest_tracks_calvin_environment_and_redacts_secrets(tmp_path):
    module = load_manifest_module()

    manifest = module.build_manifest(
        run_kind="eval",
        repo_root=tmp_path,
        environ={
            "HIMEM_CALVIN_CKPT_NAME": "calvin_eval",
            "HIMEM_CALVIN_NUM_SEQUENCES": "1000",
            "HIMEM_SERVER_URI": "ws://127.0.0.1:9000",
            "HIMEM_API_TOKEN": "secret",
            "CALVIN_ROOT": "/data/calvin",
            "UNRELATED": "ignored",
        },
        argv=["write_calvin_run_manifest.py"],
    )

    assert manifest["schema_version"] == 1
    assert manifest["run_kind"] == "eval"
    assert manifest["calvin"]["HIMEM_CALVIN_CKPT_NAME"] == "calvin_eval"
    assert manifest["calvin"]["CALVIN_ROOT"] == "/data/calvin"
    assert "HIMEM_API_TOKEN" not in manifest["metadata"]["environment"]
    assert "UNRELATED" not in manifest["metadata"]["environment"]


def test_write_calvin_manifest_creates_parent_dirs(tmp_path):
    module = load_manifest_module()
    output = tmp_path / "nested" / "manifest.json"

    written = module.write_manifest(output, {"schema_version": 1, "run_kind": "smoke", "calvin": {}})

    assert written == output
    assert json.loads(output.read_text())["run_kind"] == "smoke"
