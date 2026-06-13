from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "check_runtime_environment.py"
    spec = importlib.util.spec_from_file_location("check_runtime_environment", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_major_version_parses_simple_versions():
    module = load_module()

    assert module._major_version("1.26.4") == 1
    assert module._major_version("2.2.6") == 2
    assert module._major_version("not-a-version") is None
