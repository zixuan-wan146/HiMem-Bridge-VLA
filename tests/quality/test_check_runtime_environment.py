from __future__ import annotations

from himem_bridge_vla.cli.quality import check_runtime_environment


def load_module():
    return check_runtime_environment


def test_major_version_parses_simple_versions():
    module = load_module()

    assert module._major_version("1.26.4") == 1
    assert module._major_version("2.2.6") == 2
    assert module._major_version("not-a-version") is None
