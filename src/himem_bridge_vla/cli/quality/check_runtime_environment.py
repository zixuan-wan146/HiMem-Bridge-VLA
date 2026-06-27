#!/usr/bin/env python3
from __future__ import annotations

from importlib import metadata
import sys


def main() -> int:
    failures = []
    try:
        numpy_version = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        failures.append("numpy is not installed")
    else:
        major = _major_version(numpy_version)
        if major != 1:
            failures.append(f"numpy must be 1.x for this project, got {numpy_version}")
        else:
            print(f"[OK] runtime: numpy=={numpy_version} satisfies numpy<2")

    if failures:
        for failure in failures:
            print(f"[FAIL] runtime: {failure}", file=sys.stderr)
        return 1
    return 0


def _major_version(version: str) -> int | None:
    token = version.split(".", 1)[0]
    try:
        return int(token)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
