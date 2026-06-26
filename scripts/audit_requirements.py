#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys


IGNORED_DISCOVERY_DIRS = {
    ".himem-data",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "exports",
    "reference-paper",
    "referen-repo",
    "reference-repo",
}


@dataclass(frozen=True)
class RequirementEntry:
    file_path: str
    line_number: int
    name: str
    raw: str
    exact_pinned: bool


@dataclass(frozen=True)
class AuditMessage:
    level: str
    message: str


class AuditReport:
    def __init__(self) -> None:
        self.messages: list[AuditMessage] = []

    def ok(self, message: str) -> None:
        self.messages.append(AuditMessage("OK", message))

    def warn(self, message: str) -> None:
        self.messages.append(AuditMessage("WARN", message))

    def fail(self, message: str) -> None:
        self.messages.append(AuditMessage("FAIL", message))

    @property
    def has_failures(self) -> bool:
        return any(message.level == "FAIL" for message in self.messages)

    def print(self) -> None:
        for message in self.messages:
            print(f"[{message.level}] requirements: {message.message}")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def strip_inline_comment(line: str) -> str:
    if " #" in line:
        return line.split(" #", 1)[0].strip()
    return line.strip()


def parse_requirement_line(file_path: str, line_number: int, raw_line: str) -> RequirementEntry | None:
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("-") or stripped.startswith("--"):
        return None

    requirement = strip_inline_comment(stripped)
    if not requirement:
        return None

    egg_match = re.search(r"[#&]egg=([A-Za-z0-9_.-]+)", requirement)
    if egg_match:
        name = egg_match.group(1)
    else:
        name_match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
        if name_match is None:
            raise ValueError(f"{file_path}:{line_number}: could not parse requirement line: {raw_line.rstrip()}")
        name = name_match.group(1)

    exact_pinned = bool(re.search(r"(?<![!<>=~])==(?!=)|===", requirement))
    return RequirementEntry(
        file_path=file_path,
        line_number=line_number,
        name=normalize_name(name),
        raw=requirement,
        exact_pinned=exact_pinned,
    )


def load_policy(policy_path: Path) -> dict[str, dict[str, str]]:
    try:
        payload = json.loads(policy_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON policy {policy_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        raise ValueError(f"policy must contain a files object: {policy_path}")

    normalized_policy: dict[str, dict[str, str]] = {}
    for raw_file_path, file_policy in payload["files"].items():
        if not isinstance(raw_file_path, str) or not raw_file_path:
            raise ValueError("policy file paths must be non-empty strings")
        if not isinstance(file_policy, dict):
            raise ValueError(f"policy for {raw_file_path} must be an object")
        allow_unpinned = file_policy.get("allow_unpinned", {})
        if not isinstance(allow_unpinned, dict):
            raise ValueError(f"policy for {raw_file_path} must contain allow_unpinned object")

        normalized_allowlist = {}
        for raw_name, reason in allow_unpinned.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(f"allow_unpinned names for {raw_file_path} must be non-empty strings")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"allow_unpinned reason for {raw_file_path}:{raw_name} must be non-empty")
            normalized_allowlist[normalize_name(raw_name)] = reason.strip()
        normalized_policy[raw_file_path] = normalized_allowlist
    return normalized_policy


def discover_requirement_files(repo_root: Path) -> set[str]:
    discovered = set()
    for path in repo_root.rglob("requirements*.txt"):
        rel_parts = path.relative_to(repo_root).parts
        if any(part in IGNORED_DISCOVERY_DIRS for part in rel_parts):
            continue
        discovered.add(path.relative_to(repo_root).as_posix())
    return discovered


def read_requirement_entries(repo_root: Path, file_path: str) -> list[RequirementEntry]:
    path = repo_root / file_path
    entries = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        entry = parse_requirement_line(file_path, line_number, raw_line)
        if entry is not None:
            entries.append(entry)
    return entries


def audit_requirements(repo_root: Path, policy_path: Path) -> AuditReport:
    report = AuditReport()
    policy = load_policy(policy_path)
    discovered = discover_requirement_files(repo_root)
    policy_files = set(policy)

    for file_path in sorted(discovered - policy_files):
        report.fail(f"{file_path} is not covered by {policy_path.name}")
    for file_path in sorted(policy_files - discovered):
        report.fail(f"{file_path} is listed in {policy_path.name} but was not found")

    for file_path in sorted(policy_files & discovered):
        allowlist = policy[file_path]
        entries = read_requirement_entries(repo_root, file_path)
        observed_allowed_unpinned: set[str] = set()
        unpinned_count = 0

        for entry in entries:
            if entry.exact_pinned:
                continue
            unpinned_count += 1
            if entry.name in allowlist:
                observed_allowed_unpinned.add(entry.name)
                report.warn(
                    f"{entry.file_path}:{entry.line_number} {entry.raw!r} is intentionally unpinned: "
                    f"{allowlist[entry.name]}"
                )
            else:
                report.fail(f"{entry.file_path}:{entry.line_number} {entry.raw!r} is unpinned and not allowlisted")

        stale_allowlist = sorted(set(allowlist) - observed_allowed_unpinned)
        for name in stale_allowlist:
            report.fail(f"{file_path} allowlist contains stale unpinned dependency: {name}")

        if unpinned_count == 0:
            report.ok(f"{file_path} has exact pins for all top-level dependencies")
        else:
            report.ok(f"{file_path} checked with {unpinned_count} allowlisted unpinned dependency/dependencies")

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit requirement files for untracked dependency drift.")
    parser.add_argument("--repo-root", default=str(repo_root_from_script()), help="Repository root to check.")
    parser.add_argument(
        "--policy",
        default="requirements-policy.json",
        help="Policy JSON file, relative to repo root unless absolute.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    policy_path = Path(args.policy).expanduser()
    if not policy_path.is_absolute():
        policy_path = repo_root / policy_path

    try:
        report = audit_requirements(repo_root, policy_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report.print()
    return 1 if report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
