#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[export-unpushed] %s\n' "$*" >&2
}

fail() {
  printf '[export-unpushed] ERROR: %s\n' "$*" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${HIMEM_EXPORT_REPO:-$script_dir/..}" && pwd)"
base_ref="${HIMEM_EXPORT_BASE:-origin/main}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${HIMEM_EXPORT_DIR:-$repo_root/exports/unpushed_commits_$timestamp}"
python_bin="${PYTHON:-python3}"

case "$out_dir" in
  /*) ;;
  *) out_dir="$repo_root/$out_dir" ;;
esac

git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
  fail "not a Git repository: $repo_root"

git -C "$repo_root" rev-parse --verify --quiet "$base_ref^{commit}" >/dev/null ||
  fail "base ref does not exist or is not a commit: $base_ref"

commit_count="$(git -C "$repo_root" rev-list --count "$base_ref"..HEAD)"
if [ "$commit_count" = "0" ]; then
  fail "no commits to export: HEAD is not ahead of $base_ref"
fi

if [ -e "$out_dir" ] && [ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  fail "output directory already exists and is not empty: $out_dir"
fi

mkdir -p "$out_dir/patches"

base_commit="$(git -C "$repo_root" rev-parse "$base_ref")"
head_commit="$(git -C "$repo_root" rev-parse HEAD)"
status_short="$(git -C "$repo_root" status --short)"

if [ -n "$status_short" ]; then
  log "working tree has uncommitted changes; only committed changes are exported"
fi

git -C "$repo_root" format-patch "$base_ref"..HEAD -o "$out_dir/patches" >/dev/null

"$python_bin" - "$repo_root" "$out_dir" "$base_ref" "$base_commit" "$head_commit" "$commit_count" <<'PY'
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import subprocess
import sys


repo_root = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
base_ref = sys.argv[3]
base_commit = sys.argv[4]
head_commit = sys.argv[5]
commit_count = int(sys.argv[6])


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


commits = []
log_output = git("log", "--reverse", "--pretty=format:%H%x00%an%x00%ae%x00%aI%x00%s", f"{base_ref}..HEAD")
for line in log_output.splitlines():
    commit_hash, author_name, author_email, author_date, subject = line.split("\0", 4)
    commits.append(
        {
            "hash": commit_hash,
            "author_name": author_name,
            "author_email": author_email,
            "author_date": author_date,
            "subject": subject,
        }
    )

status_short = git("status", "--short").splitlines()
patches = sorted(str(path.relative_to(out_dir)) for path in (out_dir / "patches").glob("*.patch"))
manifest = {
    "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "repo_root": str(repo_root),
    "base_ref": base_ref,
    "base_commit": base_commit,
    "head_commit": head_commit,
    "branch": git("branch", "--show-current") or None,
    "commit_count": commit_count,
    "dirty": bool(status_short),
    "status_short": status_short,
    "patches": patches,
    "commits": commits,
}

(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

cat > "$out_dir/README.md" <<README
# HiMem-Bridge-VLA Unpushed Commit Export

Created from:

- Repository: \`$repo_root\`
- Base ref: \`$base_ref\`
- Base commit: \`$base_commit\`
- Head commit: \`$head_commit\`
- Commit count: \`$commit_count\`

Apply this bundle to another clone whose history contains \`$base_commit\`:

\`\`\`bash
git am /path/to/export/patches/*.patch
python3 -m pytest
python3 scripts/preflight.py
bash -n scripts/*.sh
PYTHONPYCACHEPREFIX=/tmp/himem_pycache python3 -m compileall -q himem_bridge_vla evaluations scripts tests
git diff --check
\`\`\`

The patch files only contain committed changes. If \`manifest.json\` has \`dirty: true\`, inspect
\`status_short\` in the manifest before treating this export as complete.
README

log "wrote $commit_count commit(s) to $out_dir"
printf '%s\n' "$out_dir"
