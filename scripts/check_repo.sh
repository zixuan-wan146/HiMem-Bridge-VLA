#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[check-repo] %s\n' "$*" >&2
}

run_step() {
  local name=$1
  shift

  if [ "${HIMEM_CHECK_DRY_RUN:-0}" = "1" ]; then
    printf '[check-repo] DRY-RUN %s:' "$name"
    printf ' %q' "$@"
    printf '\n'
    return
  fi

  log "$name"
  "$@"
}

check_shell_syntax() {
  local script
  if [ "${HIMEM_CHECK_DRY_RUN:-0}" = "1" ]; then
    printf '[check-repo] DRY-RUN shell syntax: bash -n scripts/*.sh\n'
    return
  fi

  log "Shell script syntax"
  while IFS= read -r -d '' script; do
    bash -n "$script"
  done < <(find scripts -name "*.sh" -print0)
}

run_ruff() {
  if [ "${HIMEM_CHECK_SKIP_RUFF:-0}" = "1" ]; then
    log "Skipping ruff because HIMEM_CHECK_SKIP_RUFF=1"
    return
  fi

  if [ "${HIMEM_CHECK_DRY_RUN:-0}" = "1" ]; then
    run_step "Ruff lint" "$python_bin" -m ruff check .
    return
  fi

  if "$python_bin" -m ruff --version >/dev/null 2>&1; then
    run_step "Ruff lint" "$python_bin" -m ruff check .
    return
  fi

  if [ "${HIMEM_CHECK_REQUIRE_RUFF:-0}" = "1" ]; then
    log "ERROR: ruff is required but not importable by $python_bin"
    exit 1
  fi

  log "WARN: ruff is not installed for $python_bin; skipped lint"
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON:-python3}"

cd "$repo_root"

run_step "Requirements policy audit" "$python_bin" scripts/audit_requirements.py

if [ "${HIMEM_CHECK_SKIP_PYTEST:-0}" = "1" ]; then
  log "Skipping pytest because HIMEM_CHECK_SKIP_PYTEST=1"
else
  run_step "Unit tests" "$python_bin" -m pytest
fi

run_ruff
check_shell_syntax
run_step "Repository preflight" "$python_bin" scripts/preflight.py
run_step "Bridge-HiMem config validation" "$python_bin" scripts/validate_bridge_himem_configs.py
run_step "LIBERO setup dry-run" env HIMEM_SETUP_LIBERO_DRY_RUN=1 "$script_dir/setup_libero_env.sh"
run_step \
  "LIBERO checkpoint download dry-run" \
  env HIMEM_DOWNLOAD_LIBERO_CHECKPOINT_DRY_RUN=1 "$script_dir/download_libero_checkpoint.sh"
run_step \
  "LIBERO smoke profile dry-run" \
  env HIMEM_LIBERO_DRY_RUN=1 HIMEM_LIBERO_PROFILE=configs/libero_profiles/smoke.env \
  "$script_dir/run_libero_smoke.sh"
run_step \
  "LIBERO eval profile dry-run" \
  env HIMEM_LIBERO_DRY_RUN=1 HIMEM_LIBERO_PROFILE=configs/libero_profiles/full_eval.env \
  "$script_dir/run_libero_eval.sh"
run_step \
  "CALVIN eval profile dry-run" \
  env HIMEM_CALVIN_DRY_RUN=1 HIMEM_CALVIN_PROFILE=configs/calvin_profiles/full_eval.env \
  "$script_dir/run_calvin_eval.sh"
run_step \
  "LIBERO experiment init dry-run" \
  "$python_bin" "$script_dir/init_libero_experiment.py" \
  --dry-run \
  --name check_repo_smoke \
  --root /tmp/himem_check_experiments \
  --checkpoint /tmp/HiMem_LIBERO \
  --profile configs/libero_profiles/smoke.env \
  --kind smoke

if [ "${HIMEM_CHECK_SKIP_COMPILE:-0}" = "1" ]; then
  log "Skipping compileall because HIMEM_CHECK_SKIP_COMPILE=1"
else
  run_step \
    "Python compileall" \
    env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/himem_pycache}" \
    "$python_bin" -m compileall -q himem_bridge_vla evaluations scripts tests
fi

if [ "${HIMEM_CHECK_DRY_RUN:-0}" = "1" ] || git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  run_step "Git whitespace check" git diff --check
else
  log "Skipping Git whitespace check because this copy is not a Git repository"
fi
log "All requested checks passed"
