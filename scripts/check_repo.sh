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

find_python() {
  if [ -n "${PYTHON:-}" ]; then
    if command -v "$PYTHON" >/dev/null 2>&1 || [ -x "$PYTHON" ]; then
      printf '%s\n' "$PYTHON"
      return
    fi
    log "ERROR: PYTHON is not executable: $PYTHON"
    exit 1
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return
  fi

  log "ERROR: no Python interpreter found; set PYTHON to the interpreter for this project"
  exit 1
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

run_pytest() {
  run_step "Unit tests" "$python_bin" -m pytest
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="$(find_python)"

cd "$repo_root"

run_step "Requirements policy audit" "$python_bin" scripts/audit_requirements.py
run_step "Runtime environment check" "$python_bin" scripts/check_runtime_environment.py

if [ "${HIMEM_CHECK_SKIP_PYTEST:-0}" = "1" ]; then
  log "Skipping pytest because HIMEM_CHECK_SKIP_PYTEST=1"
else
  run_pytest
fi

run_ruff
check_shell_syntax
run_step "Repository preflight" "$python_bin" scripts/preflight.py
run_step "Bridge-HiMem config validation" "$python_bin" scripts/validate_bridge_himem_configs.py
run_step "Training config validation" "$python_bin" scripts/validate_training_configs.py
run_step \
  "Direct bridge inference smoke" \
  "$python_bin" scripts/smoke_direct_bridge_inference.py --preset tiny --device cpu
run_step \
  "Direct bridge token-cache training smoke" \
  "$python_bin" scripts/smoke_direct_bridge_token_cache_training.py --preset tiny --device cpu --steps 1 --batch-size 2
run_step "Benchmark inventory" "$python_bin" scripts/inspect_benchmarks.py --allow-missing
run_step "LIBERO setup dry-run" env HIMEM_SETUP_LIBERO_DRY_RUN=1 scripts/setup_libero_env.sh
run_step \
  "LIBERO checkpoint download dry-run" \
  env HIMEM_DOWNLOAD_LIBERO_CHECKPOINT_DRY_RUN=1 scripts/download_libero_checkpoint.sh
run_step \
  "LIBERO smoke profile dry-run" \
  env HIMEM_LIBERO_DRY_RUN=1 HIMEM_LIBERO_PROFILE=configs/libero_profiles/smoke.env \
  scripts/run_libero_smoke.sh
run_step \
  "LIBERO eval profile dry-run" \
  env HIMEM_LIBERO_DRY_RUN=1 HIMEM_LIBERO_PROFILE=configs/libero_profiles/full_eval.env \
  scripts/run_libero_eval.sh
run_step \
  "LIBERO experiment init dry-run" \
  "$python_bin" scripts/init_libero_experiment.py \
  --dry-run \
  --name check_repo_smoke \
  --root run_outputs/check_repo_experiments \
  --checkpoint run_outputs/check_repo_checkpoints/HiMem_LIBERO \
  --profile configs/libero_profiles/smoke.env \
  --kind smoke
run_step \
  "RMBench eval dry-run" \
  env HIMEM_RMBENCH_DRY_RUN=1 HIMEM_RMBENCH_TASKS=press_button \
  scripts/run_rmbench_eval.sh
rmbench_root="${HIMEM_RMBENCH_ROOT:-${AUTODL_TMP:-/root/autodl-tmp}/benchmarks/RMBench}"
if [ "${HIMEM_CHECK_DRY_RUN:-0}" = "1" ] || [ -d "$rmbench_root" ]; then
  run_step \
    "RMBench eval plan-only" \
    env HIMEM_RMBENCH_PLAN_ONLY=1 HIMEM_RMBENCH_TASKS=press_button \
    HIMEM_RMBENCH_RUN_DIR=run_outputs/check_repo_rmbench_plan_only \
    HIMEM_RMBENCH_PYTHON="$python_bin" \
    scripts/run_rmbench_eval.sh
elif [ "${HIMEM_CHECK_REQUIRE_RMBENCH:-0}" = "1" ]; then
  log "ERROR: RMBench root not found: $rmbench_root"
  exit 1
else
  log "Skipping RMBench eval plan-only because RMBench root was not found: $rmbench_root"
fi

if [ "${HIMEM_CHECK_SKIP_COMPILE:-0}" = "1" ]; then
  log "Skipping compileall because HIMEM_CHECK_SKIP_COMPILE=1"
else
  run_step \
    "Python compileall" \
    env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-run_outputs/pycache}" \
    "$python_bin" -m compileall -q himem_bridge_vla evaluations scripts tests
fi

if [ "${HIMEM_CHECK_DRY_RUN:-0}" = "1" ] || git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  run_step "Git whitespace check" git diff --check
else
  log "Skipping Git whitespace check because this copy is not a Git repository"
fi
log "All requested checks passed"
