#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[download-libero-checkpoint] %s\n' "$*" >&2
}

fail() {
  printf '[download-libero-checkpoint] ERROR: %s\n' "$*" >&2
  exit 1
}

default_data_root() {
  if [ -n "${HIMEM_DATA_ROOT:-}" ]; then
    printf '%s\n' "$HIMEM_DATA_ROOT"
  else
    printf '%s\n' "run_outputs/libero_data"
  fi
}

require_project_relative() {
  local name=$1
  local value=$2
  case "$value" in
    /*) fail "$name must be project-relative: $value" ;;
  esac
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"
data_root="$(default_data_root)"
require_project_relative "HIMEM_DATA_ROOT" "$data_root"
repo_id="${HIMEM_LIBERO_CHECKPOINT_REPO:-MINT-SJTU/HiMem_LIBERO}"
checkpoint_dir="${HIMEM_LIBERO_CHECKPOINT_DIR:-$data_root/checkpoints/HiMem_LIBERO}"
hf_home="${HF_HOME:-$data_root/hf-home}"
hf_cache="${HUGGINGFACE_HUB_CACHE:-$data_root/hf-cache}"
max_workers="${HF_MAX_WORKERS:-1}"

require_project_relative "HIMEM_LIBERO_CHECKPOINT_DIR" "$checkpoint_dir"
require_project_relative "HF_HOME" "$hf_home"
require_project_relative "HUGGINGFACE_HUB_CACHE" "$hf_cache"

if [ "${HIMEM_DOWNLOAD_LIBERO_CHECKPOINT_DRY_RUN:-0}" = "1" ]; then
  printf 'HIMEM_DATA_ROOT=%s\n' "$data_root"
  printf 'HIMEM_LIBERO_CHECKPOINT_REPO=%s\n' "$repo_id"
  printf 'HIMEM_LIBERO_CHECKPOINT_DIR=%s\n' "$checkpoint_dir"
  printf 'HF_HOME=%s\n' "$hf_home"
  printf 'HUGGINGFACE_HUB_CACHE=%s\n' "$hf_cache"
  printf 'HF_MAX_WORKERS=%s\n' "$max_workers"
  printf 'HIMEM_HF_ENDPOINT=%s\n' "${HIMEM_HF_ENDPOINT:-}"
  printf 'COMMAND=env HF_HOME=%q HUGGINGFACE_HUB_CACHE=%q HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1' "$hf_home" "$hf_cache"
  if [ -n "${HIMEM_HF_ENDPOINT:-}" ]; then
    printf ' HF_ENDPOINT=%q' "$HIMEM_HF_ENDPOINT"
  fi
  printf ' hf download %q --local-dir %q --max-workers %q\n' "$repo_id" "$checkpoint_dir" "$max_workers"
  exit 0
fi

command -v hf >/dev/null 2>&1 || fail "hf CLI not found. Install huggingface_hub or run from an environment that provides hf."

mkdir -p "$checkpoint_dir" "$hf_home" "$hf_cache"

env_args=(
  "HF_HOME=$hf_home"
  "HUGGINGFACE_HUB_CACHE=$hf_cache"
  "HF_HUB_DISABLE_TELEMETRY=1"
  "HF_HUB_DISABLE_XET=1"
)
if [ -n "${HIMEM_HF_ENDPOINT:-}" ]; then
  env_args+=("HF_ENDPOINT=$HIMEM_HF_ENDPOINT")
fi

log "Downloading $repo_id to $checkpoint_dir"
env "${env_args[@]}" hf download "$repo_id" --local-dir "$checkpoint_dir" --max-workers "$max_workers"
log "Checkpoint is ready at $checkpoint_dir"
