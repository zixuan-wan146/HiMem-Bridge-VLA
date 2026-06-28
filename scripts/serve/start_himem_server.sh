#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

python_bin="${HIMEM_PYTHON:-python}"
preflight_python="${HIMEM_PREFLIGHT_PYTHON:-$python_bin}"
ckpt_dir="${HIMEM_CKPT_DIR:-${1:-}}"
host="${HIMEM_HOST:-127.0.0.1}"
port="${HIMEM_PORT:-9000}"
device="${HIMEM_DEVICE:-cuda:0}"
inference_steps="${HIMEM_INFERENCE_STEPS:-15}"
skip_preflight="${HIMEM_SKIP_PREFLIGHT:-0}"
allow_unsafe_checkpoint_load="${HIMEM_ALLOW_UNSAFE_CHECKPOINT_LOAD:-0}"
vlm_name="${HIMEM_VLM_NAME:-}"
vlm_local_files_only="${HIMEM_VLM_LOCAL_FILES_ONLY:-1}"

if [ -z "$ckpt_dir" ]; then
  printf 'Usage: HIMEM_PYTHON=.venv/bin/python %s checkpoints/HiMem_LIBERO\n' "$0" >&2
  printf 'Or set HIMEM_CKPT_DIR=checkpoints/HiMem_LIBERO.\n' >&2
  exit 2
fi

case "$ckpt_dir" in
  /*)
    printf 'Checkpoint directory must be project-relative: %s\n' "$ckpt_dir" >&2
    exit 2
    ;;
esac

cd "$repo_root"

if [ ! -d "$ckpt_dir" ]; then
  printf 'Checkpoint directory does not exist: %s\n' "$ckpt_dir" >&2
  exit 2
fi

if [ "$skip_preflight" != "1" ]; then
  "$preflight_python" scripts/quality/preflight.py \
    --repo-root "." \
    --dataset-config "" \
    --checkpoint "$ckpt_dir" \
    --skip-shell-syntax
fi

server_args=(
  scripts/serve/serve_policy.py
  --ckpt_dir "$ckpt_dir"
  --host "$host"
  --port "$port"
  --device "$device"
  --inference_steps "$inference_steps"
)

if [ "$allow_unsafe_checkpoint_load" = "1" ]; then
  server_args+=(--allow_unsafe_checkpoint_load)
fi

if [ -n "$vlm_name" ]; then
  server_args+=(--vlm_name "$vlm_name")
fi

if [ "$vlm_local_files_only" = "1" ]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  server_args+=(--vlm_local_files_only)
fi

exec "$python_bin" "${server_args[@]}"
