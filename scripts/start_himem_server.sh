#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

python_bin="${HIMEM_PYTHON:-python}"
preflight_python="${HIMEM_PREFLIGHT_PYTHON:-$python_bin}"
ckpt_dir="${HIMEM_CKPT_DIR:-${1:-}}"
host="${HIMEM_HOST:-127.0.0.1}"
port="${HIMEM_PORT:-9000}"
device="${HIMEM_DEVICE:-cuda:0}"
inference_steps="${HIMEM_INFERENCE_STEPS:-1}"
skip_preflight="${HIMEM_SKIP_PREFLIGHT:-0}"

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
  "$preflight_python" scripts/preflight.py \
    --repo-root "." \
    --dataset-config "" \
    --checkpoint "$ckpt_dir" \
    --skip-shell-syntax
fi

exec "$python_bin" scripts/himem_server.py \
  --ckpt_dir "$ckpt_dir" \
  --host "$host" \
  --port "$port" \
  --device "$device" \
  --inference_steps "$inference_steps"
