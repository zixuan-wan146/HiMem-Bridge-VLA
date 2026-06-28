#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

source "$script_dir/libero_profile.sh"
load_libero_profile "$repo_root"

fail() {
  printf '[libero-eval] ERROR: %s\n' "$*" >&2
  exit 2
}

require_env() {
  if [ -z "${!1+x}" ] || [ -z "${!1}" ]; then
    fail "$1 must be set in HIMEM_LIBERO_PROFILE or the environment"
  fi
}

python_bin="${LIBERO_PYTHON:-python}"
run_dir="${HIMEM_LIBERO_RUN_DIR:-}"

if [ -n "$run_dir" ]; then
  case "$run_dir" in
    /*)
      fail "HIMEM_LIBERO_RUN_DIR must be project-relative: $run_dir"
      ;;
  esac
  export HIMEM_LIBERO_RUN_DIR="$run_dir"
fi

require_env HIMEM_MUJOCO_GL
require_env HIMEM_LIBERO_EPISODES
require_env HIMEM_LIBERO_TASK_SUITES
require_env HIMEM_LIBERO_TASK_LIMIT
require_env HIMEM_LIBERO_EPISODE_OFFSET
require_env HIMEM_LIBERO_MAX_STEPS
require_env HIMEM_LIBERO_HORIZON
require_env HIMEM_LIBERO_CKPT_NAME

export HIMEM_SERVER_URI="${HIMEM_SERVER_URI:-ws://127.0.0.1:9000}"
export HIMEM_MUJOCO_GL
export HIMEM_LIBERO_EPISODES
export HIMEM_LIBERO_TASK_SUITES
export HIMEM_LIBERO_TASK_LIMIT
export HIMEM_LIBERO_EPISODE_OFFSET
export HIMEM_LIBERO_MAX_STEPS
export HIMEM_LIBERO_HORIZON
export HIMEM_LIBERO_CKPT_NAME

if [ -n "$run_dir" ]; then
  export HIMEM_LIBERO_LOG_DIR="${HIMEM_LIBERO_LOG_DIR:-$run_dir/logs}"
  export HIMEM_LIBERO_VIDEO_DIR="${HIMEM_LIBERO_VIDEO_DIR:-$run_dir/videos}"
  export HIMEM_LIBERO_RESULT_FILE="${HIMEM_LIBERO_RESULT_FILE:-$run_dir/results/${HIMEM_LIBERO_CKPT_NAME}_results.json}"
  export HIMEM_LIBERO_MANIFEST_FILE="${HIMEM_LIBERO_MANIFEST_FILE:-$run_dir/run_manifest.json}"
else
  export HIMEM_LIBERO_LOG_DIR="${HIMEM_LIBERO_LOG_DIR:-run_outputs/libero/log_file}"
  export HIMEM_LIBERO_VIDEO_DIR="${HIMEM_LIBERO_VIDEO_DIR:-run_outputs/libero/video_log_file/$HIMEM_LIBERO_CKPT_NAME}"
  export HIMEM_LIBERO_RESULT_FILE="${HIMEM_LIBERO_RESULT_FILE:-$HIMEM_LIBERO_LOG_DIR/${HIMEM_LIBERO_CKPT_NAME}_results.json}"
  export HIMEM_LIBERO_MANIFEST_FILE="${HIMEM_LIBERO_MANIFEST_FILE:-$HIMEM_LIBERO_LOG_DIR/${HIMEM_LIBERO_CKPT_NAME}_run_manifest.json}"
fi
export HIMEM_LIBERO_LOG_FILE="${HIMEM_LIBERO_LOG_FILE:-$HIMEM_LIBERO_LOG_DIR/$HIMEM_LIBERO_CKPT_NAME.txt}"

if [ "$HIMEM_MUJOCO_GL" = "egl" ]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

if [ "${HIMEM_LIBERO_DRY_RUN:-0}" = "1" ]; then
  env | sort | grep '^HIMEM_'
  exit 0
fi

cd "$repo_root"

mkdir -p \
  "$HIMEM_LIBERO_LOG_DIR" \
  "$HIMEM_LIBERO_VIDEO_DIR" \
  "$(dirname "$HIMEM_LIBERO_RESULT_FILE")" \
  "$(dirname "$HIMEM_LIBERO_MANIFEST_FILE")"

scripts/report/write_libero_run_manifest.py \
  --output "$HIMEM_LIBERO_MANIFEST_FILE" \
  --run-kind eval \
  --repo-root "."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" scripts/eval/eval_libero.py
