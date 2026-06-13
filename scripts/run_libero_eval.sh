#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

source "$script_dir/libero_profile.sh"
load_libero_profile "$repo_root"

python_bin="${LIBERO_PYTHON:-python}"
run_dir="${HIMEM_LIBERO_RUN_DIR:-}"

if [ -n "$run_dir" ]; then
  case "$run_dir" in
    /*) ;;
    *) run_dir="$repo_root/$run_dir" ;;
  esac
  export HIMEM_LIBERO_RUN_DIR="$run_dir"
fi

export HIMEM_SERVER_URI="${HIMEM_SERVER_URI:-ws://127.0.0.1:9000}"
export HIMEM_MUJOCO_GL="${HIMEM_MUJOCO_GL:-osmesa}"
export HIMEM_LIBERO_EPISODES="${HIMEM_LIBERO_EPISODES:-10}"
export HIMEM_LIBERO_TASK_SUITES="${HIMEM_LIBERO_TASK_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
export HIMEM_LIBERO_TASK_LIMIT="${HIMEM_LIBERO_TASK_LIMIT:-0}"
export HIMEM_LIBERO_MAX_STEPS="${HIMEM_LIBERO_MAX_STEPS:-25,25,25,95}"
export HIMEM_LIBERO_HORIZON="${HIMEM_LIBERO_HORIZON:-14}"
export HIMEM_LIBERO_CKPT_NAME="${HIMEM_LIBERO_CKPT_NAME:-HiMem_libero_eval}"

if [ -n "$run_dir" ]; then
  export HIMEM_LIBERO_LOG_DIR="${HIMEM_LIBERO_LOG_DIR:-$run_dir/logs}"
  export HIMEM_LIBERO_VIDEO_DIR="${HIMEM_LIBERO_VIDEO_DIR:-$run_dir/videos}"
  export HIMEM_LIBERO_RESULT_FILE="${HIMEM_LIBERO_RESULT_FILE:-$run_dir/results/${HIMEM_LIBERO_CKPT_NAME}_results.json}"
  export HIMEM_LIBERO_MANIFEST_FILE="${HIMEM_LIBERO_MANIFEST_FILE:-$run_dir/run_manifest.json}"
else
  export HIMEM_LIBERO_LOG_DIR="${HIMEM_LIBERO_LOG_DIR:-$repo_root/evaluations/libero/log_file}"
  export HIMEM_LIBERO_VIDEO_DIR="${HIMEM_LIBERO_VIDEO_DIR:-$repo_root/evaluations/libero/video_log_file/$HIMEM_LIBERO_CKPT_NAME}"
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

mkdir -p \
  "$HIMEM_LIBERO_LOG_DIR" \
  "$HIMEM_LIBERO_VIDEO_DIR" \
  "$(dirname "$HIMEM_LIBERO_RESULT_FILE")" \
  "$(dirname "$HIMEM_LIBERO_MANIFEST_FILE")"

"$repo_root/scripts/write_libero_run_manifest.py" \
  --output "$HIMEM_LIBERO_MANIFEST_FILE" \
  --run-kind eval \
  --repo-root "$repo_root"

cd "$repo_root/evaluations/libero"
exec "$python_bin" libero_client_4tasks.py
