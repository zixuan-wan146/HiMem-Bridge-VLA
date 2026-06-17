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
    /*)
      printf '[libero-smoke] ERROR: HIMEM_LIBERO_RUN_DIR must be project-relative: %s\n' "$run_dir" >&2
      exit 2
      ;;
  esac
  export HIMEM_LIBERO_RUN_DIR="$run_dir"
fi

export HIMEM_SERVER_URI="${HIMEM_SERVER_URI:-ws://127.0.0.1:9000}"
export HIMEM_MUJOCO_GL="${HIMEM_MUJOCO_GL:-osmesa}"
export HIMEM_LIBERO_EPISODES="${HIMEM_LIBERO_EPISODES:-1}"
export HIMEM_LIBERO_TASK_SUITES="${HIMEM_LIBERO_TASK_SUITES:-libero_spatial}"
export HIMEM_LIBERO_TASK_LIMIT="${HIMEM_LIBERO_TASK_LIMIT:-1}"
export HIMEM_LIBERO_MAX_STEPS="${HIMEM_LIBERO_MAX_STEPS:-1}"
export HIMEM_LIBERO_HORIZON="${HIMEM_LIBERO_HORIZON:-1}"
export HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT="${HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT:-0}"
export HIMEM_LIBERO_TRANSITION_DATASET_NAME="${HIMEM_LIBERO_TRANSITION_DATASET_NAME:-}"
export HIMEM_LIBERO_CKPT_NAME="${HIMEM_LIBERO_CKPT_NAME:-HiMem_libero_smoke}"

if [ -n "$run_dir" ]; then
  export HIMEM_LIBERO_LOG_DIR="${HIMEM_LIBERO_LOG_DIR:-$run_dir/logs}"
  export HIMEM_LIBERO_VIDEO_DIR="${HIMEM_LIBERO_VIDEO_DIR:-$run_dir/videos}"
  export HIMEM_LIBERO_RESULT_FILE="${HIMEM_LIBERO_RESULT_FILE:-$run_dir/results/${HIMEM_LIBERO_CKPT_NAME}_results.json}"
  export HIMEM_LIBERO_MANIFEST_FILE="${HIMEM_LIBERO_MANIFEST_FILE:-$run_dir/run_manifest.json}"
else
  export HIMEM_LIBERO_LOG_DIR="${HIMEM_LIBERO_LOG_DIR:-evaluations/libero/log_file}"
  export HIMEM_LIBERO_VIDEO_DIR="${HIMEM_LIBERO_VIDEO_DIR:-evaluations/libero/video_log_file/$HIMEM_LIBERO_CKPT_NAME}"
  export HIMEM_LIBERO_RESULT_FILE="${HIMEM_LIBERO_RESULT_FILE:-$HIMEM_LIBERO_LOG_DIR/${HIMEM_LIBERO_CKPT_NAME}_results.json}"
  export HIMEM_LIBERO_MANIFEST_FILE="${HIMEM_LIBERO_MANIFEST_FILE:-$HIMEM_LIBERO_LOG_DIR/${HIMEM_LIBERO_CKPT_NAME}_run_manifest.json}"
fi
export HIMEM_LIBERO_LOG_FILE="${HIMEM_LIBERO_LOG_FILE:-$HIMEM_LIBERO_LOG_DIR/$HIMEM_LIBERO_CKPT_NAME.txt}"
if [ -n "$HIMEM_LIBERO_TRANSITION_DATASET_NAME" ]; then
  export HIMEM_LIBERO_TRANSITION_TRACE_FILE="${HIMEM_LIBERO_TRANSITION_TRACE_FILE:-$(dirname "$HIMEM_LIBERO_RESULT_FILE")/${HIMEM_LIBERO_CKPT_NAME}_transition_trace.jsonl}"
fi

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
if [ -n "${HIMEM_LIBERO_TRANSITION_TRACE_FILE:-}" ]; then
  mkdir -p "$(dirname "$HIMEM_LIBERO_TRANSITION_TRACE_FILE")"
fi

scripts/write_libero_run_manifest.py \
  --output "$HIMEM_LIBERO_MANIFEST_FILE" \
  --run-kind smoke \
  --repo-root "."

export PYTHONPATH=".:evaluations/libero${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" evaluations/libero/libero_client_4tasks.py
