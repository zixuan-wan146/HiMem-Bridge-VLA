#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

source "$script_dir/calvin_profile.sh"
load_calvin_profile "$repo_root"

python_bin="${CALVIN_PYTHON:-python}"
run_dir="${HIMEM_CALVIN_RUN_DIR:-}"

if [ -n "$run_dir" ]; then
  case "$run_dir" in
    /*) ;;
    *) run_dir="$repo_root/$run_dir" ;;
  esac
  export HIMEM_CALVIN_RUN_DIR="$run_dir"
fi

export HIMEM_SERVER_URI="${HIMEM_SERVER_URI:-ws://127.0.0.1:9000}"
export HIMEM_MUJOCO_GL="${HIMEM_MUJOCO_GL:-osmesa}"
export HIMEM_CALVIN_ROOT="${HIMEM_CALVIN_ROOT:-/root/autodl-tmp/calvin}"
export HIMEM_CALVIN_DATASET_PATH="${HIMEM_CALVIN_DATASET_PATH:-$HIMEM_CALVIN_ROOT/dataset/task_ABC_D}"
export CALVIN_ROOT="$HIMEM_CALVIN_ROOT"
export HIMEM_CALVIN_NUM_SEQUENCES="${HIMEM_CALVIN_NUM_SEQUENCES:-1000}"
export HIMEM_CALVIN_SEQUENCE_OFFSET="${HIMEM_CALVIN_SEQUENCE_OFFSET:-0}"
export HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK="${HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK:-360}"
export HIMEM_CALVIN_HORIZON="${HIMEM_CALVIN_HORIZON:-14}"
export HIMEM_CALVIN_CKPT_NAME="${HIMEM_CALVIN_CKPT_NAME:-HiMem_calvin_eval}"
export HIMEM_CALVIN_SAVE_VIDEO="${HIMEM_CALVIN_SAVE_VIDEO:-0}"
export HIMEM_CALVIN_VIDEO_FPS="${HIMEM_CALVIN_VIDEO_FPS:-30}"
export HIMEM_CALVIN_GRIPPER_MODE="${HIMEM_CALVIN_GRIPPER_MODE:-openvla}"
export HIMEM_CALVIN_RESET_MEMORY_SCOPE="${HIMEM_CALVIN_RESET_MEMORY_SCOPE:-sequence}"
export HIMEM_CALVIN_SHOW_GUI="${HIMEM_CALVIN_SHOW_GUI:-0}"

if [ -n "$run_dir" ]; then
  export HIMEM_CALVIN_LOG_DIR="${HIMEM_CALVIN_LOG_DIR:-$run_dir/logs}"
  export HIMEM_CALVIN_VIDEO_DIR="${HIMEM_CALVIN_VIDEO_DIR:-$run_dir/videos}"
  export HIMEM_CALVIN_RESULT_FILE="${HIMEM_CALVIN_RESULT_FILE:-$run_dir/results/${HIMEM_CALVIN_CKPT_NAME}_results.json}"
  export HIMEM_CALVIN_MANIFEST_FILE="${HIMEM_CALVIN_MANIFEST_FILE:-$run_dir/run_manifest.json}"
else
  export HIMEM_CALVIN_LOG_DIR="${HIMEM_CALVIN_LOG_DIR:-$repo_root/evaluations/calvin/log_file}"
  export HIMEM_CALVIN_VIDEO_DIR="${HIMEM_CALVIN_VIDEO_DIR:-$repo_root/evaluations/calvin/video_log_file/$HIMEM_CALVIN_CKPT_NAME}"
  export HIMEM_CALVIN_RESULT_FILE="${HIMEM_CALVIN_RESULT_FILE:-$HIMEM_CALVIN_LOG_DIR/${HIMEM_CALVIN_CKPT_NAME}_results.json}"
  export HIMEM_CALVIN_MANIFEST_FILE="${HIMEM_CALVIN_MANIFEST_FILE:-$HIMEM_CALVIN_LOG_DIR/${HIMEM_CALVIN_CKPT_NAME}_run_manifest.json}"
fi
export HIMEM_CALVIN_LOG_FILE="${HIMEM_CALVIN_LOG_FILE:-$HIMEM_CALVIN_LOG_DIR/$HIMEM_CALVIN_CKPT_NAME.txt}"

if [ "$HIMEM_MUJOCO_GL" = "egl" ]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

if [ "${HIMEM_CALVIN_DRY_RUN:-0}" = "1" ]; then
  env | sort | grep -E '^(CALVIN_ROOT|HIMEM_)'
  exit 0
fi

mkdir -p \
  "$HIMEM_CALVIN_LOG_DIR" \
  "$HIMEM_CALVIN_VIDEO_DIR" \
  "$(dirname "$HIMEM_CALVIN_RESULT_FILE")" \
  "$(dirname "$HIMEM_CALVIN_MANIFEST_FILE")"

"$repo_root/scripts/write_calvin_run_manifest.py" \
  --output "$HIMEM_CALVIN_MANIFEST_FILE" \
  --run-kind eval \
  --repo-root "$repo_root"

cd "$repo_root"
exec "$python_bin" -m evaluations.calvin.calvin_client
