#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

autodl_tmp="${AUTODL_TMP:-/root/autodl-tmp}"
python_bin="${HIMEM_RMBENCH_PYTHON:-${HIMEM_PYTHON:-python}}"
rmbench_root="${HIMEM_RMBENCH_ROOT:-$autodl_tmp/benchmarks/RMBench}"
run_dir="${HIMEM_RMBENCH_RUN_DIR:-run_outputs/rmbench_eval}"

export HIMEM_RMBENCH_ROOT="$rmbench_root"
export HIMEM_RMBENCH_RUN_DIR="$run_dir"
export HIMEM_RMBENCH_LOG_DIR="${HIMEM_RMBENCH_LOG_DIR:-$run_dir/logs}"
export HIMEM_RMBENCH_PLAN_FILE="${HIMEM_RMBENCH_PLAN_FILE:-$run_dir/rmbench_eval_plan.md}"
export HIMEM_RMBENCH_MANIFEST_FILE="${HIMEM_RMBENCH_MANIFEST_FILE:-$run_dir/run_manifest.json}"
export HIMEM_RMBENCH_POLICY_NAME="${HIMEM_RMBENCH_POLICY_NAME:-HiMemBridgeVLA}"
export HIMEM_RMBENCH_TASKS="${HIMEM_RMBENCH_TASKS:-observe_and_pickup,rearrange_blocks,put_back_block,swap_blocks,swap_T,blocks_ranking_try,press_button,cover_blocks,battery_try}"
export HIMEM_RMBENCH_TASK_CONFIG="${HIMEM_RMBENCH_TASK_CONFIG:-demo_clean}"
export HIMEM_RMBENCH_CKPT_SETTING="${HIMEM_RMBENCH_CKPT_SETTING:-himem_bridge_vla}"
export HIMEM_RMBENCH_SEED="${HIMEM_RMBENCH_SEED:-0}"
export HIMEM_RMBENCH_GPU_ID="${HIMEM_RMBENCH_GPU_ID:-0}"
export HIMEM_RMBENCH_INSTRUCTION_TYPE="${HIMEM_RMBENCH_INSTRUCTION_TYPE:-unseen}"
export HIMEM_RMBENCH_ACTION_HORIZON="${HIMEM_RMBENCH_ACTION_HORIZON:-32}"
export HIMEM_RMBENCH_ACTION_DIM="${HIMEM_RMBENCH_ACTION_DIM:-14}"
export HIMEM_RMBENCH_ACTION_TYPE="${HIMEM_RMBENCH_ACTION_TYPE:-qpos}"
export HIMEM_RMBENCH_STATE_SOURCE="${HIMEM_RMBENCH_STATE_SOURCE:-endpose}"
export HIMEM_RMBENCH_ROBOT_KEY="${HIMEM_RMBENCH_ROBOT_KEY:-rmbench}"
export HIMEM_RMBENCH_REQUEST_TIMEOUT="${HIMEM_RMBENCH_REQUEST_TIMEOUT:-120.0}"
export HIMEM_RMBENCH_CHECKPOINT_PATH="${HIMEM_RMBENCH_CHECKPOINT_PATH:-}"
export HIMEM_RMBENCH_PLAN_ONLY="${HIMEM_RMBENCH_PLAN_ONLY:-0}"
export HIMEM_SERVER_URI="${HIMEM_SERVER_URI:-ws://127.0.0.1:9000}"

if [ "${HIMEM_RMBENCH_DRY_RUN:-0}" = "1" ]; then
  env | sort | grep -E '^(HIMEM_RMBENCH_|HIMEM_SERVER_URI=)'
  exit 0
fi

cd "$repo_root"

if [ ! -d "$HIMEM_RMBENCH_ROOT" ]; then
  printf '[rmbench-eval] ERROR: RMBench root does not exist: %s\n' "$HIMEM_RMBENCH_ROOT" >&2
  exit 2
fi

mkdir -p "$HIMEM_RMBENCH_LOG_DIR" \
  "$(dirname "$HIMEM_RMBENCH_PLAN_FILE")" \
  "$(dirname "$HIMEM_RMBENCH_MANIFEST_FILE")"

"$python_bin" scripts/setup/install_rmbench_policy_adapter.py \
  --rmbench-root "$HIMEM_RMBENCH_ROOT" \
  --force

"$python_bin" scripts/report/write_rmbench_run_manifest.py \
  --output "$HIMEM_RMBENCH_MANIFEST_FILE" \
  --repo-root "."

IFS=',' read -r -a rmbench_tasks <<< "$HIMEM_RMBENCH_TASKS"

plan_args=(
  scripts/eval/plan_rmbench_eval.py
  --rmbench-root "$HIMEM_RMBENCH_ROOT"
  --output "$HIMEM_RMBENCH_PLAN_FILE"
  --mode direct
  --policy-name "$HIMEM_RMBENCH_POLICY_NAME"
  --task-config "$HIMEM_RMBENCH_TASK_CONFIG"
  --ckpt-setting "$HIMEM_RMBENCH_CKPT_SETTING"
  --seed "$HIMEM_RMBENCH_SEED"
  --gpu-id "$HIMEM_RMBENCH_GPU_ID"
  --instruction-type "$HIMEM_RMBENCH_INSTRUCTION_TYPE"
  --action-horizon "$HIMEM_RMBENCH_ACTION_HORIZON"
  --override "server_uri=$HIMEM_SERVER_URI"
  --override "request_timeout=$HIMEM_RMBENCH_REQUEST_TIMEOUT"
  --override "action_dim=$HIMEM_RMBENCH_ACTION_DIM"
  --override "action_type=$HIMEM_RMBENCH_ACTION_TYPE"
  --override "state_source=$HIMEM_RMBENCH_STATE_SOURCE"
  --override "robot_key=$HIMEM_RMBENCH_ROBOT_KEY"
  --tasks
)
for task in "${rmbench_tasks[@]}"; do
  if [ -n "$task" ]; then
    plan_args+=("$task")
  fi
done
if [ -n "$HIMEM_RMBENCH_CHECKPOINT_PATH" ]; then
  plan_args+=(--override "checkpoint_path=$HIMEM_RMBENCH_CHECKPOINT_PATH")
fi
"$python_bin" "${plan_args[@]}"

if [ "$HIMEM_RMBENCH_PLAN_ONLY" = "1" ]; then
  printf '[rmbench-eval] Plan-only mode finished: %s\n' "$HIMEM_RMBENCH_PLAN_FILE"
  exit 0
fi

for task in "${rmbench_tasks[@]}"; do
  if [ -z "$task" ]; then
    continue
  fi
  log_file="$HIMEM_RMBENCH_LOG_DIR/${task}.log"
  printf '[rmbench-eval] Running task %s, log=%s\n' "$task" "$log_file"
  command=(
    "$python_bin"
    script/eval_policy.py
    --config "policy/$HIMEM_RMBENCH_POLICY_NAME/deploy_policy.yml"
    --overrides
    --task_name "$task"
    --task_config "$HIMEM_RMBENCH_TASK_CONFIG"
    --ckpt_setting "$HIMEM_RMBENCH_CKPT_SETTING"
    --seed "$HIMEM_RMBENCH_SEED"
    --policy_name "$HIMEM_RMBENCH_POLICY_NAME"
    --instruction_type "$HIMEM_RMBENCH_INSTRUCTION_TYPE"
    --action_horizon "$HIMEM_RMBENCH_ACTION_HORIZON"
    --server_uri "$HIMEM_SERVER_URI"
    --request_timeout "$HIMEM_RMBENCH_REQUEST_TIMEOUT"
    --action_dim "$HIMEM_RMBENCH_ACTION_DIM"
    --action_type "$HIMEM_RMBENCH_ACTION_TYPE"
    --state_source "$HIMEM_RMBENCH_STATE_SOURCE"
    --robot_key "$HIMEM_RMBENCH_ROBOT_KEY"
  )
  if [ -n "$HIMEM_RMBENCH_CHECKPOINT_PATH" ]; then
    command+=(--checkpoint_path "$HIMEM_RMBENCH_CHECKPOINT_PATH")
  fi
  (
    cd "$HIMEM_RMBENCH_ROOT"
    export CUDA_VISIBLE_DEVICES="$HIMEM_RMBENCH_GPU_ID"
    export PYTHONWARNINGS="ignore::UserWarning"
    export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
    "${command[@]}"
  ) 2>&1 | tee "$log_file"
done
