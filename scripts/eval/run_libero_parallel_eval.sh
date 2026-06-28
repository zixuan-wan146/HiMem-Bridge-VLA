#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

source "$script_dir/libero_profile.sh"
load_libero_profile "$repo_root"

fail() {
  printf '[libero-parallel-eval] ERROR: %s\n' "$*" >&2
  exit 2
}

positive_int() {
  case "$2" in
    ''|*[!0-9]*) fail "$1 must be a positive integer: $2" ;;
  esac
  if [ "$2" -le 0 ]; then
    fail "$1 must be a positive integer: $2"
  fi
}

non_negative_int() {
  case "$2" in
    ''|*[!0-9]*) fail "$1 must be a non-negative integer: $2" ;;
  esac
}

require_env() {
  if [ -z "${!1+x}" ] || [ -z "${!1}" ]; then
    fail "$1 must be set in HIMEM_LIBERO_PROFILE or the environment"
  fi
}

run_dir="${HIMEM_LIBERO_RUN_DIR:-run_outputs/libero_parallel_eval}"
case "$run_dir" in
  /*) fail "HIMEM_LIBERO_RUN_DIR must be project-relative: $run_dir" ;;
esac

require_env HIMEM_LIBERO_TOTAL_EPISODES
require_env HIMEM_LIBERO_PARALLEL_CLIENTS
require_env HIMEM_LIBERO_EPISODE_OFFSET
require_env HIMEM_LIBERO_CKPT_NAME

total_episodes="$HIMEM_LIBERO_TOTAL_EPISODES"
parallel_clients="$HIMEM_LIBERO_PARALLEL_CLIENTS"
base_episode_offset="$HIMEM_LIBERO_EPISODE_OFFSET"
base_ckpt_name="$HIMEM_LIBERO_CKPT_NAME"

positive_int "HIMEM_LIBERO_TOTAL_EPISODES" "$total_episodes"
positive_int "HIMEM_LIBERO_PARALLEL_CLIENTS" "$parallel_clients"
non_negative_int "HIMEM_LIBERO_EPISODE_OFFSET" "$base_episode_offset"

base_count=$((total_episodes / parallel_clients))
remainder=$((total_episodes % parallel_clients))

if [ "${HIMEM_LIBERO_DRY_RUN:-0}" = "1" ]; then
  printf 'HIMEM_LIBERO_RUN_DIR=%s\n' "$run_dir"
  printf 'HIMEM_LIBERO_TOTAL_EPISODES=%s\n' "$total_episodes"
  printf 'HIMEM_LIBERO_PARALLEL_CLIENTS=%s\n' "$parallel_clients"
  printf 'HIMEM_LIBERO_EPISODE_OFFSET=%s\n' "$base_episode_offset"
fi

cd "$repo_root"
mkdir -p "$run_dir/logs"

pids=()
cursor=$base_episode_offset
for index in $(seq 0 $((parallel_clients - 1))); do
  count=$base_count
  if [ "$index" -lt "$remainder" ]; then
    count=$((count + 1))
  fi
  if [ "$count" -le 0 ]; then
    continue
  fi

  client_run_dir="$run_dir/client_${index}_offset_${cursor}_episodes_${count}"
  client_ckpt_name="${base_ckpt_name}_client${index}_offset${cursor}_n${count}"

  if [ "${HIMEM_LIBERO_DRY_RUN:-0}" = "1" ]; then
    printf 'CLIENT_%s_HIMEM_LIBERO_RUN_DIR=%s\n' "$index" "$client_run_dir"
    printf 'CLIENT_%s_HIMEM_LIBERO_EPISODES=%s\n' "$index" "$count"
    printf 'CLIENT_%s_HIMEM_LIBERO_EPISODE_OFFSET=%s\n' "$index" "$cursor"
    printf 'CLIENT_%s_HIMEM_LIBERO_CKPT_NAME=%s\n' "$index" "$client_ckpt_name"
  else
    (
      export HIMEM_LIBERO_RUN_DIR="$client_run_dir"
      export HIMEM_LIBERO_EPISODES="$count"
      export HIMEM_LIBERO_EPISODE_OFFSET="$cursor"
      export HIMEM_LIBERO_CKPT_NAME="$client_ckpt_name"
      exec bash scripts/eval/run_libero_eval.sh
    ) > "$run_dir/logs/client_${index}.stdout.log" 2>&1 &
    pids+=("$!")
  fi

  cursor=$((cursor + count))
done

if [ "${HIMEM_LIBERO_DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
