#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

data_root="${AUTODL_TMP:-${HOME}/autodl-tmp}"
export HF_HOME="${HF_HOME:-$data_root/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$data_root/hf-home/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON:-$data_root/miniforge3/envs/Evo1/bin/python}"
device="${COARSE_PLANNER_DEVICE:-cuda}"
build_config="${COARSE_PLANNER_BUILD_CONFIG:-coarse_planner/configs/libero_horizon_ablation_build.yaml}"
batch_size="${COARSE_PLANNER_BATCH_SIZE:-480}"
epochs="${COARSE_PLANNER_EPOCHS:-8}"
run_root="${COARSE_PLANNER_RUN_ROOT:-$data_root/runs/coarse_planner}"
checkpoint_root="${COARSE_PLANNER_CHECKPOINT_ROOT:-$data_root/checkpoints/coarse_planner}"
report_path="${COARSE_PLANNER_REPORT_PATH:-$run_root/libero_horizon_ablation_report.md}"

mkdir -p "$run_root" "$checkpoint_root" "$(dirname "$report_path")"

if [[ "${COARSE_PLANNER_SKIP_BUILD:-0}" != "1" ]]; then
  "$python_bin" -m coarse_planner.build_from_libero \
    --config "$build_config" \
    --device "$device"
fi

run_dirs=()
for horizon in 32 48 64; do
  config="coarse_planner/configs/libero_h${horizon}.yaml"
  run_dir="$run_root/libero_h${horizon}"
  ae_run_dir="$run_root/libero_h${horizon}_segment_ae"
  checkpoint="$checkpoint_root/libero_h${horizon}.pt"
  "$python_bin" -m coarse_planner.train_segment_autoencoder \
    --config "$config" \
    --run-dir "$ae_run_dir" \
    --device "$device" \
    --batch-size "$batch_size" \
    --epochs "$epochs"
  "$python_bin" -m coarse_planner.train \
    --config "$config" \
    --run-dir "$run_dir" \
    --segment-autoencoder-checkpoint "$ae_run_dir/best.pt" \
    --device "$device" \
    --batch-size "$batch_size" \
    --epochs "$epochs"
  "$python_bin" -m coarse_planner.export \
    --checkpoint "$run_dir/best.pt" \
    --output "$checkpoint"
  run_dirs+=("$run_dir")
done

"$python_bin" -m coarse_planner.analyze_ablation \
  --runs "${run_dirs[@]}" \
  --output "$report_path"
