#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-home/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON:-/root/autodl-tmp/miniforge3/envs/Evo1/bin/python}"
device="${COARSE_PLANNER_DEVICE:-cuda}"
build_config="${COARSE_PLANNER_BUILD_CONFIG:-coarse_planner/configs/libero_horizon_ablation_build.yaml}"
batch_size="${COARSE_PLANNER_BATCH_SIZE:-480}"
epochs="${COARSE_PLANNER_EPOCHS:-8}"

if [[ "${COARSE_PLANNER_SKIP_BUILD:-0}" != "1" ]]; then
  "$python_bin" -m coarse_planner.build_from_libero \
    --config "$build_config" \
    --device "$device"
fi

run_dirs=()
for horizon in 32 48 64; do
  config="coarse_planner/configs/libero_h${horizon}.yaml"
  run_dir="/root/autodl-tmp/runs/coarse_planner/libero_h${horizon}"
  checkpoint="/root/autodl-tmp/checkpoints/coarse_planner/libero_h${horizon}.pt"
  "$python_bin" -m coarse_planner.train \
    --config "$config" \
    --run-dir "$run_dir" \
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
  --output /root/autodl-tmp/runs/coarse_planner/libero_horizon_ablation_report.md
