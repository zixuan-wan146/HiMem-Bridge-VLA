# HiMem-Bridge-VLA

This repository contains the active BridgeAttention + HiMem VLA adapter stack built around an InternVL3 embedder and a FlowMatching action head. The current research path is H32 single-token coarse planning for LIBERO-style 32-step action chunks.

The previous H64 multi-token suffix planner and transition-trigger refresh design has been retired from the active code path. Do not restart that line unless a new failure case justifies a longer cached horizon.

## Current Contract

```text
P_t = CoarsePlanner(H_t, s_t)   # [B, 1, D]
ActionHead(..., P_t)            # predicts a 32-step action chunk
```

One planner call produces exactly one plan token. Every inference recomputes that token from the current observation and state. There is no plan-token queue, consumed-step suffix state, or transition-trigger refresh path.

Memory is now tracked as a separate Dual-FIFO visual-memory workstream. The current implementation step is memory-side inference construction only: entry schema, deterministic short reads, external long FIFO writes, padding masks, and view-aware query compression. BridgeAttention memory integration remains separate.

## Active Entry Points

```text
README.md                                      Repository overview
docs/current_project_state.md                 Current remote state, artifacts, metrics, next work
docs/project_structure.md                     Ownership boundaries and output locations
docs/engineering_reproducibility.md           Engineering and reproducibility contract
docs/benchmark_plan.md                        LIBERO / LIBERO-Plus / RMBench status and next work
docs/bridge_himem_design.md                   BridgeAttention + model integration design
docs/coarse_planner_design.md                 H32 single-token planner design
docs/dual_fifo_visual_memory_design_zh.md    Dual-FIFO visual-token memory design
docs/dual_fifo_visual_memory_qa_zh.md        Q&A for Dual-FIFO visual memory semantics
coarse_planner/README.md                      Standalone planner data/training/eval path
configs/README.md                             Checked-in config rules
scripts/README.md                             Script entry points
```

## Project Layout

```text
himem_bridge_vla/   package code: configs, dataset loaders, model modules, runtime helpers
coarse_planner/     standalone H32 cache, action-intent AE, planner training/eval
configs/            checked-in Bridge-HiMem, dataset, DeepSpeed, LIBERO profile configs
evaluations/libero/ LIBERO client, action protocol, result handling
scripts/            training, server, checks, LIBERO run tooling
tests/              lightweight tests that avoid downloading model weights
```

Large datasets, model caches, checkpoints, and run outputs stay outside git on the remote data disk. On AutoDL this project uses `$AUTODL_TMP` as the data and run root.

## Active H32 Artifacts

```text
feature cache: $AUTODL_TMP/datasets/coarse_planner/libero_h32_single_token_s32768_seed42
action-only:   $AUTODL_TMP/datasets/coarse_planner/libero_h32_single_token_s32768_seed42_action_only
AE run:        $AUTODL_TMP/runs/coarse_planner/libero_h32_intent_ae_v1
planner run:   $AUTODL_TMP/runs/coarse_planner/libero_h32_single_token_planner_v1
```

The feature cache has 32768 samples and uses `planning_horizon=32`, `num_plan_steps=1`, `chunk_size=32`. Training history records a 29480 / 3288 train/eval split.

Current best metrics:

```text
H32 intent AE: best epoch 99, val_loss 0.0137875769
H32 planner:   best epoch 52, val_raw_latent_mse 0.0869378231, cosine 0.9047680597
```

## Installation And Checks

```bash
conda create -n HiMem python=3.10 "numpy<2" -y
conda activate HiMem
pip install -r requirements.txt
pip install -e .
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
```

On the remote server, run practical work from the data disk. If downloading from GitHub or Hugging Face, source the network helper first and keep Hugging Face caches under `$AUTODL_TMP`:

```bash
cd $AUTODL_TMP/HiMem-Bridge-VLA
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
```

Lightweight checks:

```bash
python scripts/validate_bridge_himem_configs.py
python scripts/validate_training_configs.py
scripts/check_repo.sh
```

## Standalone Coarse Planner

```bash
python -m coarse_planner.build_from_libero --config coarse_planner/configs/libero_h32_single_token_build.yaml --device cuda
python -m coarse_planner.train_segment_autoencoder --config coarse_planner/configs/libero_h32_intent_ae_v1.yaml --device cuda
python -m coarse_planner.train --config coarse_planner/configs/libero_h32_single_token_planner_v1.yaml --device cuda
```

See `coarse_planner/README.md` and `docs/coarse_planner_design.md` for the data contract.

## Server And LIBERO

Start the model server from a trained checkpoint:

```bash
python scripts/himem_server.py --ckpt_dir checkpoints/HiMem_LIBERO --port 9000
```

The active server schema does not accept `transition_frame`; transition-trigger runtime integration has been removed from the active path.

LIBERO setup and smoke run:

```bash
HIMEM_DATA_ROOT=run_outputs/libero_data CONDA_BIN=miniconda3/bin/conda scripts/setup_libero_env.sh
HIMEM_LIBERO_PROFILE=configs/libero_profiles/smoke.env HIMEM_LIBERO_RUN_DIR=run_outputs/himem_runs/libero_smoke_001 LIBERO_PYTHON=run_outputs/libero_data/envs/libero/bin/python scripts/run_libero_smoke.sh
```
