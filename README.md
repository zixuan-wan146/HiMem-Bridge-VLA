# HiMem-Bridge-VLA

This repository contains the active HiMem VLA work built around an InternVL3 embedder, a progress-state planner, short visual-token memory, and a direct bridge-attn flow-matching action head.

The current research direction separates short and long memory by function:

```text
short memory = independent recent visual-token memory
long memory  = planner-coupled task-progress state
```

The previous H64 suffix planner and transition-trigger refresh design remains retired. The previous Dual-FIFO long visual-memory design is also no longer the active long-memory route.

## Current Contract

The active contract is:

```text
H = 32
R = 16
S_t = ShortVisualMemory(V_{t-R/2}, V_{t-R})
x_t = ProgressEvidenceEncoder(h_t, s_t, u_t)
M_t = ProgressStateUpdater(M_{t-1}, x_t)
P_t = Planner(M_t, h_t, s_t)
P_t -> 8 virtual plan slots
32 noisy action tokens -> DirectBridgeActionHead
```

Where:

```text
S_t: short visual memory tokens
M_t: long-term task-progress state tokens
P_t: planner intent token
u_t: summary of the executed R-step action segment since the last replan
```

The direct bridge action head reads two functional context branches:

```text
visual evidence: [current VLM hidden states, short memory]
action condition: [plan slots, state token]
```

The existing H32 action-latent planner artifacts are now treated as baseline / warm-start assets, not as the main planner definition.

## Active Entry Points

```text
README.md                                      Repository overview
Plan.md                                       Current engineering plan
docs/current_project_state.md                 Current remote state and next work
docs/progress_state_planner_design_zh.md      Current long-memory and planner design
docs/project_structure.md                     Ownership boundaries and output locations
docs/engineering_reproducibility.md           Engineering and reproducibility contract
docs/benchmark_plan.md                        LIBERO / LIBERO-Plus / RMBench status
docs/bridge_himem_design.md                   Active progress planner + direct bridge model path
docs/direct_bridge_attention_design_zh.md     Direct bridge-attn action-head design
coarse_planner/README.md                      Legacy H32 baseline data/training/eval path
configs/README.md                             Checked-in config rules
scripts/README.md                             Script entry points
```

## Project Layout

```text
himem_bridge_vla/   package code: configs, dataset loaders, model modules, runtime helpers
coarse_planner/     legacy H32 action-intent cache, AE, planner training/eval baseline
configs/            checked-in Bridge-HiMem, dataset, DeepSpeed, LIBERO profile configs
evaluations/libero/ LIBERO client, action protocol, result handling
evaluations/rmbench/ RMBench adapter and eval-planning helpers
scripts/            training, server, checks, LIBERO/RMBench tooling
tests/              lightweight tests that avoid downloading model weights
referen-repo/       historical tracked reference repositories; kept in place to avoid churn
reference-repo/     newly added source-only external references, such as VLA-Adapter
```

Large datasets, model caches, checkpoints, and run outputs stay outside git on the remote data disk. On AutoDL this project uses `$AUTODL_TMP` as the data and run root.

## Existing H32 Baseline Artifacts

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
H32 planner:   best epoch 52, val_raw_latent_mse 0.0869378231
```

These artifacts can still be used for comparisons or auxiliary intent targets, but they no longer define the main planner architecture.

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

## Legacy Standalone H32 Baseline

```bash
python -m coarse_planner.build_from_libero --config coarse_planner/configs/libero_h32_single_token_build.yaml --device cuda
python -m coarse_planner.train_segment_autoencoder --config coarse_planner/configs/libero_h32_intent_ae_v1.yaml --device cuda
python -m coarse_planner.train --config coarse_planner/configs/libero_h32_single_token_planner_v1.yaml --device cuda
```

See `coarse_planner/README.md` for the legacy baseline data contract.

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
