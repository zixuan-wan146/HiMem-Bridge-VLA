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
configs/README.md                             Checked-in config rules
scripts/README.md                             Script entry points
```

## Project Layout

```text
himem_bridge_vla/   package code: configs, dataset loaders, model modules, runtime helpers
configs/            checked-in Bridge-HiMem, dataset, DeepSpeed, LIBERO profile configs
evaluations/libero/ LIBERO client, action protocol, result handling
evaluations/rmbench/ RMBench adapter and eval-planning helpers
scripts/            training, server, checks, LIBERO/RMBench tooling
tests/              lightweight tests that avoid downloading model weights
referen-repo/       historical tracked reference repositories; kept in place to avoid churn
reference-repo/     newly added source-only external references, such as VLA-Adapter
```

Large datasets, model caches, checkpoints, and run outputs stay outside git on the remote data disk. On AutoDL this project uses `$AUTODL_TMP` as the data and run root.

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
