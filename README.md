# HiMem-Bridge-VLA

This repository contains a HiMem + BridgeAttention VLA adapter stack built around an InternVL3
embedder and a FlowMatching action head. The project is organized as a normal Python package plus
checked-in configs, scripts, evaluation clients, and tests.

The repository still includes legacy simulation entry points:

- MetaWorld evaluation
- LIBERO evaluation
- HiMem-Bridge-VLA model server
- training on simulation datasets

Real-robot examples and external robot-integration code have been removed from this workspace.

## Project Layout

```text
himem_bridge_vla/
  bridge_himem_config.py   Bridge-HiMem YAML schema and validation
  experiment_config.py     Shared experiment config resolver
  reproducibility.py       Seed and run snapshot helpers
  dataset/                  Training dataset loaders and adapters
  model/                    HiMem-Bridge-VLA model components
  runtime_config.py         Shared runtime constants
  utils/                    Shared helpers

configs/
  bridge_himem/             Bridge-HiMem base config and experiment overlays
  datasets/                 Training dataset configs
  deepspeed/                DeepSpeed configs
  libero_profiles/          Reusable LIBERO smoke/full-eval profiles
  calvin_profiles/          Reusable CALVIN smoke/full-eval profiles
evaluations/libero/          LIBERO simulation evaluation client
evaluations/calvin/          CALVIN simulation evaluation client
evaluations/metaworld/       Legacy MetaWorld simulation evaluation client
scripts/                    Training, server, checks, LIBERO run tooling
deepspeed_setup_example.txt Accelerate/DeepSpeed setup reference
```

See `docs/project_structure.md` for the engineering ownership rules. New experiment knobs should
go through YAML configs and `himem_bridge_vla/experiment_config.py`, not directly into model or training code.
See `configs/README.md` and `scripts/README.md` for directory-level entry points.

## Installation

```bash
conda create -n HiMem python=3.10 -y
conda activate HiMem

cd .
pip install -r requirements.txt
pip install -e .

# Adjust MAX_JOBS for your machine if needed.
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
```

## Development Checks

Install the lightweight development dependencies from the repository root:

```bash
pip install -r requirements-dev.txt
scripts/check_repo.sh
```

The lightweight tests avoid downloading model weights. Tests that require PyTorch are skipped when
PyTorch is not installed.

`scripts/check_repo.sh` runs the local quality gate: dependency policy audit, unit tests, optional
`ruff`, shell syntax checks, repository preflight, LIBERO setup dry-run, LIBERO checkpoint download
dry-run, Bridge-HiMem config validation, LIBERO smoke/full-eval profile dry-runs, CALVIN eval
profile dry-run, `compileall`, and `git diff --check`. Set
`HIMEM_CHECK_REQUIRE_RUFF=1` in CI or a fully prepared dev environment to make missing `ruff` fail
instead of warn.

Bridge-HiMem YAML configs can be validated without loading model weights:

```bash
python3 scripts/validate_bridge_himem_configs.py
```

Bridge-HiMem experiment configs live under `configs/bridge_himem/experiments/` and inherit
`configs/bridge_himem/base.yaml` with `extends`. Training writes `resolved_config.json` and
`reproducibility.json` into `save_dir`, so a run directory records the final merged config and seed.

`scripts/audit_requirements.py` fails when a new `requirements*.txt` file is not covered by
`requirements-policy.json`, or when a dependency is left unpinned without an explicit reason.
Existing unpinned HiMem-Bridge-VLA runtime dependencies are recorded as known follow-up work until the exact
GPU server wheel set is captured.

If the GitHub remote is temporarily unavailable or your account lacks write permission, export the
local commits as a portable patch bundle:

```bash
scripts/export_unpushed_commits.sh
```

The export is written under `exports/` and can be applied to another clone with
`git am /path/to/export/patches/*.patch`.

## MetaWorld Evaluation

Create a separate environment for MetaWorld:

```bash
conda create -n metaworld python=3.10 -y
conda activate metaworld
pip install mujoco metaworld websockets opencv-python packaging huggingface_hub
```

Download the checkpoint:

```bash
hf download MINT-SJTU/HiMem_MetaWorld --local-dir /path/to/checkpoint
```

Start the HiMem-Bridge-VLA server:

```bash
conda activate HiMem
cd .
python scripts/himem_server.py --ckpt_dir /path/to/checkpoint --port 9000
```

The WebSocket request must be a JSON object with:

- `image`: exactly 3 RGB image arrays with pixel values in `0..255`; images are resized by the server.
- `state`: a non-empty finite numeric vector with length at most 24.
- `image_mask`: 0/1 mask with length at most 3; shorter masks are padded with zeros.
- `action_mask`: 0/1 mask with length at most 24; shorter masks are padded with zeros and at least one dimension must be active.
- `prompt`: optional task instruction string.

Run the MetaWorld client:

```bash
conda activate metaworld
cd evaluations/metaworld
python mt50_himem_client_prompt.py
```

The MetaWorld client uses `ws://127.0.0.1:9000` by default. Change `SERVER_URL` in
`evaluations/metaworld/mt50_himem_client_prompt.py` if you run the server elsewhere.

Common MetaWorld client settings can also be overridden without editing source code:

```bash
export HIMEM_SERVER_URI=ws://127.0.0.1:9000
export HIMEM_MT50_EPISODES=1
export HIMEM_MT50_EPISODE_HORIZON=100
export HIMEM_MT50_SAVE_VIDEO=false
python mt50_himem_client_prompt.py
```

## LIBERO Evaluation

Recommended setup for a server with a data disk:

```bash
HIMEM_DATA_ROOT=/root/autodl-tmp \
CONDA_BIN=/root/miniconda3/bin/conda \
scripts/setup_libero_env.sh
```

The setup script creates a Python 3.8.13 LIBERO environment, installs `libero==0.1.1`,
downloads LIBERO assets, configures `~/.libero/config.yaml`, and installs the headless
MuJoCo system libraries when run as root on Ubuntu.

The script installs top-level LIBERO packages from `requirements-libero.txt`. To validate resolved
paths without creating a conda environment or downloading assets:

```bash
HIMEM_SETUP_LIBERO_DRY_RUN=1 scripts/setup_libero_env.sh
```

Use `HIMEM_LIBERO_REQUIREMENTS=/path/to/requirements.txt` only when deliberately testing another
LIBERO dependency set.

Download the checkpoint:

```bash
HIMEM_DATA_ROOT=/root/autodl-tmp scripts/download_libero_checkpoint.sh
```

Start the HiMem-Bridge-VLA server:

```bash
HIMEM_PYTHON=/root/autodl-tmp/miniforge3/envs/HiMem/bin/python \
scripts/start_himem_server.sh /path/to/checkpoint
```

`scripts/start_himem_server.sh` runs a lightweight checkpoint preflight before loading the model.
Set `HIMEM_SKIP_PREFLIGHT=1` only when deliberately bypassing that check for debugging.
`scripts/download_libero_checkpoint.sh` writes to `$HIMEM_DATA_ROOT/checkpoints/HiMem_LIBERO` by
default. It does not set a Hugging Face mirror by default; if a single external download needs one,
use `HIMEM_HF_ENDPOINT=https://hf-mirror.com` only on that command.

Run the minimal LIBERO smoke client from another shell:

```bash
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python \
scripts/run_libero_smoke.sh
```

To keep one run's logs, videos, and result JSON together, set a run directory:

```bash
HIMEM_LIBERO_RUN_DIR=/root/autodl-tmp/himem_runs/libero_smoke_001 \
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python \
scripts/run_libero_smoke.sh
```

The run directory layout is `logs/`, `videos/`, `results/`, and `run_manifest.json`.
The manifest is written before the client starts, so failed or interrupted runs still keep the
resolved LIBERO settings, output paths, Git commit, dirty state, command, Python version, and
selected non-secret environment variables.

Reusable LIBERO settings can be stored in a profile:

```bash
HIMEM_LIBERO_PROFILE=configs/libero_profiles/smoke.env \
HIMEM_LIBERO_RUN_DIR=/root/autodl-tmp/himem_runs/libero_smoke_001 \
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python \
scripts/run_libero_smoke.sh
```

Profile files use plain `KEY=VALUE` lines and are parsed without executing shell code. Only
LIBERO-related allowlisted keys are accepted, and explicit environment variables still override
profile values.

Run the full default LIBERO evaluation when you are ready to collect comparable numbers:

```bash
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python \
scripts/run_libero_eval.sh
```

`scripts/run_libero_eval.sh` defaults to all four LIBERO suites, `HIMEM_LIBERO_HORIZON=14`,
`HIMEM_LIBERO_EPISODES=10`, and max steps `25,25,25,95`. Set `HIMEM_LIBERO_DRY_RUN=1` to print the
resolved eval environment without running the client.
Set `HIMEM_LIBERO_RUN_DIR=/path/to/run` to use the same grouped output layout as smoke runs.
Use `HIMEM_LIBERO_PROFILE=configs/libero_profiles/full_eval.env` to make the full-eval settings
explicit in command logs.

Before running on a server, generate a reproducible command plan:

```bash
python scripts/plan_libero_run.py \
  --kind eval \
  --run-dir /root/autodl-tmp/himem_runs/libero_eval_001 \
  --checkpoint /root/autodl-tmp/checkpoints/HiMem_LIBERO \
  --profile configs/libero_profiles/full_eval.env \
  --server-python /root/autodl-tmp/miniforge3/envs/HiMem/bin/python \
  --libero-python /root/autodl-tmp/envs/libero/bin/python \
  --min-total-episodes 10
```

The plan file includes the server command, LIBERO client command, artifact validation command, and
report command with the same paths and profile.

For a tracked baseline or candidate improvement, initialize an experiment directory before running:

```bash
python scripts/init_libero_experiment.py \
  --name baseline_full_eval_001 \
  --root /root/autodl-tmp/himem_experiments \
  --kind eval \
  --checkpoint /root/autodl-tmp/checkpoints/HiMem_LIBERO \
  --profile configs/libero_profiles/full_eval.env \
  --server-python /root/autodl-tmp/miniforge3/envs/HiMem/bin/python \
  --libero-python /root/autodl-tmp/envs/libero/bin/python \
  --min-total-episodes 10
```

The experiment directory contains a profile snapshot, `run_plan.md`, `notes.md`,
`experiment_manifest.json`, a planned `run/` directory for LIBERO artifacts, and a planned `report/`
directory for summaries and metric gates. The script refuses to write into a non-empty experiment
directory, so old results are not silently overwritten.

The LIBERO client stores logs, videos, and a machine-readable result summary under
`evaluations/libero/`.

Common LIBERO client settings can be overridden without editing source code:

```bash
export HIMEM_SERVER_URI=ws://127.0.0.1:9000
export HIMEM_MUJOCO_GL=osmesa
export HIMEM_LIBERO_EPISODES=1
export HIMEM_LIBERO_TASK_SUITES=libero_spatial
export HIMEM_LIBERO_TASK_LIMIT=1
export HIMEM_LIBERO_MAX_STEPS=25
export HIMEM_LIBERO_RESULT_FILE="$PWD/evaluations/libero/log_file/libero_spatial_results.json"
export HIMEM_LIBERO_MANIFEST_FILE="$PWD/evaluations/libero/log_file/libero_spatial_run_manifest.json"
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python scripts/run_libero_smoke.sh
```

The result summary JSON contains run metadata, evaluated episodes, per-suite success rates, and
failure reasons such as action parsing errors or step-limit exhaustion. Metadata includes the
current Git commit, dirty state, command, Python version, and selected non-secret environment
variables.

To compare one or more LIBERO runs after evaluation:

```bash
python scripts/summarize_libero_results.py evaluations/libero/log_file/*_results.json \
  --output outputs/libero_results.md
python scripts/summarize_libero_results.py evaluations/libero/log_file/*_results.json \
  --format csv \
  --output outputs/libero_results.csv
```

The comparison table includes the run name, Git commit, dirty state, overall metrics, and per-suite
metrics when present in the result JSON.

To inventory grouped run directories, including interrupted runs that only have a manifest:

```bash
python scripts/summarize_libero_results.py /root/autodl-tmp/himem_runs \
  --table runs \
  --output outputs/libero_run_inventory.md
```

The run inventory table reports each run directory, completeness status, manifest settings, result
path, Git metadata, and overall success metrics when a result JSON exists.

To gate a candidate result before treating it as an improvement:

```bash
python scripts/check_libero_metrics.py /root/autodl-tmp/himem_runs/candidate \
  --min-success-rate 0.10 \
  --min-total-episodes 10
python scripts/check_libero_metrics.py /root/autodl-tmp/himem_runs/candidate \
  --baseline /root/autodl-tmp/himem_runs/baseline \
  --max-regression 0.02
```

The metric gate defaults to the `overall` scope. Add `--scope suite:libero_spatial` or repeat
`--scope` to gate suite-level metrics.

To generate a report bundle for a set of runs:

```bash
python scripts/report_libero_runs.py /root/autodl-tmp/himem_runs \
  --output-dir outputs/libero_report \
  --min-success-rate 0.10 \
  --min-total-episodes 10
```

The report directory contains run inventory tables, result summary tables, a human-readable
`README.md`, a machine-readable report manifest, and a metric gate log when gate options are
provided.

For headless smoke tests, `HIMEM_MUJOCO_GL=osmesa` is the more stable default. Use
`HIMEM_MUJOCO_GL=egl` on GPU servers when EGL cleanup warnings are acceptable and
faster rendering is preferred.

Before running a longer evaluation, use the lightweight preflight checks:

```bash
python scripts/preflight.py \
  --checkpoint /path/to/checkpoint \
  --check-imports himem
```

The checkpoint check validates required files, basic `config.json` dimensions, and `norm_stats.json`
state/action min-max structure without loading model weights.

After evaluation, validate result JSON files and run manifests before summarizing or syncing them:

```bash
python scripts/preflight.py \
  --dataset-config "" \
  --libero-result "evaluations/libero/log_file/*_results.json" \
  --libero-manifest "evaluations/libero/log_file/*_run_manifest.json"
```

The result check verifies both schema and consistency between overall/per-suite summaries and the
episode records. The manifest check verifies run kind, key LIBERO settings, Git metadata, and that
recorded environment variables do not include common secret fields.

If the run used `HIMEM_LIBERO_RUN_DIR`, validate the whole run directory instead:

```bash
python scripts/preflight.py \
  --dataset-config "" \
  --libero-run-dir /root/autodl-tmp/himem_runs/libero_smoke_001
```

The run-directory check validates both files and verifies that the manifest points to the result
JSON from the same run, with matching checkpoint name and Git metadata.

For a strict training-data check, add `--strict-data` after downloading the dataset.

## Training

Download the example MetaWorld dataset:

```bash
mkdir HiMem_training_dataset
cd HiMem_training_dataset
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/MINT-SJTU/HiMem_MetaWorld_Dataset
cd HiMem_MetaWorld_Dataset
git lfs pull
```

Configure the dataset path in `configs/datasets/simulation.yaml`. The default path expects:

```text
HiMem_training_dataset/HiMem_MetaWorld_Dataset
```

Before starting a training run on a new dataset copy, validate the dataset structure from the
repository root:

```bash
python scripts/validate_training_dataset.py \
  --dataset-config configs/datasets/simulation.yaml \
  --dataset-base-dir .
```

The validator checks `tasks.jsonl`, `episodes.jsonl`, `stats.json` or `episodes_stats.jsonl`,
`data/*/*.parquet`, and expected video paths derived from the dataset `view_map`.

Run stage 1 training:

```bash
conda activate HiMem
cd .

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file configs/deepspeed/ds_config.json scripts/train.py \
  --run_name HiMem_metaworld_stage1 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 5000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path configs/datasets/simulation.yaml \
  --dataset_config_base_dir . \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /path/to/checkpoints/stage1
```

Run stage 2 training:

```bash
conda activate HiMem
cd .

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file configs/deepspeed/ds_config.json scripts/train.py \
  --run_name HiMem_metaworld_stage2 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 80000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_vlm \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path configs/datasets/simulation.yaml \
  --dataset_config_base_dir . \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /path/to/checkpoints/stage2 \
  --resume \
  --resume_pretrain \
  --resume_path /path/to/checkpoints/stage1/step_5000
```

Relative dataset paths inside `configs/datasets/*.yaml` are resolved from `--dataset_config_base_dir`.
The examples above use `.` because they run from the repository root.
Use `--cache_dir /path/to/cache` if you want the generated training cache outside the project directory.

## Remote Deployment Notes

On remote servers with a separate data disk, keep code, checkpoints, caches, and outputs outside the
system disk. For example:

```bash
cd /root/autodl-tmp
git clone https://github.com/zixuan-wan146/HiMem-Bridge-VLA.git
export HF_HOME=/root/autodl-tmp/hf-home
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
```

Only set `HF_ENDPOINT=https://hf-mirror.com` for a single Hugging Face download command when that
specific external download benefits from it. Do not put it in shell startup files or global env
configuration, because it can slow down downloads from domestic resources.

Download checkpoints to the data disk:

```bash
hf download MINT-SJTU/HiMem_MetaWorld --local-dir /root/autodl-tmp/checkpoints/HiMem_MetaWorld --max-workers 1
hf download MINT-SJTU/HiMem_LIBERO --local-dir /root/autodl-tmp/checkpoints/HiMem_LIBERO --max-workers 1
```

If `flash-attn` installation fails with a cross-device link error, set `TMPDIR` to the same data disk:

```bash
mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
pip install flash-attn --no-build-isolation
```

## Citation

```bibtex
@article{lin2025himem,
  title={HiMem-Bridge-VLA: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment},
  author={Lin, Tao and Zhong, Yilei and Du, Yuxin and Zhang, Jingjing and Liu, Jiting and Chen, Yinxinyu and Gu, Encheng and Liu, Ziyan and Cai, Hongyi and Zou, Yanwen and others},
  journal={arXiv preprint arXiv:2511.04555},
  year={2025}
}
```
