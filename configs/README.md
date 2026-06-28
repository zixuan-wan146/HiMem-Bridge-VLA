# Configs

This directory contains reusable, checked-in configuration only. Runtime outputs and machine-local paths should not be written here.

## Bridge-HiMem

```text
models/bridge_himem/
  base.yaml                         Shared defaults for current direct bridge-attn experiments
experiments/bridge_himem/
  baseline.yaml                     Fused-token baseline
  direct_progress_w4.yaml           Direct bridge with frozen W4 progress planner
  crosskv_clean.yaml                Legacy cross-attention bridge baseline
  mixed_latent_clean.yaml           Legacy mixed-latent bridge baseline
  mixed_latent_skill.yaml           Mixed-latent plus learnable skill tokens
```

Rules:

- Experiment files use `extends` and should only override the fields that define the ablation.
- Shared dimensions, VLM raw layers `[3, 6, 9, 12]`, direct bridge-attn defaults, and planner defaults live in `models/bridge_himem/base.yaml`.
- Current direct bridge uses 32 noisy action tokens from the flow-matching horizon, not learned intermediate bridge tokens.
- Progress planner output remains one base token and is expanded to 8 action-condition plan slots inside the direct action head.
- Validate before training with `python scripts/quality/validate_bridge_himem_configs.py`.

## LIBERO Profiles

```text
runtime/libero_profiles/
  smoke.env      Minimal smoke run
  full_eval.env  Default full evaluation profile
  single_task_20_parallel.env  One LIBERO task, 20 total episodes split across 4 clients
```

Profile files are plain `KEY=VALUE` files parsed by the LIBERO run scripts. They are not shell scripts and should not contain secrets.

## Dataset Configs

```text
datasets/
  simulation.yaml  Generic LeRobot-style simulation training data
```

Relative dataset paths in these YAML files are resolved from `--dataset_config_base_dir`, which defaults to the repository root in the active training entrypoint.

## Training Profiles

Training profiles keep experiment hyperparameters out of shell commands. Use CLI arguments only for machine-local overrides such as `--save_dir`, `--cache_dir`, `--resume_path`, or one-off ablations.

Validate profiles before training with `python scripts/quality/validate_training_configs.py`.

Current active Stage 1 profile:

```text
training/stage1/libero/libero_10_direct_progress_w4.yaml
```

This template expects a repo-local symlink `local_data -> $AUTODL_TMP` on the remote server. The resolved cache manifest is:

```text
local_data/token_caches/libero_10_episode_feature_internvl3_hidden_l3_6_9_12_stride16/manifest.json
```

This profile is for the current episode-level fixed-replan-node Stage1 path. `batch_size` counts episodes, so `batch_size: 1` means each optimizer step processes one full episode, advances the frozen progress-state planner through that episode's fixed replan nodes in chronological order, and computes flow-matching loss on every node with a full `horizon=32` action chunk. With the current LIBERO-10 cache of 500 episodes, `max_steps: 5000` is roughly 10 passes over the episode set. Resume-specific values such as `--resume_path` and the `500 -> 5000` total-step conversion should stay as CLI overrides.

Distributed training configs are intentionally absent. The active training path is single-card only.
