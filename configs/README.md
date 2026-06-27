# Configs

This directory contains reusable, checked-in configuration only. Runtime outputs and machine-local paths should not be written here.

## Bridge-HiMem

```text
bridge_himem/
  base.yaml                         Shared defaults for current direct bridge-attn experiments
  experiments/
    baseline.yaml                   Fused-token baseline
    direct_progress_w4.yaml         Direct bridge with frozen W4 progress planner
    crosskv_clean.yaml              Legacy cross-attention bridge baseline
    mixed_latent_clean.yaml         Legacy mixed-latent bridge baseline
    mixed_latent_skill.yaml         Mixed-latent plus learnable skill tokens
    coarse_planner_crosskv.yaml     Legacy H32 planner bridge config
```

Rules:

- Experiment files use `extends` and should only override the fields that define the ablation.
- Shared dimensions, VLM raw layers `[3, 6, 9, 12]`, direct bridge-attn defaults, and planner defaults live in `bridge_himem/base.yaml`.
- Current direct bridge uses 32 noisy action tokens from the flow-matching horizon, not learned intermediate bridge tokens.
- Progress planner output remains one base token and is expanded to 8 action-condition plan slots inside the direct action head.
- Legacy H32 coarse-planner integration keeps `coarse_planner.num_plan_steps: 1`, `coarse_planner.planning_horizon: 32`, and `coarse_planner.input_memory: false`.
- Validate before training with `python scripts/validate_bridge_himem_configs.py`.

Standalone coarse-planner cache, AE, and planner configs live under `coarse_planner/configs/`, not in this directory.

## LIBERO Profiles

```text
libero_profiles/
  smoke.env      Minimal smoke run
  full_eval.env  Default full evaluation profile
```

Profile files are plain `KEY=VALUE` files parsed by the LIBERO run scripts. They are not shell scripts and should not contain secrets.

## Dataset Configs

```text
datasets/
  simulation.yaml  Generic LeRobot-style simulation training data
```

Relative dataset paths in these YAML files are resolved from `--dataset_config_base_dir`, which defaults to the repository root in `scripts/train.py`.

## Training Profiles

Training profiles keep experiment hyperparameters out of shell commands. Use CLI arguments only for machine-local overrides such as `--save_dir`, `--cache_dir`, `--resume_path`, or one-off ablations.

Validate profiles before training with `python scripts/validate_training_configs.py`.

Current checked-in template:

```text
training_templates/libero_10_stage1_direct_progress_w4.yaml
```

This template expects a repo-local symlink `local_data -> /root/autodl-tmp` on the remote server. The resolved cache manifest is:

```text
local_data/token_caches/libero_10_memory_replay_internvl3_hidden_l3_6_9_12_s16_dedup_parts4/manifest.json
```

## DeepSpeed

```text
deepspeed/
  ds_config.json
```
