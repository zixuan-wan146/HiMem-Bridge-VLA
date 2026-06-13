# Configs

This directory contains reusable, checked-in configuration only. Runtime outputs and machine-local
paths should not be written here.

## Bridge-HiMem

```text
bridge_himem/
  base.yaml                 Shared defaults for Bridge-HiMem experiments
  experiments/
    baseline.yaml           Fused-token baseline
    crosskv_clean.yaml      Memory enters BridgeAttention cross-attention
    mixed_latent_clean.yaml Memory enters action-head context
    mixed_latent_skill.yaml Mixed-latent plus learnable skill tokens
```

Rules:

- Experiment files use `extends` and should only override the fields that define the ablation.
- Shared dimensions, raw VLM layers, writer settings, and segment accumulator defaults live in
  `bridge_himem/base.yaml`.
- Validate before training:

```bash
python3 scripts/validate_bridge_himem_configs.py
```

## LIBERO Profiles

```text
libero_profiles/
  smoke.env      Minimal smoke run
  full_eval.env  Default full evaluation profile
```

Profile files are plain `KEY=VALUE` files parsed by the LIBERO run scripts. They are not shell
scripts and should not contain secrets.

## CALVIN Profiles

```text
calvin_profiles/
  smoke.env      One-sequence smoke run
  full_eval.env  Default 1000-sequence CALVIN ABC->D profile
```

CALVIN profiles are parsed by `scripts/run_calvin_eval.sh` without executing shell code. Keep
machine-local paths in environment variables when they differ from the defaults.

## Dataset Configs

```text
datasets/
  simulation.yaml  Generic LeRobot-style simulation training data
  calvin.yaml      CALVIN LeRobot-style training data
```

Relative dataset paths in these YAML files are resolved from `--dataset_config_base_dir`, which
defaults to the repository root in `scripts/train.py`.

## Training Profiles

```text
training/
  calvin_stage1.yaml  FlowMatching warm-up profile
  calvin_stage2.yaml  Bridge-HiMem fine-tuning profile
```

Training profiles keep experiment hyperparameters out of shell commands. Use CLI arguments only for
machine-local overrides such as `--save_dir`, `--cache_dir`, `--resume_path`, or one-off ablations.
The default cache path is `run_outputs/training_data_cache`; cache entries are automatically
namespaced by the dataset config and action horizon.

Validate profiles before training:

```bash
python3 scripts/validate_training_configs.py
```

## DeepSpeed

```text
deepspeed/
  ds_config.json
```
