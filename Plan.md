# HiMem-Bridge-VLA Current Plan

This workspace is the active project for BridgeAttention + HiMem VLA experiments.
Any older sibling checkout should be treated as a legacy reference until it is explicitly archived or deleted.

## Active Entry Points

- Design: `docs/bridge_himem_design.md`
- Engineering boundaries: `docs/project_structure.md`
- Bridge-HiMem configs: `configs/bridge_himem/base.yaml` and `configs/bridge_himem/experiments/*.yaml`
- Training: `scripts/train.py`
- Model composition: `himem_bridge_vla/model/himem_bridge_vla.py`
- Local quality gate: `scripts/check_repo.sh`

## Current Experiment Matrix

- `baseline.yaml`: fused-token baseline, no bridge, no memory. This is a control, not a target model.
- `crosskv_clean.yaml`: target route A; memory enters BridgeAttention as cross-attention K/V.
- `mixed_latent_clean.yaml`: target route B; memory is appended to action-head context tokens.
- `mixed_latent_skill.yaml`: target route B plus learnable skill tokens.

All experiment YAML files inherit `configs/bridge_himem/base.yaml`. New experiments should only
override the fields that define the experimental difference.

## Initialization Decision Point

If a compatible Evo VLA checkpoint is available, prefer using it as the shared initialization for
all CALVIN finetuning branches:

```text
Evo checkpoint
  -> baseline.yaml control
  -> crosskv_clean.yaml
  -> mixed_latent_clean.yaml / mixed_latent_skill.yaml
```

If the Evo checkpoint only contains the original VLM/action-head stack, add a partial pretrain
loader before training Bridge/HiMem variants. The current strict DeepSpeed resume path expects
matching module keys and is only suitable for checkpoints from the same model architecture.

## Reproducibility Rules

- Run configs through `python3 scripts/validate_bridge_himem_configs.py` before training.
- Training writes `resolved_config.json` and `reproducibility.json` into `save_dir`.
- Use the run directory snapshot as the source of truth for reproduced runs, not the raw YAML alone.
- Keep large outputs, datasets, checkpoints, and model caches off the system disk on the server.

## Cleanup Policy

- Generated caches such as `__pycache__/`, `.pytest_cache/`, logs, checkpoints, and run outputs are
  ignored and may be deleted locally.
- Do not delete any sibling legacy checkout until it is intentionally archived or removed, because it may
  contain a separate `.git` history.
