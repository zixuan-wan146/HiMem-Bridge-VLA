# Standalone Coarse Planner

This package is the standalone training/evaluation path for the Coarse Planner
before it is attached to the full HiMem-Bridge-VLA training loop.

The active LIBERO path is H32 single-token intent planning:

```text
cache VLM tokens and robot state
build one 32-step action-intent target per sample
train the action-only intent autoencoder
freeze AE
train CoarsePlanner to predict one AE latent from one plan token
evaluate latent MSE, decoded chunk quality, and cosine similarity
use the single plan token at inference/integration time
```

The old compressed coarse-action target and the H64 multi-token/suffix queue path
are deprecated and should not be used for new experiments.

## Current Active Artifacts

Datasets on the remote data disk:

```text
../datasets/coarse_planner/libero_h32_single_token_s32768_seed42
```

Checkpoints/runs:

```text
../runs/coarse_planner/libero_h32_intent_ae_v1
../runs/coarse_planner/libero_h32_single_token_planner_v1
```

## Active Configs

Use these configs for the current LIBERO H32 single-token path:

```text
coarse_planner/configs/libero_h32_single_token_build.yaml
coarse_planner/configs/libero_h32_intent_ae_v1.yaml
coarse_planner/configs/libero_h32_single_token_planner_v1.yaml
```

## Feature Cache Format

The planner feature cache is sharded:

```text
manifest.json
train/planner_samples_00000.pt
eval/planner_samples_00001.pt
```

Each sample contains:

```text
vlm_tokens
state
action_segments
action_segment_mask
```

For LIBERO H32 single-token planning, the important target contract is:

```text
planning_horizon: 32
num_plan_steps: 1
chunk_size: 32
gripper_indices: [6]
```

## Common Commands

Build the 32k LIBERO H32 cache:

```bash
python -m coarse_planner.build_from_libero \
  --config coarse_planner/configs/libero_h32_single_token_build.yaml \
  --device cuda
```

Train the H32 intent AE:

```bash
python -m coarse_planner.train_segment_autoencoder \
  --config coarse_planner/configs/libero_h32_intent_ae_v1.yaml \
  --device cuda
```

Train the H32 single-token planner:

```bash
python -m coarse_planner.train \
  --config coarse_planner/configs/libero_h32_single_token_planner_v1.yaml \
  --device cuda
```

## Current Metrics

```text
H32 intent AE:
  best epoch:     99
  val_loss:       0.0137875769
  val_rec_loss:   0.0123513332
  val_dist_loss:  0.0143624358

H32 single-token planner:
  training stopped at the 2026-06-22 04:45 wall-clock deadline
  checkpoint selection metric: val_raw_latent_mse
  best epoch:     52
  best raw MSE:   0.0869378231
  best val loss:  0.2716650516
  best cosine:    0.9047680597
  last epoch:     60
  last raw MSE:   0.0878419157
```

## Notes

Detailed design and experiment records live in:

```text
docs/coarse_planner_design.md
docs/action_segment_autoencoder_coarse_planner_config.md
docs/current_project_state.md
```

The next training stage is not more standalone planner fine-tuning by default.
BridgeAttention / ActionHead joint training is deferred until the memory rewrite
direction is settled.

```text
P_t = f_plan(H_t, s_t)       # shape [B, 1, D]
ActionHead(H_t, s_t, P_t)    # predicts a 32-step action chunk
```
