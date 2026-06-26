# Standalone Coarse Planner

This package contains the standalone data, training, and evaluation path for the H32 single-token Coarse Planner before it is attached to the full HiMem-Bridge-VLA training loop.

## Active Path

```text
cache VLM tokens and robot state
build one 32-step action-intent target per sample
train the action-only intent autoencoder
freeze AE
train CoarsePlanner to predict one AE latent from one plan token
evaluate latent MSE, decoded chunk quality, and cosine similarity
use the single plan token at inference/integration time
```

The old compressed coarse-action target and H64 multi-token suffix queue path are retired.

## Current Contract

```text
planning_horizon: 32
num_plan_steps: 1
chunk_size: 32
action_segments: [B, 1, 32, 7]
action_segment_mask: [B, 1]
planner output: [B, 1, D]
```

## Active Artifacts

```text
../datasets/coarse_planner/libero_h32_single_token_s32768_seed42
../datasets/coarse_planner/libero_h32_single_token_s32768_seed42_action_only
../runs/coarse_planner/libero_h32_intent_ae_v1
../runs/coarse_planner/libero_h32_single_token_planner_v1
```

## Active Configs

```text
coarse_planner/configs/libero_h32_single_token_build.yaml
coarse_planner/configs/libero_h32_intent_ae_v1.yaml
coarse_planner/configs/libero_h32_single_token_planner_v1.yaml
```

## Common Commands

```bash
python -m coarse_planner.build_from_libero --config coarse_planner/configs/libero_h32_single_token_build.yaml --device cuda
python -m coarse_planner.extract_fields_cache --source ../datasets/coarse_planner/libero_h32_single_token_s32768_seed42 --output ../datasets/coarse_planner/libero_h32_single_token_s32768_seed42_action_only --fields action_segments action_segment_mask
python -m coarse_planner.train_segment_autoencoder --config coarse_planner/configs/libero_h32_intent_ae_v1.yaml --device cuda
python -m coarse_planner.train --config coarse_planner/configs/libero_h32_single_token_planner_v1.yaml --device cuda
```

## Current Metrics

```text
H32 intent AE: best epoch 99, val_loss 0.0137875769
H32 planner:   best epoch 52, raw MSE 0.0869378231, val loss 0.2716650516, cosine 0.9047680597
```

## Reuse In Current Design

The trained H32 intent autoencoder can be reused as the frozen label encoder for progress-state planner warmup:

```text
z_k = IntentEncoder(normalized_A_k) # [B, 128], frozen target
P_k = Planner(M_k, h_k, s_k)        # [B, 1, D]
```
