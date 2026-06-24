# Current Project State

Date: 2026-06-23

## Active Direction

The active path is H32 single-token open-loop planning:

```text
P_t = CoarsePlanner(H_t, s_t)   # [B, 1, D]
ActionHead(..., P_t)            # predicts a 32-step action chunk
```

Every inference recomputes one plan token from the current observation and state. The token is an intent summary for the next 32 low-level action steps. There is no plan-token queue, no consumed-step suffix state, and no transition-trigger refresh path.

Memory is now a separate Dual-FIFO visual-memory workstream. The current implementation focus is memory-side inference construction only: entry schema, deterministic short reads, external long FIFO writes, padding masks, and view-aware query compression. BridgeAttention memory integration remains out of scope for this step.

## Removed From Active Path

```text
H64 multi-token planner caches and checkpoints
PlanTokenQueue / suffix consumption
transition trigger model, server wiring, eval trace code, and tests
transition-trigger dataset conversion scripts
LIBERO transition-frame trace logging
```

Old design notes were removed from checked-in docs to avoid treating them as live roadmap items.

## Remote Locations

```text
repo:      $AUTODL_TMP/HiMem-Bridge-VLA
data root: $AUTODL_TMP
```

Large datasets, caches, checkpoints, and runs stay on the remote data disk. Local sync is for code and documentation only.

## Active H32 Artifacts

```text
$AUTODL_TMP/datasets/coarse_planner/libero_h32_single_token_s32768_seed42
  samples: 32768
  train/eval split from training history: 29480 / 3288
  target: planning_horizon=32, num_plan_steps=1, chunk_size=32
  sample action_segments shape: [1, 32, 7]

$AUTODL_TMP/datasets/coarse_planner/libero_h32_single_token_s32768_seed42_action_only
$AUTODL_TMP/runs/coarse_planner/libero_h32_intent_ae_v1
$AUTODL_TMP/runs/coarse_planner/libero_h32_single_token_planner_v1
```

## Active Configs

```text
coarse_planner/configs/libero_h32_single_token_build.yaml
coarse_planner/configs/libero_h32_intent_ae_v1.yaml
coarse_planner/configs/libero_h32_single_token_planner_v1.yaml
configs/bridge_himem/base.yaml
configs/bridge_himem/experiments/coarse_planner_crosskv.yaml
```

Important settings:

```text
ActionHead horizon: 32
CoarsePlanner num_plan_steps: 1
CoarsePlanner planning_horizon: 32
planner training shuffle_mode: shard
planner shard_cache_size: 32
planner best metric: val_raw_latent_mse
```

## Completed

```text
deleted old H64 active datasets/checkpoints/configs from remote data disk
deleted transition_trigger package and transition-trigger manager
removed transition-trigger server protocol and LIBERO client/eval trace plumbing
removed PlanTokenQueue and cached suffix state from model inference
changed CoarsePlanner defaults to one token over a 32-step horizon
changed ActionHead default horizon to 32
rebuilt LIBERO H32 single-token planner feature cache
derived action-only cache for AE training
trained H32 action intent AE
trained H32 single-token planner until the wall-clock deadline
```

## Current Metrics

```text
H32 Action Intent AE:
  best checkpoint: epoch 99
  val_loss:       0.0137875769
  val_rec_loss:   0.0123513332
  val_dist_loss:  0.0143624358

H32 Single-Token Planner:
  stopped at the 2026-06-22 04:45 wall-clock deadline
  best epoch:               52
  val_raw_latent_mse:       0.0869378231
  val_normalized_latent_mse:0.2122704722
  val_loss:                 0.2716650516
  decoded_chunk_loss:       0.2625050992
  latent_cosine:            0.9047680597
  last epoch:               60
  last raw latent MSE:      0.0878419157
```

The planner plateaued around 0.09 raw latent MSE. Use the current best checkpoint for H32 integration unless a new model/data strategy is chosen.

## Next Work

```text
1. integrate one H32 plan token into BridgeAttention / ActionHead training
2. keep planner token shape fixed at [B, 1, D]
3. evaluate against fused-only and bridge-clean baselines
4. keep BridgeAttention memory integration separate until the memory-side inference path is tested
```
