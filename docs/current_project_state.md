# Current Project State

Date: 2026-06-22

## Active Direction

The project has switched from H64 multi-token cached suffix planning to H32
single-token open-loop planning.

The active inference contract is:

```text
P_t = CoarsePlanner(H_t, s_t)   # [B, 1, D]
ActionHead(..., P_t)            # predicts a 32-step action chunk
```

One planner call produces exactly one plan token. That token represents the
future 32 low-level action steps. There is no plan-token queue, no consumed-step
suffix state, and no transition-trigger refresh path.

Memory work is intentionally out of scope for this round. Memory is not being
redesigned here.

## Removed From Active Path

The following old path has been removed or deprecated from active use:

```text
H64 multi-token planner caches and checkpoints
PlanTokenQueue / suffix consumption
transition trigger model, server wiring, eval trace code, and tests
old transition-trigger dataset conversion scripts
```

The old rationale for transition-trigger refresh no longer applies to H32
single-token planning. Since every inference recomputes one plan token from the
current observation, closed-loop correction comes from frequent replanning rather
than a separate trigger.

## Remote Locations

Remote repo:

```text
/root/autodl-tmp/HiMem-Bridge-VLA
```

Remote data and run root:

```text
/root/autodl-tmp
```

Large datasets, caches, and checkpoints stay on the remote data disk. Local sync
is for code and docs only.

## Active H32 Artifacts

Feature cache:

```text
/root/autodl-tmp/datasets/coarse_planner/libero_h32_single_token_s32768_seed42
  32768 samples
  train/eval split: 29480 / 3288
  target: planning_horizon=32, num_plan_steps=1, chunk_size=32
  sample action_segments shape: [1, 32, 7]
```

Action-only AE cache:

```text
/root/autodl-tmp/datasets/coarse_planner/libero_h32_single_token_s32768_seed42_action_only
  derived cache for AE training
  retains action_segments and action_segment_mask only
```

AE run:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h32_intent_ae_v1
```

Planner run:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h32_single_token_planner_v1
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
planner early stopping patience: 12
```

`shuffle_mode: shard` keeps training batches local to feature-cache shards. This
avoids requiring a huge shard cache while still shuffling shard order and samples
within each shard.

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
started H32 single-token planner training
```

## Current Metrics

H32 Action Intent AE:

```text
best checkpoint: epoch 99
val_loss:       0.0137875769
val_rec_loss:   0.0123513332
val_dist_loss:  0.0143624358
peak reserved:  about 23.0 GiB
```

H32 Single-Token Planner:

```text
training status: stopped at the 2026-06-22 04:45 wall-clock deadline
checkpoint selection metric: val_raw_latent_mse
best checkpoint:
  epoch:                     52
  val_raw_latent_mse:        0.0869378231
  val_normalized_latent_mse: 0.2122704722
  val_loss:                  0.2716650516
  decoded_chunk_loss:        0.2625050992
  latent_cosine:             0.9047680597
last checkpoint:
  epoch:                     60
  val_raw_latent_mse:        0.0878419157
  val_normalized_latent_mse: 0.2148081090
  val_loss:                  0.2770999521
  decoded_chunk_loss:        0.2702323571
  latent_cosine:             0.9047670666
```

Planner did not reach the desired 0.05-ish raw latent MSE range in this time
box. It plateaued around 0.09 raw latent MSE, while cosine similarity reached
about 0.90.

Final planner metrics were read from:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h32_single_token_planner_v1/best.pt
/root/autodl-tmp/runs/coarse_planner/libero_h32_single_token_planner_v1/last.pt
```

## Resource Constraints

The server memory budget for training is treated as 120 GB. Planner training was
adjusted to stay well below that budget:

```text
planner shard_cache_size: 32
observed CPU memory during planner training: roughly 45-55 GiB used
observed GPU memory during planner training: roughly 23-24 GiB used/reserved
```

AE training uses an action-only cache so it does not repeatedly load VLM feature
tokens.

## Next Engineering Work

Do not restart transition trigger work for this H32 path.

Next useful work after planner training finishes:

```text
1. inspect final H32 single-token planner metrics
2. decide whether 0.05-ish raw latent MSE is sufficient for integration
3. update BridgeAttention / ActionHead training to consume one H32 plan token
4. keep memory rewrite separate from this planner refactor
```

BridgeAttention / ActionHead joint training is intentionally deferred until this
planner checkpoint is available and the memory direction is decided.
