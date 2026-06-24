# Coarse Planner Design

Date: 2026-06-22

Status: active H32 single-token intent planner.

## Core Decision

The planner predicts one plan token at every inference step:

```text
P_t = CoarsePlanner(H_t, s_t)
P_t: [B, 1, D]
```

The token represents the intent for the next 32 low-level action steps. Every inference recomputes it from the current observation and robot state. Closed-loop correction comes from frequent replanning, not from a transition-triggered refresh of a cached long plan.

## Removed Semantics

```text
PlanTokenQueue
cached plan suffixes
consumed-token offsets
execution-horizon refresh policy
transition-trigger refresh decisions
H64 multi-token targets
```

Do not add these concepts back unless the project explicitly returns to a longer cached planning horizon.

## Action Intent Target

The planner target is defined by an action-only autoencoder:

```text
A_t: [32, action_dim]
z_t* = E_phi(A_t)
```

The autoencoder must not use VLM tokens, language tokens, state, or memory. Its role is only to define a compact latent space for 32-step action chunks.

The active H32 target contract is:

```text
planning_horizon: 32
num_plan_steps: 1
chunk_size: 32
action_segments: [B, 1, 32, action_dim]
action_segment_mask: [B, 1]
```

## Planner Objective

```text
P_t = f_plan(H_t, s_t)
z_hat_t = W_z P_t
L_z = ||z_hat_t - z_t*||_2^2
A_hat_t = D_theta(z_hat_t)
L_chunk = rec(A_hat_t, A_t)
L_planner = lambda_z * L_z + lambda_A * L_chunk
```

The latent head and frozen decoder are training-time shaping tools. Inference uses only the plan token.

## Inference Contract

```text
P_t = f_plan(H_t_vlm, s_t)       # [B, 1, D]
H_t_bridge = BridgeAttention(H_t_vlm, s_t, P_t, M_t)
ActionHead(H_t_bridge, s_t)      # predicts a 32-step action chunk
```

There is no plan cache key and no requested/consumed control-step bookkeeping in this path.

## Data And Training

The full feature cache stores VLM tokens, state, and action targets. The AE training path uses a derived action-only cache to avoid loading VLM features:

```text
libero_h32_single_token_s32768_seed42
libero_h32_single_token_s32768_seed42_action_only
```

Planner training uses shard-local batching:

```text
training.shuffle_mode: shard
data.shard_cache_size: 32
```

## Active Configs

```text
coarse_planner/configs/libero_h32_single_token_build.yaml
coarse_planner/configs/libero_h32_intent_ae_v1.yaml
coarse_planner/configs/libero_h32_single_token_planner_v1.yaml
```

Checkpoint selection uses `val_raw_latent_mse`. The current best checkpoint is epoch 52 from `$AUTODL_TMP/runs/coarse_planner/libero_h32_single_token_planner_v1/best.pt`.

## Integration Rules

- The integrated model should consume exactly one plan token.
- The ActionHead horizon should remain 32 for this path.
- The planner should not read memory in this version.
- BridgeAttention / ActionHead training should decide whether the token is useful before any memory rewrite is added.
