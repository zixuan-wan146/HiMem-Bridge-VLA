# Coarse Planner Design

Date: 2026-06-22

Status: H32 single-token intent planning.

## Core Decision

The active planner no longer predicts a cached multi-token suffix. It predicts
one plan token at every inference step:

```text
P_t = CoarsePlanner(H_t_vlm, s_t)
P_t: [B, 1, D]
```

The single token represents the intent for the next 32 low-level action steps.
Every inference recomputes it from the current observation.

This removes the previous failure mode where a long cached plan could remain in
force after the world state had drifted. The closed-loop mechanism is now
frequent replanning, not transition-triggered hard refresh.

## Removed Semantics

The active H32 path does not use:

```text
PlanTokenQueue
cached plan suffixes
consumed-token offsets
execution-horizon refresh policy
transition-trigger refresh decisions
H64 multi-token targets
```

Transition trigger is deleted from the active server/eval/test path. It should
not be retrained for this design unless the planning horizon is lengthened again
and a concrete closed-loop failure case requires it.

## Action Intent Target

The planner target is still defined by an action-only autoencoder. For each
sample:

```text
A_t: [32, action_dim]
z_t* = E_phi(A_t)
```

The autoencoder is action-only. It must not use VLM tokens, language tokens,
state, or memory. Its role is to define a latent space for 32-step action
chunks.

The active H32 target contract is:

```text
planning_horizon: 32
num_plan_steps: 1
chunk_size: 32
action_segments: [B, 1, 32, action_dim]
action_segment_mask: [B, 1]
```

## Planner Objective

The planner receives current VLM tokens and robot state:

```text
P_t = f_plan(H_t_vlm, s_t)
z_hat_t = W_z P_t
```

Training uses the frozen autoencoder for supervision:

```text
L_z = ||z_hat_t - z_t*||_2^2
A_hat_t = D_theta(z_hat_t)
L_chunk = rec(A_hat_t, A_t)
L_planner = lambda_z * L_z + lambda_A * L_chunk
```

The latent head and frozen decoder are training-time shaping tools. Inference
uses only the plan token.

## Inference Contract

The intended integration contract is:

```text
P_t = f_plan(H_t_vlm, s_t)       # [B, 1, D]
H_t_bridge = BridgeAttention(H_t_vlm, s_t, P_t, M_t)
ActionHead(H_t_bridge, s_t)      # predicts a 32-step action chunk
```

There is no plan cache key and no requested/consumed control-step bookkeeping in
the H32 planner path.

## Data And Training Notes

The full feature cache stores VLM tokens, state, and action targets. The AE
training path uses a derived action-only cache to avoid loading VLM features:

```text
libero_h32_single_token_s32768_seed42
libero_h32_single_token_s32768_seed42_action_only
```

Planner training uses shard-local batching:

```text
training.shuffle_mode: shard
data.shard_cache_size: 32
```

This preserves useful randomization without requiring the whole feature cache to
live in CPU memory.

## Active Configs

```text
coarse_planner/configs/libero_h32_single_token_build.yaml
coarse_planner/configs/libero_h32_intent_ae_v1.yaml
coarse_planner/configs/libero_h32_single_token_planner_v1.yaml
```

The current best metric for planner checkpoint selection is:

```text
val_raw_latent_mse
```

Training should stop when the explicit wall-clock deadline expires. The final
checkpoint choice should still use the best validation checkpoint, not the last
epoch by default.
