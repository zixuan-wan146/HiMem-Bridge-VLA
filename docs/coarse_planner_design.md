# Coarse Planner Design

Date: 2026-06-18

Status: refactoring from compressed coarse-action supervision to action-segment intent
latents. Do not start training until the action-segment autoencoder architecture and
training recipe are explicitly approved.

Concrete first-version module parameters are tracked in:

```text
docs/action_segment_autoencoder_coarse_planner_config.md
```

## Core Decision

The Coarse Planner still predicts coarse plan tokens:

```text
P_tau = CoarsePlanner(H_tau_vlm, s_tau)
P_tau: [B, K, D]
```

Those tokens are the only planner signal used by the inference backbone:

```text
H_t_bridge = BridgeAttention(H_t_vlm, s_t, P_active, M_t)
ActionHead(H_t_bridge, s_t)
```

The old compressed coarse-action target is removed. The planner is no longer trained
to predict hand-built chunk summaries such as motion sums or terminal gripper values.
That target collapses distinct action segments into identical labels.

The new target is an action-segment intent latent:

```text
z_i* = E_phi(A_i)
z_hat_i = W_z p_i
```

`E_phi`, `D_theta`, and `W_z` are training-time shaping tools. Inference uses only
`P_active`.

## Action-Segment Autoencoder

Future actions are split into `K` full segments:

```text
A_tau: [H_p, action_dim]
A_tau -> [A_0, ..., A_{K-1}]
A_i: [c, action_dim]
c = H_p / K
```

The action-segment autoencoder is action-only:

```text
z_i* = E_phi(A_i)
A_hat_i = D_theta(z_i*)
```

It must not use `s_t`, VLM tokens, memory tokens, language tokens, or any other state
condition. Its job is to define the latent geometry of action segments themselves.

The intended training objective is:

```text
L_AE = L_rec + lambda_dist * L_dist
```

`L_rec` reconstructs the full segment, with a motion reconstruction term and a
separate gripper term. `L_dist` keeps latent distances aligned with action-segment
distances, for example from low-frequency trajectory shape, endpoint displacement,
and gripper change.

After AE training:

```text
freeze(E_phi, D_theta)
```

Main-model training then uses the frozen encoder and decoder for supervision only.

## Planner Intent Loss

The planner still receives current VLM tokens and proprioception at an anchor time
`tau`:

```text
P_tau = f_plan(H_tau_vlm, s_tau)
z_hat_tau_i = W_z P_tau_i
z_tau_i* = E_phi(A_tau_i)
```

The auxiliary planner objective is:

```text
L_z = sum_i mask_i * ||z_hat_tau_i - z_tau_i*||_2^2
A_hat_tau_i = D_theta(z_hat_tau_i)
L_chunk = sum_i mask_i * rec(A_hat_tau_i, A_tau_i)
L_planner = lambda_z * L_z + lambda_A * L_chunk
```

The full training loss is:

```text
L = L_flow + L_bridge_aux + lambda_cp * L_planner
```

The latent head and frozen decoder are not part of the inference path.

## Plan Token Queue

Inference maintains a per-session plan cache:

```text
C = (P_tau, N)
```

`N` is the cumulative number of low-level control steps executed from this cached
plan. Token consumption is derived from `N`, not from call count:

```text
u = floor(N / c)
r = N mod c
P_active = P_tau[u:K]
```

This avoids the partial-execution bug:

```text
4 // 8 + 4 // 8 = 0
floor((4 + 4) / 8) = 1
```

The recommended default for the first implementation is:

```text
H_p = 64
K = 8
c = 8
H_e = 16
```

A refresh is required when:

```text
cache is empty
episode reset happened
transition trigger requested refresh
N + requested_execute_steps > H_p
```

Otherwise BridgeAttention receives the remaining suffix `P_tau[u:K]`.

## Strict Training State

Training must simulate the cached inference state. For a sample, choose an anchor
time `tau` and a consumed-token offset:

```text
u in {0, q, 2q, ..., K-q}
t = tau + u * c
```

The planner input comes from the anchor time:

```text
P_tau = f_plan(H_tau_vlm, s_tau)
P_active = P_tau[u:K]
```

The action prediction input comes from the current time:

```text
ActionHead(H_t_vlm, s_t, P_active, M_t)
```

Using `P_t = f_plan(H_t, s_t)` and then cropping only trains variable-length
suffixes. It does not train the actual cached-plan state where the plan is stale
relative to the current observation.

## Data Contract

Main training samples for planner-enabled runs should contain:

```text
current image/state/action fields:
  images
  image_mask
  prompt
  state
  action
  action_mask

planner anchor fields:
  planner_images
  planner_image_mask
  planner_prompt
  planner_state

action-segment supervision from the anchor:
  action_segments: [K, c, action_dim]
  action_segment_mask: [K]

queue/suffix metadata:
  plan_consumed_steps = N
  plan_consumed_tokens = floor(N / c)
  plan_residual_steps = N mod c
  plan_active_mask: [K]
```

`plan_active_mask` is false for tokens before `u` and true for the active suffix.
BridgeAttention receives padded plan tokens plus a key-padding mask, so the action
head learns to operate with suffix lengths `K, K-q, ..., q`.

## Module Boundaries

```text
ActionSegmentAutoencoder   action-only definition of segment intent latent
CoarsePlanner              predicts plan tokens and training-time latents
PlanTokenQueue             per-session cached suffix consumption by executed steps
BridgeAttention            fuses current VLM/state, active plan suffix, and memory
ActionHead                 predicts executable low-level action chunk
TransitionTrigger          asks for cache refresh and memory writes
```

The planner does not read memory in this version. Memory and plan tokens remain
parallel BridgeAttention conditions.

## Implementation Notes

- Remove `coarse_actions`, `coarse_action_mask`, and `build_coarse_action_target`.
- Keep network dimensions and placeholder AE parameters in config for now.
- Build the new dataset/cache format before running any training.
- Do not keep compatibility shims for old compressed targets.
- Old H32/H48/H64 LIBERO results are retained only as a baseline record of the
  discarded target, not as evidence for the new intent-latent design.
