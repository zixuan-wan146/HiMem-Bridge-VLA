# Coarse Planner LIBERO Horizon Ablation

Date: 2026-06-18

Status: historical baseline only. The target used here has been rejected and is
being removed from the active implementation.

## What This Experiment Tested

The old warm-up experiment trained Coarse Planner with a manually compressed
coarse-action target:

```text
motion_i = sum(actions[t+i*c : t+(i+1)*c, motion_dims])
gripper_i = actions[t+(i+1)*c-1, gripper_dim]
```

For LIBERO, the sweep used `chunk_size=8`:

```text
H=32, K=4
H=48, K=6
H=64, K=8
```

The build used up to 2048 samples with a shared sampled index set across horizons.

## Recorded Results

```text
H=32: val loss 0.725982, val MAE 1.121518, peak reserved 22.506 GB
H=48: val loss 0.767584, val MAE 1.169856
H=64: val loss 0.768933, val MAE 1.176484
```

These results are useful only as a debugging record for the old target. They are
not a validation of the current intent-latent planner design.

## Why This Target Was Removed

The compressed target collapses distinct action segments into the same label:

```text
A_i != A_j but sum(A_i) == sum(A_j)
```

That makes the planner learn `P_t -> sum(action_chunk)` instead of a latent that
represents the full coarse intent of the segment.

## Replacement Direction

The active design replaces compressed targets with action-segment latents:

```text
z_i* = E_phi(A_i)
z_hat_i = W_z P_i
L_planner = lambda_z * ||z_hat_i - z_i*||^2 + lambda_A * rec(D_theta(z_hat_i), A_i)
```

Training and inference also use plan-token suffix semantics:

```text
P_tau = f_plan(H_tau, s_tau)
t = tau + u*c
P_active = P_tau[u:K]
ActionHead(H_t, s_t, P_active, M_t)
```

No further training should be launched from the old ablation scripts until they
are converted to the action-segment latent target.
