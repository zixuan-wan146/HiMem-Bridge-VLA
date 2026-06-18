# Coarse Planner Design

Date: 2026-06-18

Status: standalone warm-up first, then main-model integration.

## Current Decision

The first Coarse Planner does not read memory.

```text
P_t = CoarsePlanner(H_t_vlm, s_t)
M_t = MemoryRead(q_t)
H_t_bridge = BridgeAttention(H_t_vlm, s_t, P_t, M_t)
```

`P_t` and `M_t` are parallel BridgeAttention conditions. Memory-conditioned planning is left for a later ablation.

The updated development order is:

```text
1. Train CoarsePlanner as a standalone module.
2. Diagnose its coarse future-action predictions without the action head.
3. Export a planner checkpoint.
4. Load it into HiMemBridgeVLA and finetune Bridge/ActionHead.
```

This makes planner quality observable before it is coupled to memory, BridgeAttention, FlowMatching, or closed-loop runtime.

## Module Boundaries

```text
Coarse Planner     current-observation coarse motor intent
MemoryBank         historical events and completed stage state
BridgeAttention    fusion of action queries, state, plan tokens, and memory tokens
Action Head        short-horizon executable action generation
Transition Trigger runtime timing for plan refresh and memory write
```

This avoids making the small planner adapter interpret memory, translate memory into action-space intent, and predict future motion at the same time.

## Planner Input And Output

Input:

```text
vlm_tokens: [B, L, D]
state:      [B, state_dim]
```

Output:

```text
plan_tokens:    [B, K, D]
coarse_actions: [B, K, action_dim]
```

The implementation lives in:

```text
himem_bridge_vla/model/planner/coarse_planner.py
```

The first version uses learnable plan queries and Transformer decoder blocks:

```text
context = concat(vlm_tokens, state_token)
plan_tokens = TransformerDecoder(plan_queries, context)
coarse_actions = Linear(plan_tokens)
```

The planner must use at least three Transformer layers.

## Standalone Training Module

The standalone training wrapper lives in:

```text
coarse_planner/
  config.py
  data.py
  build_dataset.py
  train.py
  evaluate.py
  export.py
```

The important boundary is:

```text
coarse_planner/                     training, feature cache, checkpoint export
himem_bridge_vla/model/planner/     actual planner model used by both standalone and main model
```

There must not be two planner implementations. Standalone training imports the same `CoarsePlanner` class that `HiMemBridgeVLA` uses later.

The standalone route consumes cached features rather than running the full VLA stack:

```text
vlm_tokens: [T, L, D]
states:     [T, state_dim]
actions:    [T, action_dim]
```

and writes sharded planner samples:

```text
manifest.json
train/planner_samples_00000.pt
eval/planner_samples_00001.pt
```

Each sample contains:

```text
vlm_tokens
state
coarse_actions
coarse_action_mask
episode_id
frame_index
```

This dataset is dedicated to planner warm-up. It is not the same as the full action-head training dataset.

The default VLM token source is:

```text
feature.source = fused
```

This matches the main training path:

```text
InternVL3Embedder(..., return_cls_only=False) -> embedding_output.fused_tokens
```

For layer ablations, use:

```text
feature.source = hidden_state
feature.hidden_state_layer = shallow | mid | deep | last | integer index
```

State is not pre-concatenated into the cached VLM tensor. The cache stores state separately, and `CoarsePlanner` projects it into a state token:

```text
context = [vlm_tokens, state_proj(state)]
```

That keeps the cache model-agnostic while still training the exact VLM/state fusion used by the planner module.

## Coarse Action Target

For each training sample, future demonstration actions over `planning_horizon` are compressed into `num_plan_steps` chunks.

```text
coarse_actions:      [K, action_dim]
coarse_action_mask:  [K]
```

For relative end-effector actions:

```text
motion target  = sum(chunk motion deltas)
gripper target = last chunk gripper value
```

For absolute actions, use either endpoint delta or endpoint terminal value, selected by `action_convention`:

```text
relative
absolute_delta
absolute_terminal
```

Target construction lives in:

```text
himem_bridge_vla/dataset/coarse_actions.py
```

`SimulationDataset` writes the targets into cached samples when `coarse_planner_enabled=true`. The cache namespace includes the coarse target config so old action-only caches are not reused.

Standalone planner cache construction uses the same target function:

```text
coarse_planner.data.build_planner_feature_cache
  -> himem_bridge_vla.dataset.coarse_actions.build_coarse_action_target
```

So the standalone target and joint-training target stay identical.

## Bridge Injection

`BridgeAttentionBlock` condition tokens are now:

```text
[action_queries, proprio_embedding, plan_tokens, memory_context]
```

The injection path is:

```text
HiMemBridgeVLA._augment_context_with_bridge()
  -> CoarsePlanner(...)
  -> BridgeAdapter(..., plan_tokens=P_t)
  -> BridgeAttentionBlock(..., plan_tokens=P_t, memory_context=M_t)
```

The planner still receives no memory input.

## Training Objective

The action head keeps the existing FlowMatching loss:

```text
L_flow
```

The planner adds a masked Smooth L1 loss:

```text
L_cp = SmoothL1(predicted_coarse_actions, coarse_actions, coarse_action_mask)
```

The training loss is:

```text
L = L_flow + lambda_cp * L_cp + existing_bridge_aux_losses
```

The current default is:

```yaml
coarse_planner:
  loss_weight: 0.2
  gripper_loss_weight: 2.0
  smoothness_weight: 0.01
  max_age_steps: 16
```

Loss implementation:

```text
himem_bridge_vla/training_loss.py::coarse_planner_smooth_l1_loss
scripts/train.py::compute_coarse_planner_loss
```

For standalone warm-up, there is no `L_flow`:

```text
L = L_cp
```

The standalone output checkpoint should be evaluated by:

```text
planner loss decreases
coarse trajectory visualization looks plausible
gripper timing is not washed out by motion dimensions
```

Only after those checks should the checkpoint be loaded into the main VLA model.

## Configuration

Defaults are in:

```text
configs/bridge_himem/base.yaml
```

Runnable first experiment config:

```text
configs/bridge_himem/experiments/coarse_planner_crosskv.yaml
configs/bridge_himem/experiments/coarse_planner_plan_only.yaml
```

Important invariants enforced by config validation:

```text
coarse_planner.enabled=true requires bridge.enabled=true
coarse_planner.input_memory must be false
coarse_planner.num_layers >= 3
coarse_planner.hidden_dim == vlm.hidden_dim
coarse_planner.planning_horizon % coarse_planner.num_plan_steps == 0
```

## Runtime Refresh

During training, the planner predicts a fresh plan for every sample so the supervised planner loss is always available.

During inference, `refresh_policy=transition_or_expire` uses a per episode/session cache:

```text
refresh if no cached plan
refresh if transition_trigger.should_plan is true
refresh if cached plan age >= coarse_planner.max_age_steps
otherwise reuse cached plan_tokens
```

The server passes `transition_trigger.should_plan` into `HiMemBridgeVLA.run_inference(...)` as the plan refresh signal. Memory write remains separate and still uses the existing memory gate path.

## First Experiment Scope

Use these controlled comparisons first:

```text
baseline
memory only
plan only
plan + memory parallel
```

The current config mapping is:

```text
baseline:               baseline.yaml
memory only:            crosskv_clean.yaml
plan only:              coarse_planner_plan_only.yaml
plan + memory parallel: coarse_planner_crosskv.yaml
```

Do not claim LIBERO performance improvement from this module until plan masking, memory masking, and closed-loop evaluation are run.

## Standalone Commands

Pipeline smoke test:

```bash
python -m coarse_planner.build_dataset \
  --config coarse_planner/configs/synthetic_smoke.yaml \
  --synthetic-smoke \
  --output /root/autodl-tmp/datasets/coarse_planner/smoke
```

Planner warm-up:

```bash
python -m coarse_planner.train \
  --config coarse_planner/configs/default.yaml \
  --run-dir /root/autodl-tmp/runs/coarse_planner/libero_warmup
```

Checkpoint export:

```bash
python -m coarse_planner.export \
  --checkpoint /root/autodl-tmp/runs/coarse_planner/libero_warmup/best.pt \
  --output /root/autodl-tmp/checkpoints/coarse_planner/libero_warmup.pt
```

The real dataset-building step requires precomputed VLM token sources. A valid source is a `.pt` or `.npz` file containing episode records with:

```text
episode_id
vlm_tokens [T, L, D]
states     [T, state_dim]
actions    [T, action_dim]
```

To build those token sources directly from a `SimulationDataset`, use:

```bash
python -m coarse_planner.build_from_simulation \
  --config coarse_planner/configs/calvin_abc_d_smoke.yaml \
  --dry-run
```

and then:

```bash
python -m coarse_planner.build_from_simulation \
  --config coarse_planner/configs/calvin_abc_d_smoke.yaml \
  --device cuda \
  --max-samples 128
```

Verified smoke path:

```text
source subset: /root/autodl-tmp/datasets/calvin/lerobot/task_ABC_D_smoke
feature cache: /root/autodl-tmp/datasets/coarse_planner/calvin_abc_d_smoke
token source: fused
state source: normalized/padded SimulationDataset state
```

## Required Diagnostics

Before making a performance claim:

```text
planner loss decreases
coarse action visualization points toward plausible task progress
zeroing plan_tokens hurts plan-sensitive tasks
zeroing memory_context hurts memory-dependent tasks
BridgeAttention attends to both plan and memory condition slots
```
