# Bridge-HiMem Design

This document summarizes the active model path after the memory/planner redesign. The checked-in code still contains the legacy H32 `CoarsePlanner` route as a baseline, but the active architecture is:

```text
progress-state planner
+ short visual-token memory
+ direct bridge-attn flow-matching action head
```

Older H64 suffix queues, transition-trigger refresh logic, and Dual-FIFO long visual-memory logic are not part of this design.

## Target Model Path

```text
RGB views + prompt
  -> InternVL3
  -> progress VL summary h_t
  -> VLM hidden-state token layers [3, 6, 9, 12]
  -> ShortVisualMemory -> S_t
  -> ProgressEvidenceEncoder(h_t, s_t, u_t) -> x_t
  -> ProgressStateUpdater(M_{t-1}, x_t) -> M_t
  -> ProgressPlanner(M_t, h_t, s_t) -> P_t
  -> expand P_t to 8 plan slots
  -> DirectBridgeActionHead
       action self-attn over 32 noisy action tokens
       visual cross-attn over [VLM hidden states, short memory]
       action-condition cross-attn over [plan slots, state token]
       flow-matching Euler inference with 15 steps
```

## Config Entry

Bridge-HiMem experiment files live under:

```text
configs/bridge_himem/base.yaml
configs/bridge_himem/experiments/*.yaml
```

New experiment knobs should go through YAML and `himem_bridge_vla/bridge_himem_config.py`. Do not hard-code experiment behavior in model or training scripts.

Validate configs before training:

```bash
python scripts/validate_bridge_himem_configs.py
```

## VL Inputs

Progress-state planner uses one summary vector:

```text
h_t: [B, 896]
```

Warm-up caches store `h_t` directly as `vl_summary`. Runtime policy code accepts an explicit `planner_vl_summary` and uses it when available.

Direct bridge-attn uses token hidden states, not a pooled vector:

```text
bridge raw layers: [3, 6, 9, 12]
action block schedule: [3, 3, 6, 6, 9, 9, 12, 12]
```

## Short Visual Memory

Short memory is independent visual-token context:

```text
H = 32
R = 16
S_t = ShortVisualMemory(V_{t-R/2}, V_{t-R})
```

Each recent visual observation is compressed with a BottleneckSE-style visual compressor, not a learnable-query compression head. This keeps the short-memory path close to MemoryVLA's perceptual compression and reduces the risk that free queries collapse into fixed templates under weak downstream supervision.

It is responsible for local continuity: recent pose, contact, occlusion, and motion evidence. It should not maintain task progress.

## Long Memory And Planner

Long memory is the planner's task-progress state:

```text
M_t = [C_t, G_t]
```

Where:

```text
C_t: completed-events state token
G_t: current-stage state token
```

The planner updates this state recurrently:

```text
x_t = ProgressEvidenceEncoder(h_t, s_t, u_t)
M_t = ProgressStateUpdater(M_{t-1}, x_t)
P_t = ProgressPlanner(M_t, h_t, s_t)
```

The long memory is not a FIFO visual bank and does not grow with time.

## Direct Bridge-Attn Action Head

The active action head does not generate intermediate bridge tokens. It uses the 32 noisy action tokens from the flow-matching horizon as the query sequence.

```text
visual branch:
  current VLM hidden-state tokens
  short memory tokens

action-condition branch:
  8 plan slots expanded from one planner token
  1 state token
```

Each action block contains action self-attn, visual cross-attn, action-condition cross-attn, and FFN. See `docs/direct_bridge_attention_design_zh.md`.

## Legacy H32 Planner Token

The old H32 path:

```text
P_t = CoarsePlanner(fused_tokens, state)
```

is now a baseline / auxiliary action-intent route. Its trained artifacts can be used for comparisons or as optional intent targets, but the action-latent supervision no longer defines the main planner semantics.

Legacy coarse-planner defaults:

```yaml
coarse_planner:
  num_plan_steps: 1
  planning_horizon: 32
  input_memory: false
```

## Legacy Experiment Files

```text
baseline.yaml                  fused-token control
crosskv_clean.yaml             cross-attention bridge baseline
mixed_latent_clean.yaml        mixed-latent bridge baseline
mixed_latent_skill.yaml        skill-token ablation
coarse_planner_crosskv.yaml    legacy H32 action-latent planner config
```

New progress-state planner experiments should be added under new names rather than changing the meaning of `coarse_planner_crosskv.yaml`.
