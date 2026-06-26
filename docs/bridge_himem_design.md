# Bridge-HiMem Design

This document describes the active progress-state planner warmup surface after the memory/planner redesign. The checked-in code still contains the legacy H32 `CoarsePlanner` path; the new progress-state planner is the target design and should be implemented under new config names.

Older H64 suffix queues, transition-trigger refresh logic, and Dual-FIFO long visual-memory logic are not part of this design.

Current design work is scoped to progress-state planner warmup.

## Target Model Path

```text
RGB views + prompt
  -> InternVL3 final hidden states F_t
  -> AttnPool(F_t) -> h_t
  -> ShortVisualMemory -> S_t
  -> ProgressEvidenceEncoder(h_t, s_t, u_t) -> x_t
  -> ProgressStateUpdater(M_{t-1}, x_t) -> M_t
  -> ProgressPlanner(M_t, h_t, s_t) -> P_t
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

## VL Embeddings

Use the final InternVL3 hidden sequence as the first implementation choice:

```text
F_t = final InternVL3 fused hidden states
```

The final layer is the most task- and prompt-aligned representation available in the current stack. Keep the token sequence for evidence pooling, but the planner receives the pooled summary `h_t` rather than cross-attending over the full `F_t` sequence.

If later diagnostics show that the final layer loses low-level object detail, add a small layer-mixing projection as a second-stage refinement. Do not add that complexity before the final-layer route has a measured failure case.

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
