# HiMem-Bridge-VLA Current Plan

This checkout is the active project for HiMem VLA progress-state planner work. The active method is:

```text
short memory = independent recent visual-token memory
long memory  = recurrent task-progress state inside the planner
```

Older transition-trigger, H64 suffix-planner, and Dual-FIFO long visual-memory designs are retired from the active roadmap.

## Current Contract

```text
H = 32
R = 16
S_k = ShortVisualMemory(V_{t_k-R/2}, V_{t_k-R})
x_k = ProgressEvidenceEncoder(h_k, s_k, u_k)
M_k = ProgressStateUpdater(M_{k-1}, x_k)
P_k = Planner(M_k, h_k, s_k)
```

Current warm-up runs use:

```text
M_k = [C_k, G_k]: [B, 2, 896]
P_k: [B, 1, 896]
intent target z_k: [B, 128]
```

The planner-token count is intentionally kept at one for the completed warm-up weights. Changing it should be a separate model revision after discussion.

## Completed Training Work

```text
LIBERO W=4 progress warm-up cache built
LIBERO W=4 progress-state warm-up trained
RMBench 14-dim H32 intent AE trained
RMBench W=4 and W=8 progress warm-up caches built
RMBench W=4 and W=8 progress-state warm-up weights trained
RMBench W=8 stopped after step_000700.pt
training summaries generated from logs
```

## Best Current Artifacts

```text
RMBench intent AE:
  /root/autodl-tmp/runs/progress_warmup/rmbench_h32_intent_ae_v1/best.pt
  best step: 950
  val_loss: 0.015030

LIBERO W=4 progress warm-up:
  /root/autodl-tmp/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt
  best step: 310
  val_loss: 0.017872

RMBench W=4 progress warm-up:
  /root/autodl-tmp/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt
  best step: 590
  val_loss: 0.001225

RMBench W=8 progress warm-up:
  /root/autodl-tmp/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w8_bs6656_epval_v1/best.pt
  best step: 660
  val_loss: 0.001016
```

RMBench W=8 is the best current warm-up run on the main validation objective.

## Active Entry Points

```text
docs/progress_state_planner_design_zh.md      current long-memory and planner design
docs/current_project_state.md                 detailed state, artifacts, metrics, next work
docs/engineering_reproducibility.md           reproducibility and warm-up commands
docs/bridge_himem_design.md                   Progress-state planner surface
docs/project_structure.md                     code/config/docs/output boundaries
coarse_planner/README.md                      legacy H32 baseline commands
```

## Next Work

1. Decide whether the planner remains one intent token or splits into multiple intent tokens.
2. Inspect W=4 vs W=8 per-suite behavior before changing the planner structure.
3. Integrate short visual-token memory on the policy side after the planner-token decision.
4. Keep LIBERO and RMBench data code separated unless a shared abstraction becomes clearly useful.

## Guardrails

- Do not restart transition-trigger work for this path.
- Do not reintroduce `PlanTokenQueue`, consumed-step suffix state, or cached plan refresh policy.
- Do not model long memory as a growing visual-token bank.
- Do not use future actions as long-memory input; future actions may only be targets.
- Do not skip planner warm-up.
- Keep large datasets, caches, checkpoints, and run outputs off the system disk.
