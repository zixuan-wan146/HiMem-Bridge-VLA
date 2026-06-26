# Current Project State

Date: 2026-06-26

## Active Direction

The active design is the progress-state planner with separate short and long memory roles:

```text
short memory = recent visual-token memory
long memory  = planner-coupled recurrent task-progress state
```

The current runtime contract is:

```text
H = 32
R = 16
S_k = ShortVisualMemory(V_{t_k-R/2}, V_{t_k-R})
x_k = ProgressEvidenceEncoder(h_k, s_k, u_k)
M_k = ProgressStateUpdater(M_{k-1}, x_k)
P_k = Planner(M_k, h_k, s_k)
```

Current warm-up weights use one planner token:

```text
P_k: [B, 1, 896]
M_k = [C_k, G_k]: [B, 2, 896]
intent latent z_k: [B, 128]
```

See `docs/progress_state_planner_design_zh.md` for the full design.

## Completed Today

```text
implemented separate LIBERO and RMBench progress warm-up cache/dataset paths
implemented RMBench 14-dim H32 action-intent autoencoder training
built LIBERO W=4 progress warm-up cache
built RMBench W=4 and W=8 progress warm-up caches
trained LIBERO W=4 progress-state warm-up
trained RMBench W=4 progress-state warm-up
trained RMBench W=8 progress-state warm-up and stopped after step_000700.pt
generated early-stop summaries from logs
documented current single-token planner contract
```

## Remote Locations

```text
repo:      /root/autodl-tmp/HiMem-Bridge-VLA
data root: /root/autodl-tmp
runs:      /root/autodl-tmp/runs/progress_warmup
caches:    /root/autodl-tmp/token_caches
```

Large datasets, caches, checkpoints, and run outputs stay on the remote data disk. Local sync is for code and documentation only.

## Cache Inventory

```text
LIBERO W=4:
  cache: /root/autodl-tmp/token_caches/libero_progress_vl_embedding_h32_r16_w4
  step_count:   18199
  window_count: 12199

RMBench W=4:
  cache: /root/autodl-tmp/token_caches/rmbench_progress_vl_embedding_h32_r16_w4
  step_count:   16676
  window_count: 15326

RMBench W=8:
  cache: /root/autodl-tmp/token_caches/rmbench_progress_vl_embedding_h32_r16_w8
  step_count:   16676
  window_count: 13526
```

Warm-up cache stores pooled VL embedding, not full visual tokens:

```text
vl_summary: [896]
state
executed_actions: [16, action_dim]
executed_action_mask: [16]
target_intent: [128]
```

## Current Metrics

RMBench H32 action-intent AE:

```text
run: /root/autodl-tmp/runs/progress_warmup/rmbench_h32_intent_ae_v1
best step: 950
val_loss: 0.015030
val_segment_ae_rec_loss: 0.014776
```

LIBERO progress-state warm-up, W=4:

```text
run: /root/autodl-tmp/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1
best step: 310
val_loss: 0.017872
val_plan_loss: 0.011099
val_stage_loss: 0.011190
val_mem_pool_loss: 0.011780
val_cos_g_p: -0.021132
val_stage_effective_rank: 118.357338
```

RMBench progress-state warm-up, W=4:

```text
run: /root/autodl-tmp/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w4_bs12800_epval_v1
best step: 590
val_loss: 0.001225
val_plan_loss: 0.000732
val_stage_loss: 0.000779
val_mem_pool_loss: 0.001029
val_cos_g_p: 0.005919
val_stage_effective_rank: 93.020485
```

RMBench progress-state warm-up, W=8:

```text
run: /root/autodl-tmp/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w8_bs6656_epval_v1
stopped after: step_000700.pt
best checkpoint: best.pt
best step: 660
val_loss: 0.001016
val_plan_loss: 0.000563
val_stage_loss: 0.000707
val_mem_pool_loss: 0.000991
val_cos_g_p: 0.000006
val_stage_effective_rank: 83.066261
```

RMBench W=8 improves the main validation objective over W=4:

```text
val_loss:      0.001225 -> 0.001016
val_plan_loss: 0.000732 -> 0.000563
```

`val_cos_g_p` remains close to zero, so the raw current-stage token and raw planner token did not collapse into the same representation in these warm-up runs.

## Superseded From Active Path

```text
H64 multi-token planner caches and checkpoints
PlanTokenQueue / suffix consumption
transition trigger model, server wiring, eval trace code, and tests
transition-trigger dataset conversion scripts
LIBERO transition-frame trace logging
Dual-FIFO long visual-memory design
H32 action-latent planner as the main method definition
```

The checked-in H32 standalone planner artifacts remain useful as a baseline and auxiliary target source. They should not define the current planner semantics.

## Next Work

```text
1. decide whether planner should keep one token or split into multiple intent tokens
2. inspect W=4 vs W=8 behavior per suite/task before changing model structure
3. integrate short visual-token memory on the policy side after planner-token count is decided
4. keep LIBERO and RMBench data code separated unless a truly shared abstraction becomes obvious
```
