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

`h_k` is the progress planner VL summary. Warm-up caches store it directly as `vl_summary`. Active direct bridge Stage1 token caches must store the same value as `planner_vl_summary`; if a non-Stage1 smoke batch only has token features, the model falls back to deterministic token mean pooling.

Current warm-up weights use one planner token, and the action head expands that token into plan slots:

```text
P_k base token: [B, 1, 896]
plan slots:     [B, 8, 896]
M_k = [C_k, G_k]: [B, 2, 896]
intent latent z_k: [B, 128]
```

The current policy-side action architecture is direct bridge-attention:

```text
32 noisy action tokens
  -> action self-attn
  -> visual cross-attn over [current VLM hidden states, short memory]
  -> action-condition cross-attn over [plan slots, state token]
  -> FFN
```

Fixed policy-side context:

```text
VLM layers: [3, 6, 9, 12]
action block layer schedule: [3, 3, 6, 6, 9, 9, 12, 12]
short memory: 32 tokens = 16 tokens from t_k - R + 16 tokens from t_k - R/2
plan slots: 8 tokens expanded from one planner token
state token: 1 token from proprio/state MLP
Euler inference steps: 15
```

See `docs/progress_state_planner_design_zh.md` and `docs/direct_bridge_attention_design_zh.md` for the full design.

## Completed Today

```text
implemented separate LIBERO and RMBench progress warm-up cache/dataset paths
implemented RMBench 14-dim H32 action-intent autoencoder training
implemented direct bridge-attention action head
implemented plan-token virtual slot expansion from 1 token to 8 slots
implemented short-memory adapter and source/time embeddings in the action head
connected progress-state planner checkpoints into the direct bridge policy path
added explicit planner_vl_summary input for reproducible planner conditioning
added reproducible direct bridge inference smoke test
added direct bridge token-cache training smoke test
added direct bridge token-cache training smoke with progress checkpoint plan-token source
built LIBERO W=4 progress warm-up cache
built RMBench W=4 and W=8 progress warm-up caches
trained LIBERO W=4 progress-state warm-up
trained RMBench W=4 progress-state warm-up
trained RMBench W=8 progress-state warm-up and stopped after step 700
generated early-stop summaries from logs
documented current single-token planner contract
```

## Remote Locations

```text
repo:      $AUTODL_TMP/HiMem-Bridge-VLA
data root: $AUTODL_TMP
runs:      $AUTODL_TMP/runs/progress_warmup
caches:    $AUTODL_TMP/token_caches
```

Large datasets, caches, checkpoints, and run outputs stay on the remote data disk. Local sync is for code and documentation only.

## Cache Inventory

```text
LIBERO W=4:
  cache: $AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w4
  step_count:   18199
  window_count: 12199

LIBERO W=8:
  cache: $AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w8
  step_count:   18199
  window_count: 5429

RMBench W=4:
  cache: $AUTODL_TMP/token_caches/rmbench_progress_vl_embedding_h32_r16_w4
  step_count:   16676
  window_count: 15326

RMBench W=8:
  cache: $AUTODL_TMP/token_caches/rmbench_progress_vl_embedding_h32_r16_w8
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

Small direct-bridge smoke cache:

```text
LIBERO image_stats H896 smoke:
  cache: $AUTODL_TMP/token_caches/libero_memory_replay_image_stats_h896_smoke
  format: memory_replay_visual_token_cache
  sample_count: 2
  purpose: checkpoint-conditioned direct bridge H32 training smoke only
```

## Current Metrics

RMBench H32 action-intent AE:

```text
run: $AUTODL_TMP/runs/progress_warmup/rmbench_h32_intent_ae_v1
best step: 950
val_loss: 0.015030
val_segment_ae_rec_loss: 0.014776
```

LIBERO progress-state warm-up, W=4:

```text
run: $AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1
best step: 310
val_loss: 0.017872
val_plan_loss: 0.011099
val_stage_loss: 0.011190
val_mem_pool_loss: 0.011780
val_cos_g_p: -0.021132
val_stage_effective_rank: 118.357338
```

LIBERO progress-state warm-up, W=8:

```text
run: $AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_bs6656_epval_v1
best step: 280
val_loss: 0.021811
val_plan_loss: 0.013405
val_stage_loss: 0.013884
val_mem_pool_loss: 0.014639
val_cos_g_p: -0.023982
val_stage_effective_rank: 78.541298
```

RMBench progress-state warm-up, W=4:

```text
run: $AUTODL_TMP/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w4_bs12800_epval_v1
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
run: $AUTODL_TMP/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w8_bs6656_epval_v1
stopped after: step 700
per-step checkpoints: pruned after summary generation
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
1. build real InternVL3 visual-token replay caches at training scale
2. run direct bridge policy training with progress planner checkpoint loaded
3. run LIBERO/RMBench inference smoke from saved policy checkpoints
4. keep LIBERO and RMBench data code separated unless a truly shared abstraction becomes obvious
```
