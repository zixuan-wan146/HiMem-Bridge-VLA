# Current Project State

Date: 2026-06-18

## Repository Locations

```text
local:  <local_workspace>/HiMem-Bridge-VLA
remote: $AUTODL_TMP/HiMem-Bridge-VLA
head:   f1cd2ec
```

The local and remote working trees have been synced. The repository is intentionally dirty because
the transition-trigger integration has not yet been split into commits.

Remote runtime/data/model artifacts live on the data disk:

```text
$AUTODL_TMP/runs/transition_trigger
$AUTODL_TMP/datasets
$AUTODL_TMP/checkpoints
$AUTODL_TMP/hf-home
```

Repo-local LIBERO smoke/eval outputs live under ignored `runs/` directories inside the remote
repository checkout.

## Active Workstream

The active workstream is the standalone `transition_trigger` module and its HiMem/LIBERO runtime
integration.

Current status:

```text
offline transition-trigger training: done
offline ablation summary: done
selected runtime package: done
HiMem server integration: implemented
LIBERO transition-frame adapter: implemented
LIBERO JSONL trace logging: implemented
trace summary utility: implemented
closed-loop LIBERO smoke/eval: integration verified, not a performance claim
```

The next workstream started on 2026-06-18 is being refactored into the new
Coarse Planner intent-latent path:

```text
CoarsePlanner(H_t_vlm, s_t) -> plan_tokens
ActionSegmentAutoencoder(action segments) -> intent latents for training supervision
active plan-token suffix + memory_context -> BridgeAttention condition path
```

Current Coarse Planner implementation status:

```text
old compressed coarse-action target: removed from the active standalone planner path
action-segment target construction: implemented for planner feature caches
action-segment autoencoder module: implemented and trained on LIBERO H64
planner latent prediction head: implemented and trained on frozen AE latents
plan suffix queue by cumulative executed steps: implemented for runtime/session semantics
strict anchor/current training sample contract: reserved for BridgeAttention/ActionHead integration
training status: AE and standalone Coarse Planner warm-up completed for LIBERO H64
```

Planner token source decision:

```text
default token source: InternVL3 fused_tokens with return_cls_only=False
optional ablation: selected hidden_state layer
state handling: stored separately, fused inside CoarsePlanner as state_proj(state) token
```

Current CALVIN data note:

```text
parquet files are present under $AUTODL_TMP/datasets/calvin/lerobot/task_ABC_D
full videos/chunk-000/image is currently empty
smoke subset with 4 parquet/video pairs was created under $AUTODL_TMP/datasets/calvin/lerobot/task_ABC_D_smoke
real planner feature cache smoke succeeded with 20 fused-token samples
cache: $AUTODL_TMP/datasets/coarse_planner/calvin_abc_d_smoke
train smoke: $AUTODL_TMP/runs/coarse_planner/calvin_abc_d_smoke
```

Primary Coarse Planner documents:

```text
docs/coarse_planner_design.md
docs/action_segment_autoencoder_coarse_planner_config.md
docs/coarse_planner_intent_latent_libero_h64_results.md
to-do/6-18.md
docs/coarse_planner_libero_ablation.md
```

Final LIBERO H64 experiment configs:

```text
coarse_planner/configs/libero_h64_segment_ae_v2.yaml
coarse_planner/configs/libero_h64_planner_v2.yaml
coarse_planner/configs/libero_h64_planner_znorm_v3.yaml
coarse_planner/configs/libero_h64_planner_znorm_chunk1_v4.yaml
coarse_planner/configs/libero_h64_s8192_build.yaml
coarse_planner/configs/libero_h64_planner_znorm_latew_s8192_v5.yaml
coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s8192_v8.yaml
coarse_planner/configs/libero_h64_s16384_build.yaml
coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s16384_v10.yaml
coarse_planner/configs/libero_h64_s32768_build.yaml
coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s32768_v11.yaml
coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s32768_v12.yaml
coarse_planner/configs/libero_h64_planner_znorm_latefocus_s32768_v13.yaml
coarse_planner/configs/libero_h64_s63044_build.yaml
coarse_planner/configs/libero_h64_planner_znorm_latefocus_s63044_v16.yaml
```

Current LIBERO H64 intent-latent artifacts:

```text
train/eval cache:
$AUTODL_TMP/datasets/coarse_planner/libero_h64

external holdout cache:
$AUTODL_TMP/datasets/coarse_planner/libero_h64_holdout_seed43

AE run:
$AUTODL_TMP/runs/coarse_planner/libero_h64_segment_ae_v2

Planner run:
$AUTODL_TMP/runs/coarse_planner/libero_h64_planner_v2

Recommended z-normalized Planner run:
$AUTODL_TMP/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075

final evaluation:
$AUTODL_TMP/runs/coarse_planner/libero_h64_planner_v2/final_eval_seed43.json

z-normalized comparison:
$AUTODL_TMP/runs/coarse_planner/znorm_comparison_seed43.json
```

Current LIBERO H64 intent-latent results:

```text
ActionSegmentAutoencoder:
  epochs: 100
  batch_size: 2048
  peak CUDA reserved: 19.984 GB
  best epoch: 100
  original eval loss: 0.015466
  seed43 holdout all loss: 0.015194
  32k cache eval loss: 0.016337
  32k cache eval rec loss: 0.014796
  32k cache eval gripper accuracy: 0.994475

CoarsePlanner:
  epochs: 100
  batch_size: 640
  AMP: true
  peak CUDA reserved: 23.025 GB
  best epoch: 85
  original eval loss: 0.242264
  original eval latent MSE: 0.158486
  seed43 holdout all loss: 0.242503
  seed43 holdout all latent MSE: 0.157849

CoarsePlanner z-normalized v4:
  AE: frozen
  warm start: raw-z v2 best checkpoint with latent-head conversion
  chunk_loss_weight: 1.0
  best epoch: 102
  peak CUDA reserved: 22.969 GB
  original eval raw latent MSE: 0.155831
  original eval decoded chunk loss: 0.342241
  original eval cosine: 0.828862
  seed43 holdout all raw latent MSE: 0.153206
  seed43 holdout all decoded chunk loss: 0.338868
  seed43 holdout all cosine: 0.828580

CoarsePlanner z-normalized v10:
  AE: frozen
  data: 16384 LIBERO H64 samples
  warm start: v9/v8/v5/v4 planner chain
  loss: normalized latent MSE + 0.5 decoded chunk loss, moderate late-token weights
  selection metric: val_raw_latent_mse
  checkpoint: $AUTODL_TMP/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s16384_v10/best.pt
  original eval raw latent MSE: 0.112382
  original eval decoded chunk loss: 0.233182
  original eval cosine: 0.874485
  seed43 holdout all raw latent MSE: 0.099159
  seed43 holdout all decoded chunk loss: 0.178007
  seed43 holdout all cosine: 0.888414

CoarsePlanner z-normalized v11:
  AE: frozen
  data: 32768 LIBERO H64 samples
  warm start: v10 best checkpoint
  loss: normalized latent MSE + 0.5 decoded chunk loss, moderate late-token weights
  selection metric: val_raw_latent_mse
  checkpoint: $AUTODL_TMP/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v11/best.pt
  training note: stopped after epoch 4 because epoch 2 remained best
  original eval raw latent MSE: 0.113005
  original eval decoded chunk loss: 0.205989
  original eval cosine: 0.872520
  seed43 holdout all raw latent MSE: 0.098461
  seed43 holdout all decoded chunk loss: 0.155727
  seed43 holdout all cosine: 0.888590

CoarsePlanner z-normalized v12:
  AE: frozen
  data: 32768 LIBERO H64 samples
  warm start: v11 best checkpoint
  loss: normalized latent MSE + 0.25 decoded chunk loss, moderate late-token weights
  model dropout: 0.0
  selection metric: val_raw_latent_mse
  checkpoint: $AUTODL_TMP/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v12/best.pt
  training note: stopped after epoch 4 because epoch 2 remained best
  original eval raw latent MSE: 0.110009
  original eval decoded chunk loss: 0.238227
  original eval cosine: 0.876685
  seed43 holdout all raw latent MSE: 0.091676
  seed43 holdout all decoded chunk loss: 0.174122
  seed43 holdout all cosine: 0.896749

CoarsePlanner z-normalized v13:
  AE: frozen
  data: 32768 LIBERO H64 samples
  warm start: v12 best checkpoint
  loss: normalized latent MSE + 0.1 decoded chunk loss, aggressive late-token weights
  model dropout: 0.0
  selection metric: val_raw_latent_mse
  checkpoint: $AUTODL_TMP/runs/coarse_planner/libero_h64_planner_znorm_latefocus_s32768_v13/best.pt
  training note: stopped after epoch 3 because epoch 1 remained best
  original eval raw latent MSE: 0.109229
  original eval decoded chunk loss: 0.258267
  original eval cosine: 0.877654
  seed43 holdout all raw latent MSE: 0.088456
  seed43 holdout all decoded chunk loss: 0.180509
  seed43 holdout all cosine: 0.900542

CoarsePlanner z-normalized v17:
  AE: frozen
  data: 32768 LIBERO H64 samples
  artifact: checkpoint interpolation, v12 + 0.75 * (v13 - v12)
  checkpoint: $AUTODL_TMP/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075/best.pt
  original eval raw latent MSE: 0.108633
  original eval decoded chunk loss: 0.251312
  original eval cosine: 0.878329
  seed43 holdout all raw latent MSE: 0.088253
  seed43 holdout all decoded chunk loss: 0.176643
  seed43 holdout all cosine: 0.900730

63k cache:
  data: $AUTODL_TMP/datasets/coarse_planner/libero_h64_s63044_seed42
  samples: 63044
  split: train 56745, eval 6299
  size: about 108 GB
  status: built successfully, but v15/v16 training did not improve planner quality;
          removed during post-run disk cleanup
```

Interpretation:

```text
The AE latent space generalizes to a newly sampled LIBERO H64 cache.
The planner-only push improved seed43 holdout all raw latent MSE from 0.153218
to 0.088253 without changing the AE or inference path.
The near-term <=0.12 gate is reached.
The mid-term <=0.10 gate is reached on seed43 holdout all.
The hard ~=0.08 usable target is not reached.
The remaining gap is concentrated in late plan tokens p5-p7.
The 32k v11 diagnosis also shows that the input-context nearest-neighbor
baseline is now about 0.0825 on seed43 holdout all. The v12/v13/v17 latent-focused
continuations close much of that gap, but the current planner is still above the
strict 0.08 line.
```

Current planner quality target:

```text
near-term gate: raw_latent_mse <= 0.12
mid-term gate:  raw_latent_mse <= 0.10
usable target:  raw_latent_mse ~= 0.08
```

The `0.08` target is the usable-stage goal before promoting the standalone
planner into the main BridgeAttention / ActionHead training path. It is not a
single-metric objective: decoded chunk loss should not regress from v4
(`0.338868` on seed43 holdout), latent cosine should improve from v4
(`0.828580` on seed43 holdout), and late-token / suffix behavior must remain
stable.

Next decision boundary:

```text
Do not start full end-to-end VLA training in this round. The current 0.088 raw
latent MSE planner is good enough to move to the next integration stage, but the
next stage is BridgeAttention / ActionHead suffix training, not joint main-model
training.
First diagnose and improve the frozen-AE Coarse Planner toward the 0.08 usable
target.
Then train BridgeAttention / ActionHead with active plan-token suffixes.
After that integration path is stable, run joint end-to-end main model training.
```

Scope for the current round:

```text
included: standalone frozen-AE Coarse Planner diagnosis and improvement toward 0.08
excluded: BridgeAttention / ActionHead integration training
excluded: joint end-to-end main VLA training
```

Current checkpoint decision:

```text
Use v17 as the current standalone planner checkpoint.
Do not start BridgeAttention / ActionHead integration training in this current
round; it is the next stage after this planner-only checkpoint.
Further planner-only improvement should change model/data strategy rather than
blindly continuing low-lr fine-tuning; v14-v16 did not improve over v17.
```

Remote cleanup state:

```text
retained datasets:
  $AUTODL_TMP/datasets/coarse_planner/libero_h64
  $AUTODL_TMP/datasets/coarse_planner/libero_h64_holdout_seed43
  $AUTODL_TMP/datasets/coarse_planner/libero_h64_s32768_seed42

retained planner runs:
  $AUTODL_TMP/runs/coarse_planner/libero_h64_segment_ae_v2
  $AUTODL_TMP/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075

removed:
  intermediate 8k/16k/63k caches
  failed v14-v16 runs
  old v2-v13/probe/smoke planner runs
```

## Selected Runtime Model

```text
package:
$AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512

config:
transition_trigger/configs/selected/robomme_rmbench_w32_value_delta_transformer_d512.yaml

model: causal Transformer
window_size: 32
feature_set: value_delta_mask
input_dim: 144
d_model: 512
num_layers: 4
num_heads: 8
parameters: 12,980,737

planner_threshold: 0.70
memory_write_threshold: 0.80
score_mode: causal_peak
```

Backbone cache:

```text
OpenGVLab/InternVL3-1B
$AUTODL_TMP/hf-home/hub/models--OpenGVLab--InternVL3-1B
snapshot: 4415a3b810e636d11dfa86b0e9ba40bb00535aa8
model.safetensors: 1,876,463,472 bytes
```

## Experiment Record

Primary result document:

```text
docs/transition_trigger_ablation_results.md
```

Paper-facing temporal-window figure:

```text
metric: Transition-F1 (%)
W24: 46.5
W32: 47.9
W48: 36.8
```

Online replay metrics for the selected runtime package:

```text
test all_plan:     F1 0.685, precision 0.801, recall 0.598
test memory_write: F1 0.532, precision 0.982, recall 0.365
eval all_plan:     F1 0.702, precision 0.812, recall 0.619
eval memory_write: F1 0.551, precision 0.977, recall 0.383
```

These are trigger sets from the same transition-trigger runtime family, not two separate models.
`all_plan` is the replan trigger set. `memory_write` is the high-confidence memory-write subset.

## Closed-Loop LIBERO Record

35-step smoke:

```text
run:
$AUTODL_TMP/HiMem-Bridge-VLA/runs/transition_trigger/libero_closed_loop_trace_smoke/20260617_212202_trace_jsonl_smoke

result: client_status 0, 1 episode, 35/35 steps, max_steps_exhausted
trace: 35 rows, 3 scored rows, max score 0.5480, 0 triggers
```

80-step trace eval:

```text
run:
$AUTODL_TMP/HiMem-Bridge-VLA/runs/transition_trigger/libero_closed_loop_trace_eval/20260617_214140_long_trace_eval

result: client_status 0, 3 episodes, 80/80 steps each, max_steps_exhausted
trace: 240 rows, 144 scored rows, max score 0.6934, 0 triggers
```

Interpretation:

```text
The closed-loop integration is stable.
The current threshold is conservative for LIBERO: max observed score 0.6934 < planner_threshold 0.70.
LIBERO should remain an integration/observability result unless we calibrate or train for LIBERO.
```

## Dirty Tree Grouping

Use these groups when splitting commits. Do not mix experiment artifacts with runtime integration
unless the file boundary requires it.

### 1. Transition Trigger Runtime

```text
transition_trigger/__init__.py
transition_trigger/config.py
transition_trigger/trigger_policy.py
transition_trigger/online_features.py
transition_trigger/runtime.py
transition_trigger/configs/selected/
transition_trigger/scripts/evaluate_runtime_policy.py
tests/test_transition_trigger_causal.py
```

Purpose:

```text
Selected causal-peak runtime policy, online feature construction, selected config, and replay eval.
```

### 2. HiMem Server Integration

```text
himem_bridge_vla/server_protocol.py
himem_bridge_vla/transition_trigger_manager.py
himem_bridge_vla/model/himem_bridge_vla.py
scripts/himem_server.py
tests/test_server_protocol.py
tests/test_transition_trigger_manager.py
tests/test_himem_server_transition_trigger.py
```

Purpose:

```text
Load the selected trigger package in the server, keep per-episode trigger sessions, and route
memory writes/replanning through trigger decisions.
```

### 3. LIBERO Adapter, Trace Logging, And Run Scripts

```text
evaluations/libero/libero_action_protocol.py
evaluations/libero/libero_client_4tasks.py
evaluations/libero/libero_client_config.py
evaluations/libero/libero_transition_frame.py
evaluations/libero/libero_transition_trace.py
evaluations/libero/libero_transition_trace_summary.py
scripts/libero_profile.sh
scripts/run_libero_eval.sh
scripts/run_libero_smoke.sh
scripts/summarize_libero_transition_trace.py
scripts/write_libero_run_manifest.py
tests/test_libero_action_protocol.py
tests/test_libero_client_config.py
tests/test_libero_transition_frame.py
tests/test_libero_transition_trace.py
tests/test_libero_transition_trace_summary.py
tests/test_run_libero_eval_script.py
tests/test_run_libero_smoke_script.py
tests/test_write_libero_run_manifest.py
```

Purpose:

```text
Send transition_frame from LIBERO, parse debug responses, write per-decision JSONL traces, and
summarize those traces after smoke/eval runs.
```

### 4. Documentation And Experiment Notes

```text
docs/causal_memory_replan_design.md
docs/transition_trigger_ablation_results.md
docs/current_project_state.md
transition_trigger/README.md
to-do/6-17.md
to-do/project_cleanup_plan.md
```

Purpose:

```text
Explain the module, selected model, ablation results, closed-loop trace results, and next steps.
```

### 5. Repository Hygiene

```text
.gitignore
```

Purpose:

```text
Ignore generated run outputs such as runs/.
```

## Verification Commands

Focused tests already used for the trace logging path:

```bash
$AUTODL_TMP/miniforge3/envs/Evo1/bin/python -m pytest -q \
  tests/test_libero_client_config.py \
  tests/test_libero_transition_trace.py \
  tests/test_run_libero_smoke_script.py \
  tests/test_run_libero_eval_script.py \
  tests/test_write_libero_run_manifest.py
```

Trace summary tests:

```bash
$AUTODL_TMP/miniforge3/envs/Evo1/bin/python -m pytest -q \
  tests/test_libero_transition_trace_summary.py
```

Trace summary command:

```bash
python scripts/summarize_libero_transition_trace.py <trace.jsonl> --format text
```

Runtime replay command:

```bash
python transition_trigger/scripts/evaluate_runtime_policy.py \
  --package-dir $AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512 \
  --split test \
  --device cuda
```

Coarse Planner lightweight verification used locally:

```bash
python3 -m py_compile \
  himem_bridge_vla/model/planner/coarse_planner.py \
  himem_bridge_vla/model/planner/session.py \
  himem_bridge_vla/dataset/action_segments.py \
  himem_bridge_vla/dataset/cache_utils.py \
  himem_bridge_vla/dataset/simulation_dataset.py \
  himem_bridge_vla/model/bridge/bridge_attention.py \
  himem_bridge_vla/model/bridge/adapter.py \
  himem_bridge_vla/model/himem_bridge_vla.py \
  himem_bridge_vla/bridge_himem_config.py \
  himem_bridge_vla/training_loss.py \
  scripts/train.py
  scripts/himem_server.py
```

```bash
python3 -m pytest -q \
  tests/test_coarse_planner.py \
  tests/test_coarse_planner_bridge_integration.py \
  tests/test_coarse_plan_session.py \
  tests/test_action_segment_targets.py \
  tests/test_bridge_attention.py \
  tests/test_bridge_himem_config.py \
  tests/test_training_loss.py \
  tests/test_train_script_config.py \
  tests/test_dataset_cache_utils.py \
  tests/test_himem_server_transition_trigger.py -rs
```

The local machine currently lacks `torch`, so tensor forward/loss tests are skipped locally and
must be rerun in the remote training environment after the server is available again.

## Next Decisions

1. Split the dirty tree into the commit groups above.
2. Decide LIBERO policy scope:
   keep `planner_threshold=0.70` for conservative integration only, or add an explicit
   LIBERO/demo calibration threshold such as `0.65`.
3. Keep paper claims anchored on the offline transition-trigger result unless LIBERO-specific
   calibration/training is added.
