# Transition Trigger Ablation Results

Status: completed transformer-only ablation on RoboMME four-task + RMBench nine-task transition data.

## Selected Model

Runtime recommendation:

```text
config: transition_trigger/configs/selected/robomme_rmbench_w32_value_delta_transformer_d512.yaml
checkpoint: $AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512/checkpoint.pt
source run: $AUTODL_TMP/runs/transition_trigger/robomme_rmbench_ablations/w32_value_delta_mask_transformer_d512
checkpoint type: best_memory_write
```

This checkpoint is selected for the deployed memory-write plus replan policy. It is not the
highest boundary F1 checkpoint, but it keeps memory-write behavior usable at the fixed high
threshold while preserving strong event-level triggering.

Selected trigger size and input contract:

```text
model: causal Transformer
input_dim: 144
window_size: 32
d_model: 512
num_layers: 4
num_heads: 8
parameters: 12,980,737
```

This is the transition-trigger head only. It is separate from the HiMem/InternVL3 backbone. The
remote runtime cache contains `OpenGVLab/InternVL3-1B` under:

```text
$AUTODL_TMP/hf-home/hub/models--OpenGVLab--InternVL3-1B
snapshot: 4415a3b810e636d11dfa86b0e9ba40bb00535aa8
model.safetensors: 1,876,463,472 bytes
```

Boundary upper checkpoint:

```text
checkpoint: $AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512/boundary_upper_checkpoint.pt
checkpoint type: best_event_f1
```

Use the boundary upper checkpoint for analysis tables where the question is purely event trigger
quality. Use the selected checkpoint for system integration.

Conservative memory-write fallback:

```text
checkpoint: $AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512/conservative_memory_checkpoint.pt
source run: w24_value_delta_mask_transformer_d512
checkpoint type: best_memory_write
```

This W24 fallback has the strongest fixed-threshold memory F1 among the compact temporal-window
models, but it gives up some event/replan quality compared with the W32 runtime default.

Runtime loader:

```python
from transition_trigger.runtime import load_selected_trigger

trigger = load_selected_trigger(device="cuda")
output = trigger.decide_window(features)  # features: [32, 144]
```

Online replay:

```bash
python transition_trigger/scripts/evaluate_runtime_policy.py \
  --package-dir $AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512 \
  --split test \
  --device cuda
```

## Metric Glossary

The detector is evaluated at two levels. Frame-level metrics score every valid frame, while
event-level metrics first turn scores into triggers and then match those triggers to transition
events.

```text
AUPRC / frame_auprc:
  Area under the precision-recall curve over per-frame scores. This measures score ranking quality
  before choosing a runtime threshold.

threshold:
  Score cutoff used to convert scores into triggers. For runtime policy, tau_p=0.70 controls
  planner triggers and tau_w=0.80 controls memory writes.

precision:
  Fraction of predicted triggers that match a true transition. Higher precision means fewer false
  triggers.

recall:
  Fraction of true transitions that are matched by a predicted trigger. Higher recall means fewer
  missed transitions.

F1:
  Harmonic mean of precision and recall. It is the main compact event-level summary when precision
  and recall are both important.

mean_trigger_delay:
  Average frame delay between a matched prediction and the true event. Positive delay means the
  trigger fired after the event, which is expected for causal post-boundary detection.

early_trigger_rate:
  Fraction of events with a nearby trigger that fired before the valid post-boundary window. Lower
  is better.

duplicate_triggers_per_event:
  Extra triggers near an event after that event has already been matched. Lower is better.

triggers_per_100_frames:
  Trigger density. This is useful for detecting overly chatty policies even when F1 looks acceptable.

all_plan:
  Runtime trigger set equal to soft_plan OR hard_plan. This is the planner/replan behavior.

memory_write:
  High-confidence subset that writes memory and implies hard_plan. It is intentionally more
  precision-oriented than all_plan.
```

## Main Ablation

Fixed settings:

```text
dataset: RoboMME four-task + RMBench nine-task split manifest
feature set: value_delta_mask
model: causal Transformer
d_model: 512
memory-write threshold: 0.8
```

| Window | Checkpoint | Test Event F1 | Test AUPRC | Memory F1@0.8 |
| ---: | --- | ---: | ---: | ---: |
| 24 | best_event_f1 | 0.465 | 0.567 | 0.291 |
| 24 | best_memory_write | 0.431 | 0.643 | 0.392 |
| 32 | best_event_f1 | 0.479 | 0.519 | 0.051 |
| 32 | best_memory_write | 0.442 | 0.564 | 0.363 |
| 48 | best_event_f1 | 0.368 | 0.531 | 0.118 |
| 48 | best_memory_write | 0.349 | 0.624 | 0.302 |

Conclusion:

```text
W32 gives the best boundary/replan event trigger.
W24 gives the most conservative memory-write behavior at threshold 0.8.
W48 degrades despite larger context, so longer history is not automatically better.
```

This supports using a medium causal history. A short window misses part of the action/state trend,
while an overly long window dilutes local transition evidence and increases compute.

## Checkpoint Roles

These are not separate architectures; they are different checkpoints or runtime trigger sets from
the same transition-trigger family.

| Role | Run / Checkpoint | Use | Key Test Result |
| --- | --- | --- | --- |
| Runtime default | W32 `best_memory_write` | Closed-loop integration, memory-write plus replan | online `all_plan` F1 0.685, online `memory_write` precision 0.982 |
| Boundary upper | W32 `best_event_f1` | Paper/event upper analysis | event F1 0.479 |
| Conservative memory fallback | W24 `best_memory_write` | More conservative memory-write fallback | memory F1@0.8 0.392 |

The runtime default and boundary upper are from the same W32 architecture. The runtime default is
chosen for deployed policy behavior rather than the maximum single-threshold boundary F1.

## Feature And Width Findings

The strongest feature set is:

```text
value_delta_mask
```

The weaker alternatives were:

```text
value_mask
value_mask_domain
full
```

Adding domain/full features did not improve the trigger and often lowered event F1. For this
module, explicit values, deltas, and validity masks are enough; extra source/domain information is
not worth the added coupling yet.

For the W32 Transformer d512 setting, the feature comparison on test event F1 is:

| Feature Set | Best Event F1 | Best-Memory Event F1 | Best-Memory Memory F1@0.8 |
| --- | ---: | ---: | ---: |
| value_delta_mask | 0.479 | 0.442 | 0.363 |
| full | 0.419 | 0.362 | 0.339 |
| value_mask | 0.336 | 0.305 | 0.244 |
| value_mask_domain | 0.318 | 0.250 | 0.126 |

The stronger model width is:

```text
d_model = 512
```

The d256 variants are consistently weaker for the same window and feature set.

For the W32 value_delta_mask Transformer setting:

| Width | Best Event F1 | Best-Memory Event F1 | Best-Memory Memory F1@0.8 |
| ---: | ---: | ---: | ---: |
| 256 | 0.414 | 0.282 | 0.100 |
| 512 | 0.479 | 0.442 | 0.363 |

## Model Family Finding

Transformer runs dominate the lightweight SSM runs in this setup. The remaining development path
should focus on Transformer variants, thresholds, and online integration rather than spending more
budget on the current SSM branch.

## Runtime Policy

After causal peak confirmation, the action policy is:

```text
if score >= memory_write_threshold:
    write_memory()
    hard_plan()
elif score >= planner_threshold:
    soft_plan()
```

Current selected thresholds:

```text
score_mode = causal_peak
planner_threshold = 0.70
memory_write_threshold = 0.80
replan_cooldown_frames = 10
memory_write_cooldown_frames = 10
memory_write_implies_plan = true
```

Memory write is a high-confidence commit. Planner triggering can be lower threshold because soft
replanning does not pollute memory. A memory write must always trigger one hard replan.

Online replay showed that naive per-frame threshold triggering fires too often on rising score
plateaus. The selected runtime therefore uses causal one-step peak confirmation before applying
the thresholds.

Runtime integrations should instantiate one `TransitionTriggerOnlineSession` per rollout when they
own raw action/state blocks, or one `TransitionTriggerSession` when they already own `[32, 144]`
windows. Stateless single-window decisions are intentionally not used for the selected
`causal_peak` policy.

Final online replay metrics:

| Split | Trigger | F1 | Precision | Recall | Early Rate | Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| eval | all_plan | 0.702 | 0.812 | 0.619 | 0.040 | 340 |
| eval | memory_write | 0.551 | 0.977 | 0.383 | 0.007 | 175 |
| test | all_plan | 0.685 | 0.801 | 0.598 | 0.048 | 327 |
| test | memory_write | 0.532 | 0.982 | 0.365 | 0.007 | 163 |

There were zero `memory_write => hard_plan` violations on both eval and test replay.

## Closed-Loop Trace Smoke

After adding per-decision LIBERO trace logging, a one-task smoke run was completed to verify the
runtime observability path.

```text
run:
$AUTODL_TMP/HiMem-Bridge-VLA/runs/transition_trigger/libero_closed_loop_trace_smoke/20260617_212202_trace_jsonl_smoke

trace:
runs/transition_trigger/libero_closed_loop_trace_smoke/20260617_212202_trace_jsonl_smoke/client/results/transition_trace_smoke_transition_trace.jsonl

client_status: 0
task suite: libero_spatial
episodes: 1
decision steps: 35
control steps: 35
task result: fail, max_steps_exhausted
trace rows: 35
ready/scored rows: 3
score range: 0.5269 to 0.5480
soft_plan/hard_plan/memory_write triggers: 0 / 0 / 0
```

This run validates that the server/client transition path and JSONL trace are working. It should
not be used as a closed-loop performance result because the current policy has not been trained or
tuned for LIBERO success.

## Paper Figure Recommendation

Use one small temporal-context ablation figure.

```text
metric name in paper: Transition-F1 (%)
definition: event-level F1 under causal post-boundary matching, shown as percentage points
x-axis: temporal window size [24, 32, 48]
y-axis: Transition-F1 (%)
values: [46.5, 47.9, 36.8]
optional annotation: W32 is +30.1% relative to W48
```

This single plot is enough for the module ablation: W32 gives the best transition-trigger quality,
while W48 drops clearly despite using more history. The intended message is simply:

```text
Medium temporal context is best; longer history is not automatically better.
```

Keep memory-write/planner-specific numbers in the text or appendix as implementation details, not
as the main ablation figure.

## Result Files

```text
summary table:
$AUTODL_TMP/runs/transition_trigger/robomme_rmbench_ablations/ablation_summary.tsv

selected runtime package:
$AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512

selected package manifest:
$AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512/manifest.json

online replay outputs:
$AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512/runtime_policy_eval.json
$AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512/runtime_policy_test.json

per-run training/evaluation logs:
$AUTODL_TMP/runs/transition_trigger/robomme_rmbench_ablations/<run_name>/train.log
$AUTODL_TMP/runs/transition_trigger/robomme_rmbench_ablations/<run_name>/train_history.json
$AUTODL_TMP/runs/transition_trigger/robomme_rmbench_ablations/<run_name>/test_metrics_*.json
```
