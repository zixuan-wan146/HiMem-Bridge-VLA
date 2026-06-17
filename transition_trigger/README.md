# Causal TransitionTrigger

This folder is a benchmark-agnostic workspace for a standalone motion/state TransitionTrigger.

The first concrete benchmark target is RoboMME because its H5 format exposes native subgoal
transition supervision through `info/is_subgoal_boundary`. RMBench is the first memory-dependent
benchmark adapter because its `language_annotation.json` files provide subtask durations that can
be converted into internal transition events.

Current RoboMME data/download notes live in:

```text
transition_trigger/robomme_data_plan.md
```

Current causal memory-write and replan design notes live in:

```text
docs/causal_memory_replan_design.md
```

## Module Question

The module should answer one narrow question:

```text
Given motion/state history, has a relevant transition already happened recently?
```

It outputs one scalar transition score:

```text
b_t = P(recent transition has occurred | motion/state history up to t)
```

It does not predict the next task, does not generate a skill latent, and does not implement a
planner.

## Input Contract

The input should remain motion/state based, not action-only:

```text
[A, S, delta_A, delta_S, delta_gripper]
```

Where:

```text
A: robot action
S: robot proprioceptive state
delta_A: action difference
delta_S: state difference
delta_gripper: gripper transition feature
```

## Desired Boundary Sidecar

The eventual benchmark adapter should export a sidecar with one row per segment:

```json
{
  "episode_id": "episode_000123",
  "task": "task_name",
  "subtask_id": 2,
  "subtask_name": "optional subtask name",
  "start": 120,
  "end": 183,
  "is_terminal": false,
  "label_source": "annotation_or_oracle"
}
```

Internal subtask boundaries and final episode termination should be evaluated separately:

```text
internal boundary: end(segment_i), i < last segment
terminal event:    end(last segment)
```

## Dataset Adapters

RoboMME conversion:

```bash
python scripts/convert_robomme_h5_to_transition_trigger.py \
  --dataset StopCube=/path/to/record_dataset_StopCube.h5 \
  --output-root /root/autodl-tmp/datasets/robomme_transition_trigger/example
```

RMBench download and conversion:

```bash
python scripts/download_rmbench_tasks.py \
  --local-dir /root/autodl-tmp/benchmarks/RMBench

python scripts/convert_rmbench_to_transition_trigger.py \
  --input-root /root/autodl-tmp/benchmarks/RMBench \
  --output-root /root/autodl-tmp/datasets/rmbench_transition_trigger/nine_tasks
```

RMBench flat-vector smoke features use:

```text
action: joint_action/vector, 14 dims
state:  left_endpose + right_endpose + left_gripper + right_gripper, 16 dims
label:  cumulative subtask duration boundaries from language_annotation.json
```

The corresponding config is:

```text
transition_trigger/configs/rmbench_9tasks.yaml
```

## Model Candidates

Use causal sequence models only. The current comparison is between a pure state-space
TransitionTriggerHead and a causal Transformer over the same action/state history.

Comparison matrix:

```text
window_size: [16, 24, 32, 48]
feature_set: [value_mask, value_delta_mask, value_mask_domain, full]
model.type: [ssm, transformer]
model.d_model: [256, 512]
```

Important difference from action-jump detectors:

```text
TransitionTriggerHead input is canonical action/state blocks plus explicit missing-field masks.
```

## Evaluation

Frame-level:

```text
AUPRC
```

Event-level:

```text
precision
recall
F1
mean/median trigger delay
duplicate triggers per event
```

Use two thresholds:

```text
tau_p: soft-plan threshold, recall-oriented
tau_w: memory-write threshold, precision-oriented; memory_write implies hard_plan
tau_p < tau_w
```

## Data Audit Required Before Training

Before adding benchmark-specific code, answer:

```text
1. What is the native storage format?
2. Where are actions and robot states stored?
3. Are gripper values explicit?
4. Does the benchmark provide subtask/stage annotations?
5. If not, is there an oracle or success predicate per subtask?
6. What are the episode length and subtask-count distributions?
7. Which subset has medium-horizon episodes suitable for first experiments?
```

Only after this audit should we add:

```text
configs/datasets/<benchmark>.yaml
configs/training/<benchmark>_*.yaml
benchmark-specific dataset reader or conversion script
```
