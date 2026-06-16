# Motion-State BoundaryHead

This folder is a benchmark-agnostic workspace for a standalone motion/state BoundaryHead.

The first concrete benchmark target is RoboMME because its H5 format exposes native subgoal
boundary supervision through `info/is_subgoal_boundary`. RMBench is the first memory-dependent
benchmark adapter because its `language_annotation.json` files provide subtask durations that can
be converted into internal boundary events.

Current RoboMME data/download notes live in:

```text
motion_boundary/robomme_data_plan.md
```

Current planning notes for the LIBERO/RMBench/RoboMME scope live in:

```text
docs/eval_memory_boundary_plan_2026-06-17.md
```

## Module Question

The module should answer one narrow question:

```text
Given motion/state history, is the current intent or subtask reaching a boundary?
```

It outputs one scalar boundary score:

```text
b_t = P(boundary at t | motion/state history)
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
python scripts/convert_robomme_h5_to_motion_boundary.py \
  --dataset StopCube=/path/to/record_dataset_StopCube.h5 \
  --output-root /root/autodl-tmp/datasets/robomme_motion_boundary/example
```

RMBench download and conversion:

```bash
python scripts/download_rmbench_tasks.py \
  --local-dir /root/autodl-tmp/benchmarks/RMBench

python scripts/convert_rmbench_to_motion_boundary.py \
  --input-root /root/autodl-tmp/benchmarks/RMBench \
  --output-root /root/autodl-tmp/datasets/rmbench_motion_boundary/nine_tasks
```

RMBench flat-vector smoke features use:

```text
action: joint_action/vector, 14 dims
state:  left_endpose + right_endpose + left_gripper + right_gripper, 16 dims
label:  cumulative subtask duration boundaries from language_annotation.json
```

The corresponding config is:

```text
motion_boundary/configs/rmbench_9tasks.yaml
```

## Model Candidates

Keep the causal TCN as the baseline, but the main long-history candidate should be an SSM-style
BoundaryHead inspired by DualTreeVLA's JumpAwareHead.

Comparison matrix:

```text
B0: Causal TCN baseline
B1: causal SSM BoundaryHead over motion/state history
B2: small TCN stem + causal SSM BoundaryHead
```

Important difference from action-jump detectors:

```text
BoundaryHead input is action + state + deltas, not action-only.
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
tau_p: replan threshold, recall-oriented
tau_w: memory-write threshold, precision-oriented
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
