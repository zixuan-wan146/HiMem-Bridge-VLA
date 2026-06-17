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

Completed ablation results and the selected runtime model live in:

```text
docs/transition_trigger_ablation_results.md
transition_trigger/configs/selected/robomme_rmbench_w32_value_delta_transformer_d512.yaml
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

Current selected model:

```text
feature_set: value_delta_mask
window_size: 32
model.type: transformer
model.d_model: 512
checkpoint type: best_memory_write
runtime score_mode: causal_peak
planner_threshold: 0.70
memory_write_threshold: 0.80
```

Runtime loading:

```python
from transition_trigger.runtime import load_selected_trigger

trigger = load_selected_trigger(device="cuda")
online = trigger.new_online_session(dataset_name="robomme_four_tasks")

for frame_idx, frame in rollout_frames:
    # frame contains the raw canonical blocks used by the selected config,
    # e.g. action/eef_state/joint_state/gripper_state for RoboMME.
    output = online.append(frame, frame_index=frame_idx)
    if output is None:
        continue

    if output.decision.memory_write:
        write_memory()
        hard_plan()
    elif output.decision.soft_plan:
        soft_plan()
```

For the selected `causal_peak` runtime policy, use `trigger.score_window(...)`
for pure batched scoring, `trigger.new_session().decide_window(...)` when the
caller already owns `[32, 144]` windows, and `trigger.new_online_session(...)`
when the caller owns raw action/state blocks. The stateless
`trigger.decide_window(...)` path is only valid for `score_mode: threshold`.

Online replay evaluation:

```bash
python transition_trigger/scripts/evaluate_runtime_policy.py \
  --package-dir /root/autodl-tmp/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512 \
  --split test \
  --device cuda
```

Optional HiMem server integration:

```bash
python scripts/himem_server.py \
  --ckpt_dir checkpoints/HiMem_LIBERO \
  --transition_trigger_package /root/autodl-tmp/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512 \
  --transition_dataset_name robomme_four_tasks
```

Requests should include a stable `episode_id` or `session_id` and a `transition_frame`
containing the raw canonical keys for the selected schema. With `return_debug: true`,
the response is an object with `actions` and `transition_trigger`; otherwise it remains
the legacy action list. The LIBERO action parser accepts both formats. `transition_trigger`
contains `ready`, `score`, `memory_write`, `soft_plan`, and `hard_plan`. When
`transition_frame` is present, the server routes HiMem memory writes through the
transition-trigger decision: memory is accumulated while the trigger is not ready or
below threshold, and committed only when `memory_write` is true.

For chunked action clients such as LIBERO, `HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT`
controls whether a transition-trigger event shortens the returned action chunk:

```bash
HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT=0  # default, execute the full chunk
HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT=1  # execute one action, then request a fresh plan
```

LIBERO can optionally send a RoboMME-like transition frame built from the previous executed
action and the current observation:

```bash
HIMEM_LIBERO_TRANSITION_DATASET_NAME=robomme_four_tasks
HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT=1
```

This adapter maps LIBERO observations to the selected single-arm schema:
`action`, `eef_state`, `joint_state`, and `gripper_state`. It is useful for closed-loop
smoke tests, but should be treated as a cross-benchmark approximation until validated by
success-rate and step-count comparisons. Both switches are recorded in the LIBERO run manifest.

When a transition dataset is enabled, the LIBERO client writes a per-decision JSONL trace by
default:

```bash
HIMEM_LIBERO_TRANSITION_TRACE_FILE=<result-dir>/<ckpt-name>_transition_trace.jsonl
```

Each row records the task/episode/step, whether a `transition_frame` was sent, the returned
`score`, `ready`, `soft_plan`, `hard_plan`, `memory_write`, `should_plan`, and whether the action
chunk was shortened by `HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT`. Override
`HIMEM_LIBERO_TRANSITION_TRACE_FILE` to choose a different path.

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
