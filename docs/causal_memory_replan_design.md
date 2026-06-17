# Causal Memory Write And Replan Design

Status: working design. This records the current direction for the transition detector, memory write, and planner trigger policy.

Latest finished ablation results and the selected runtime checkpoint are recorded in:

```text
docs/transition_trigger_ablation_results.md
```

## Core Position

The transition trigger should not be trained or used as a boundary-frame oracle. A model that fires exactly on the annotated boundary frame is too close to memorizing dataset timing and is not the causal mechanism we want.

The module should instead answer this question:

```text
Given motion/state history up to now, has a relevant transition already happened recently?
```

This makes the module a causal post-boundary transition detector. It detects evidence after the environment has changed, rather than predicting the future or matching one offline annotation frame exactly.

## Trigger Semantics

There are two planner-related actions with different costs and thresholds:

```text
soft plan / soft replan:
  Low-threshold control adjustment inside the current subtask.
  It does not write memory.

memory write + hard plan:
  High-threshold semantic commit that records the completed segment.
  Every memory write must trigger one hard plan/replan.
```

The implication is one-way:

```text
memory_write => hard_plan
soft_plan    does not imply memory_write
```

Online policy:

```text
execute action
observe new state
update motion/state history
output = transition_trigger_session.decide_window(history, frame_index=t)

if output.decision.memory_write:
    write_memory(completed_segment)
    hard_plan()
elif output.decision.soft_plan:
    soft_plan()
else:
    continue_current_plan()
```

Thresholds:

```text
tau_w: memory-write threshold, high precision, initially fixed at 0.8
tau_p: planner threshold, lower than tau_w, swept over values such as 0.2 to 0.7
```

The planner is therefore not blocked by memory. It can replan at lower confidence for control robustness. Memory write is stricter because a bad memory entry can pollute later planning.

Current selected runtime policy:

```text
score_mode = causal_peak
planner_threshold = 0.70
memory_write_threshold = 0.80
replan_cooldown_frames = 10
memory_write_cooldown_frames = 10
memory_write_implies_plan = true
```

The online policy uses separate cooldowns. A recent soft plan suppresses repeated soft plans, but
it must not suppress a later high-confidence memory write. Memory writes are throttled only by the
memory-write cooldown and always emit a hard plan.

`causal_peak` confirms a local score peak with one-frame delay before applying the thresholds.
This stays causal and avoids repeatedly firing on rising score plateaus. The online integration
must therefore keep a stateful session per rollout: `TransitionTriggerOnlineSession` for raw
action/state blocks, or `TransitionTriggerSession` for prebuilt windows. It must not call a stateless
single-window threshold decision.

Current selected runtime model:

```text
feature_set = value_delta_mask
window_size = 32
model.type = transformer
model.d_model = 512
checkpoint = best_memory_write
```

## Runtime Integration Contract

The selected trigger is usable as a standalone runtime package, but it should not be wired into the
HiMem server by feeding arbitrary normalized model inputs. Its trained input is a 32-frame canonical
action/state history with value, delta, and missing-value mask blocks. The integration must therefore
own a feature adapter with the same block layout and scale as the training data.

Minimum online contract:

```text
1. Keep one TransitionTriggerOnlineSession per session_id:episode_id and dataset/embodiment.
2. After executing an action and observing the next state, append that action/state pair to the
   canonical transition history through online.append(frame, frame_index=t).
3. The online session returns None until 32 frames are available, then builds the exact
   value_delta_mask window used in training.
4. If decision.memory_write is true, commit the accumulated memory segment and force a hard plan.
5. Else if decision.soft_plan is true, replan without writing memory.
6. Otherwise continue the current plan while still accumulating the segment.
```

The current HiMem model still has an internal bridge-token boundary gate for memory writes. That gate
is not equivalent to the selected transition trigger. The server integration therefore uses an
explicit `transition_frame` request field rather than guessing features from normalized inference
state. When that field is present and the transition trigger package is enabled, memory commits are
routed through the transition trigger decision, with `memory_write => hard_plan` preserved as an
invariant. Requests without `transition_frame` keep the legacy bridge-boundary memory behavior.

## Time Definitions

Use three distinct times:

```text
B: offline annotated boundary or segment transition
C: observable completion evidence in state/history
T: online trigger time chosen by the model and threshold
```

The target is not `T == B`. The causal target is:

```text
T >= B
```

In practice we train and evaluate with a post-boundary acceptance window:

```text
positive examples: B + d_min through B + d_max
ignored examples:  frames before B and the ambiguous boundary frame itself
```

Initial defaults:

```text
d_min = 1
d_max = 5
pre-boundary ignore = 6 frames
boundary frame is ignored, not positive
```

## Data Scope

Main training data:

```text
RoboMME:
  Uses native info/is_subgoal_boundary as the primary subgoal transition label.
  Uses info/is_completed as optional terminal evidence, not the main soft-plan signal.

RMBench:
  Uses language_annotation duration segments to build semantic subtask transitions.
  Useful for long memory-dependent episodes.
```

CALVIN is excluded from the main transition training path for now. Its labels are mostly short episode terminal completion labels, not the kind of internal subskill boundaries needed for the main module. It can remain a weak sanity check later, but it should not define the module objective.

## Current Local Dataset Facts

RoboMME converted four-task subset:

```text
root: /root/autodl-tmp/datasets/robomme_transition_trigger/four_tasks
frames: 91633
episodes: 400
subgoal transition events: 1701
label source: robomme/info/is_subgoal_boundary
```

Converted RoboMME feature fields:

```text
action = eef_action                         7 dims
state  = eef_state + joint_state + gripper_state + is_gripper_close
       = 6 + 7 + 2 + 1 = 16 dims
current flat feature config with deltas = 7 + 16 + 7 + 16 + 1 = 47 dims
```

RMBench nine-task subset:

```text
root: /root/autodl-tmp/datasets/rmbench_transition_trigger/nine_tasks
source benchmark files: 9 tasks, 50 episodes per task
boundary source: rmbench/language_annotation/duration
```

Converted RMBench smoke features:

```text
action = joint_action/vector                                      14 dims
state  = left_endpose + right_endpose + left_gripper + right_gripper
       = 7 + 7 + 1 + 1 = 16 dims
```

## Cross-Embodiment Input Design

The long-term input should be a masked canonical motion representation, not raw benchmark-specific vectors.

Canonical structure:

```text
left_arm_token
right_arm_token
global/source token, optional
field-level valid masks
dataset/source embedding, optional
```

Per-arm candidate fields:

```text
eef pose
delta eef pose
joint state
delta joint state
gripper state
gripper transition
action
delta action
valid mask per field
```

Missing fields are represented as:

```text
value = 0
valid_mask = 0
```

This allows single-arm RoboMME, dual-arm RMBench, and future embodiments to share one TransitionTrigger/TransitionTriggerHead without pretending that missing values are real zeros.

## Training Objective

Use causal post-boundary labels for the main experiments:

```yaml
data:
  label_mode: causal_post
  positive_min_delay: 1
  positive_max_delay: 5
  ignore_min_delay: -6
  ignore_max_delay: 0
```

This means:

```text
B - 6 through B: ignored
B + 1 through B + 5: positive
outside that window: negative or hard negative by distance
```

Do not use symmetric labels for the main memory-write detector because they reward early firing.

## Evaluation

Event matching should be post-boundary:

```text
valid trigger: B + match_min_delay through B + match_max_delay
early trigger: before the valid post-boundary window
```

Initial evaluation settings:

```yaml
evaluation:
  match_min_delay: 1
  match_max_delay: 5
  early_tolerance: 6
  memory_write_fixed_threshold: 0.8
  threshold_grid: [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
```

Metrics to report:

```text
AUPRC
memory-write precision/recall at tau_w = 0.8
soft-plan precision/recall across tau_p sweeps
early trigger count and rate
mean trigger delay
duplicate triggers per event
triggers per 100 scored frames
```

## Experiment Plan

1. Train the memory-write detector on RoboMME + RMBench with causal post-boundary labels.
2. Fix `tau_w = 0.8` for memory write and hard plan.
3. Sweep lower `tau_p` values for soft planning.
4. Compare RoboMME-only, RMBench-only, and mixed training.
5. Compare raw flat features against the masked canonical adapter.
6. Keep CALVIN out of the main result table unless explicitly labeled as weak terminal-completion supervision.

## Naming

The module has been renamed to `transition_trigger`. Code and documentation should use the precise concept names:

```text
TransitionTrigger
memory_write
soft_plan
hard_plan
causal_post labels
```
