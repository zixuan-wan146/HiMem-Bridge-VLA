# Evaluation, Memory, And Boundary Planning Notes

Date: 2026-06-17

This note records the current working decisions from the recent planning discussions. It is not a
final architecture spec. The memory mechanism, high-level planner, and final motion-boundary design
are still open research decisions.

## Evaluation Scope

The benchmark scope should stay small enough to execute under the current compute and time budget.

Current recommendation:

- Use LIBERO Goal, Spatial, and Object for short-horizon memory-free manipulation.
- Use LIBERO Long for long-horizon memory-free manipulation.
- Use LIBERO-Plus as a direct zero-shot robustness evaluation when available.
- Use RMBench for memory-dependent / POMDP evaluation.
- Do not include full RoboTwin 2.0 in the main evaluation suite for now.
- Do not include RobotWin as a main benchmark unless a later ablation specifically needs it.

Rationale:

- LIBERO cleanly covers the memory-free MDP side of the short/long axis.
- RMBench is explicitly designed for memory-dependent manipulation and is a better fit for the
  POMDP/memory axis.
- Full RoboTwin 2.0 has 50 bimanual tasks and is broad enough to become its own project. It is useful
  as a bimanual and robustness stress test, but it is too expensive for the main scope.
- RoboMME has useful boundary annotations, but several suites use video-demo conditioning and are not
  a good fit for a non-video-conditioned VLA policy.

## RMBench Task Split

RMBench is the planned memory-dependent benchmark. The paper describes nine tasks split by Task
Memory Complexity:

M(1) tasks:

- `observe_and_pickup`
- `rearrange_blocks`
- `put_back_block`
- `swap_blocks`
- `swap_T`

M(n) tasks:

- `blocks_ranking_try`
- `press_button`
- `cover_blocks`
- `battery_try`

Working horizon interpretation:

- Short / medium memory-dependent: `observe_and_pickup`, `put_back_block`, `swap_T`
- Long memory-dependent: `rearrange_blocks`, `swap_blocks`, `cover_blocks`,
  `blocks_ranking_try`, `press_button`, `battery_try`

The RMBench repository contains additional artifacts such as `place_block_mat`,
`classify_blocks`, `storage_blocks`, and `place_object_box`. These should not be treated as the
paper's nine-task benchmark unless we explicitly expand the benchmark later.

## RoboMME Status

RoboMME is useful for the first motion-boundary training loop because the H5 files expose native
boundary labels:

- `info/is_subgoal_boundary`
- `info/is_completed`
- online subgoal text fields

The four locally prepared RoboMME tasks are:

- `StopCube`
- `VideoUnmask`
- `PickHighlight`
- `PatternLock`

They should be treated as boundary-supervision data, not as the main VLA policy dataset. Some
RoboMME suites include `video-demo` frames where the robot first observes a video demonstration before
acting. These frames are not appropriate as direct inputs for a non-video-conditioned VLA policy and
should be filtered out for motion-boundary training unless explicitly studying video-conditioned
policies.

Converted RoboMME motion-boundary data currently keeps action, state, frame index, episode id, task,
and boundary sidecars. The scalar label is whether a frame is near an annotated subgoal boundary.

## Motion Boundary Contract

The motion-boundary module should remain narrow:

```text
input:  motion/state history window
output: scalar boundary logit
```

It should not decide the next high-level task and should not directly implement the planner.

For a single embodiment with compatible dimensions, the current practical input is:

```text
[action, state, delta_action, optional_delta_state, gripper_transition]
```

For cross-embodiment training, fixed flat dimensions are the wrong abstraction. The better long-term
direction is:

```text
raw benchmark action/state
  -> benchmark adapter
  -> canonical per-arm motion tokens
  -> shared BoundaryHead
  -> boundary logit
```

Canonical tokenization should include end-effector translation/rotation motion, gripper state or
action, gripper transition, and arm activity masks. Single-arm datasets can fill one active arm and
mask the other.

## RMBench Data Adapter Requirements

The RMBench raw HDF5 files expose dual-arm proprioceptive information. The known useful fields are:

- `/joint_action/left_arm`
- `/joint_action/left_gripper`
- `/joint_action/right_arm`
- `/joint_action/right_gripper`
- `/joint_action/vector`
- `/endpose/left_endpose`
- `/endpose/right_endpose`
- related gripper/endpose fields

The Mem-0 LeRobot conversion path constructs a 16-dimensional robot state from dual-arm joints plus
grippers. For our boundary module, the first adapter should preserve enough information to support
both flat-vector smoke tests and later canonical per-arm tokenization.

RMBench does not provide the same direct `is_subgoal_boundary` label as RoboMME. For smoke testing,
use available episode/subtask metadata if present. If metadata is unavailable, create a clearly marked
heuristic boundary sidecar from trajectory structure only, and do not report it as oracle supervision.

## Current RMBench Remote Artifacts

Remote paths:

```text
/root/autodl-tmp/benchmarks/RMBench
/root/autodl-tmp/benchmarks/RMBench/data/rmbench_9tasks_manifest.json
/root/autodl-tmp/datasets/rmbench_motion_boundary/nine_tasks
/root/autodl-tmp/runs/motion_boundary/rmbench_9tasks_smoke
```

Downloaded data status:

- 9 official RMBench tasks.
- 50 HDF5 episodes per task.
- 50 trajectory pkl files per task.
- 50 instruction JSON files per task.
- 50 video files per task.
- 1 `language_annotation.json` per task.
- 1,809 files in the download manifest.
- Raw RMBench `data/` directory size: about 24 GB.

Converted motion-boundary status:

- 450 episodes.
- 277,350 frames.
- 3,094 internal boundary events.
- 450 terminal events tracked separately.
- 0 duration/frame-count mismatches.
- Converted parquet/annotation directory size: about 67 MB.
- Full dataset build smoke: 237,257 train windows, 26,143 validation windows.
- Flat-vector input shape: `(window=32, dim=44)`.
- One-epoch CPU smoke train completed and wrote checkpoints under
  `/root/autodl-tmp/runs/motion_boundary/rmbench_9tasks_smoke`.

## Open Design Questions

- Whether the final BoundaryHead should be a TCN, SSM, or hybrid TCN+SSM model.
- How memory-write thresholds and replan thresholds should differ.
- Whether high-level planning should be event-triggered only or periodically refreshed.
- How to combine key-frame memory, sliding memory, and anchor memory for the final VLA policy.
- Whether boundary supervision should be trained on raw motion/state vectors or canonical per-arm
  motion tokens from the beginning.
