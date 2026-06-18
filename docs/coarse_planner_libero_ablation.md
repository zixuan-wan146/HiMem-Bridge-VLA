# Coarse Planner LIBERO Horizon Ablation

This note records the current Coarse Planner design used for the LIBERO-only warm-up experiments.

## Module Boundary

The standalone planner input is:

```text
Z_t = InternVL3_14(observation_t, instruction)
S_t = ProprioProjector(state_t)
P_t, coarse_actions_t = CoarsePlanner(Q_plan, [Z_t, S_t])
```

`Z_t` contains observation and language. It does not contain robot proprioception. The state path is a two-layer projector:

```text
LayerNorm(state_dim)
Linear(state_dim, 896)
GELU
Linear(896, 896)
```

The planner does not read memory. Memory is a parallel condition later in BridgeAttention:

```text
C_bridge = [Q_action, S_t, P_t, M_t]
```

## LIBERO State and Action Targets

LIBERO hdf5 demonstrations expose several state fields. For Evo-1 compatibility we use:

```text
state_raw = concat(obs/ee_states, obs/gripper_states)  # 8 dims
state_norm = minmax_normalize(state_raw, Evo1_LIBERO/norm_stats.json)
state = pad_to_24(state_norm)
```

Actions are 7D. Motion dimensions are kept in the raw relative action convention. The gripper dimension is converted from LIBERO env convention to model convention:

```text
env -1 -> model 1
env  1 -> model 0
```

Coarse targets are built per chunk:

```text
motion_i = sum(actions[t+i*8 : t+(i+1)*8, 0:6])
gripper_i = actions[t+(i+1)*8-1, 6]
```

## Horizon Sweep

The boundary statistics from previous transition-trigger datasets center around 50-65 control steps, while LIBERO-Spatial/Goal/Object are short enough that 128-step targets cause heavy tail padding. The first sweep therefore fixes `chunk_size=8` and tests:

```text
H=32, K=4
H=48, K=6
H=64, K=8
```

All three horizons share the same sampled `(suite, task, demo, timestep)` index. The build config currently samples up to 2048 points from LIBERO with `require_full_max_horizon=true`, so no target uses tail padding in the first ablation.

## Commands

Dry run:

```bash
/root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m coarse_planner.build_from_libero \
  --config coarse_planner/configs/libero_horizon_ablation_build.yaml \
  --dry-run
```

Build caches:

```bash
/root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m coarse_planner.build_from_libero \
  --config coarse_planner/configs/libero_horizon_ablation_build.yaml \
  --device cuda
```

Train and analyze:

```bash
COARSE_PLANNER_BATCH_SIZE=512 \
COARSE_PLANNER_EPOCHS=8 \
scripts/run_coarse_planner_libero_ablation.sh
```

Outputs:

```text
/root/autodl-tmp/datasets/coarse_planner/libero_h32
/root/autodl-tmp/datasets/coarse_planner/libero_h48
/root/autodl-tmp/datasets/coarse_planner/libero_h64

/root/autodl-tmp/runs/coarse_planner/libero_h32
/root/autodl-tmp/runs/coarse_planner/libero_h48
/root/autodl-tmp/runs/coarse_planner/libero_h64

/root/autodl-tmp/runs/coarse_planner/libero_horizon_ablation_report.md
```
