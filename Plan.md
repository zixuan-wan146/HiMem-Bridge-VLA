# HiMem-Bridge-VLA Current Plan

This checkout is the active project for BridgeAttention + H32 single-token coarse planner work. Older transition-trigger and H64 suffix-planner designs have been retired from the active roadmap.

## Active Goal

Promote the H32 standalone planner into BridgeAttention / ActionHead training without reintroducing cached suffix queues or transition-trigger refresh logic.

```text
H_t, s_t -> CoarsePlanner -> one plan token P_t -> BridgeAttention -> FlowMatchingActionHead -> 32-step action chunk
```

## Current State

- H32 feature cache is built on the remote data disk.
- H32 action-only intent AE is trained.
- H32 single-token planner is trained to the current best checkpoint.
- Old H64 planner configs/checkpoints and transition-trigger active code paths have been removed.
- Memory work is now a separate Dual-FIFO visual-memory track. The active step is memory-side inference construction, not BridgeAttention integration.

## Active Entry Points

```text
docs/current_project_state.md   detailed state, artifacts, metrics, next work
docs/engineering_reproducibility.md reproducibility and engineering gates
docs/benchmark_plan.md          LIBERO / LIBERO-Plus / RMBench status and next work
docs/coarse_planner_design.md   H32 planner target and training contract
docs/bridge_himem_design.md     BridgeAttention integration contract
docs/project_structure.md       code/config/docs/output boundaries
coarse_planner/README.md        standalone cache, AE, and planner commands
```

## Next Engineering Work

1. Keep the current H32 best planner checkpoint as the standalone planner baseline.
2. Update BridgeAttention / ActionHead training data to consume one H32 plan token.
3. Train the joint BridgeAttention / ActionHead path with the planner token enabled.
4. Evaluate against fused-only and bridge-clean baselines.
5. Keep BridgeAttention memory integration separate until the memory-side inference path is tested.

## Guardrails

- Do not restart transition-trigger work for this H32 path.
- Do not reintroduce PlanTokenQueue, consumed-step suffix state, or cached plan refresh policy.
- Do not wire memory into BridgeAttention before the memory-side inference path is tested.
- Keep large datasets, caches, checkpoints, and run outputs off the system disk.
