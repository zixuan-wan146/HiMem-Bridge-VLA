# Project Cleanup Plan

Date: 2026-06-17

Goal: turn the current transition-trigger working tree into readable, reviewable commits without
losing experiment context.

## Ground Rules

- Do not delete run outputs or remote artifacts during cleanup.
- Do not reset or checkout files to hide dirty state.
- Do not mix unrelated files into one commit just because they were edited on the same day.
- Do not claim LIBERO success-rate improvement from the current closed-loop runs.
- Keep large artifacts outside git. Runtime packages and model caches stay under `$AUTODL_TMP/`.

## Current State Snapshot

```text
repo head: f1cd2ec
local repo: <local-repo>/HiMem-Bridge-VLA
remote repo: $AUTODL_TMP/HiMem-Bridge-VLA
remote selected package:
$AUTODL_TMP/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512
```

The current dirty tree is expected. It contains one coherent feature line plus documentation:

```text
transition trigger runtime
HiMem server integration
LIBERO transition-frame and trace logging
experiment/result documentation
```

## Commit Split

### Commit 1: Transition Trigger Runtime

Candidate paths:

```bash
git add \
  transition_trigger/__init__.py \
  transition_trigger/config.py \
  transition_trigger/trigger_policy.py \
  transition_trigger/online_features.py \
  transition_trigger/runtime.py \
  transition_trigger/configs/selected \
  transition_trigger/scripts/evaluate_runtime_policy.py \
  tests/test_transition_trigger_causal.py
```

Verification:

```bash
python -m pytest -q tests/test_transition_trigger_causal.py
python -m py_compile \
  transition_trigger/trigger_policy.py \
  transition_trigger/online_features.py \
  transition_trigger/runtime.py \
  transition_trigger/scripts/evaluate_runtime_policy.py
```

### Commit 2: HiMem Server Transition-Trigger Integration

Candidate paths:

```bash
git add \
  himem_bridge_vla/server_protocol.py \
  himem_bridge_vla/transition_trigger_manager.py \
  himem_bridge_vla/model/himem_bridge_vla.py \
  scripts/himem_server.py \
  tests/test_server_protocol.py \
  tests/test_transition_trigger_manager.py \
  tests/test_himem_server_transition_trigger.py
```

Verification:

```bash
python -m pytest -q \
  tests/test_server_protocol.py \
  tests/test_transition_trigger_manager.py \
  tests/test_himem_server_transition_trigger.py
```

### Commit 3: LIBERO Transition Frame And Trace Logging

Candidate paths:

```bash
git add \
  evaluations/libero/libero_action_protocol.py \
  evaluations/libero/libero_client_4tasks.py \
  evaluations/libero/libero_client_config.py \
  evaluations/libero/libero_transition_frame.py \
  evaluations/libero/libero_transition_trace.py \
  evaluations/libero/libero_transition_trace_summary.py \
  scripts/libero_profile.sh \
  scripts/run_libero_eval.sh \
  scripts/run_libero_smoke.sh \
  scripts/summarize_libero_transition_trace.py \
  scripts/write_libero_run_manifest.py \
  tests/test_libero_action_protocol.py \
  tests/test_libero_client_config.py \
  tests/test_libero_transition_frame.py \
  tests/test_libero_transition_trace.py \
  tests/test_libero_transition_trace_summary.py \
  tests/test_run_libero_eval_script.py \
  tests/test_run_libero_smoke_script.py \
  tests/test_write_libero_run_manifest.py
```

Verification:

```bash
python -m pytest -q \
  tests/test_libero_action_protocol.py \
  tests/test_libero_client_config.py \
  tests/test_libero_transition_frame.py \
  tests/test_libero_transition_trace.py \
  tests/test_libero_transition_trace_summary.py \
  tests/test_run_libero_eval_script.py \
  tests/test_run_libero_smoke_script.py \
  tests/test_write_libero_run_manifest.py
```

### Commit 4: Experiment Documentation

Candidate paths:

```bash
git add \
  docs/causal_memory_replan_design.md \
  docs/transition_trigger_ablation_results.md \
  docs/current_project_state.md \
  docs/project_structure.md \
  transition_trigger/README.md \
  to-do/6-17.md \
  to-do/project_cleanup_plan.md
```

Verification:

```bash
python scripts/summarize_libero_transition_trace.py \
  runs/transition_trigger/libero_closed_loop_trace_eval/20260617_214140_long_trace_eval/client/results/transition_trigger_trace_eval_transition_trace.jsonl \
  --format text
```

### Commit 5: Repository Hygiene

Candidate paths:

```bash
git add .gitignore
```

Purpose:

```text
Ignore generated run outputs.
```

## Recommended Order

1. Commit repository hygiene first if `runs/` is currently noisy in local status.
2. Commit runtime code.
3. Commit server integration.
4. Commit LIBERO adapter/trace support.
5. Commit docs and todo records last.

If a test fails during cleanup, stop and fix only within that commit group.

## Remaining Research Decision

The runtime policy is conservative in LIBERO:

```text
max LIBERO trace score: 0.6934
planner_threshold: 0.70
```

Choose one path before presenting LIBERO:

```text
1. Keep 0.70 and describe LIBERO as integration/observability.
2. Add an explicit LIBERO/demo threshold, e.g. 0.65, for visualizing replans.
3. Calibrate or train on LIBERO-like traces before making LIBERO behavior claims.
```
