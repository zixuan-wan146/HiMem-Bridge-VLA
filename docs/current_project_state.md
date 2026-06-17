# Current Project State

Date: 2026-06-17

## Repository Locations

```text
local:  /home/myser/Project/Evo/HiMem-Bridge-VLA
remote: /root/autodl-tmp/HiMem-Bridge-VLA
head:   f1cd2ec
```

The local and remote working trees have been synced. The repository is intentionally dirty because
the transition-trigger integration has not yet been split into commits.

Remote runtime/data/model artifacts live on the data disk:

```text
/root/autodl-tmp/runs/transition_trigger
/root/autodl-tmp/datasets
/root/autodl-tmp/checkpoints
/root/autodl-tmp/hf-home
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

## Selected Runtime Model

```text
package:
/root/autodl-tmp/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512

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
/root/autodl-tmp/hf-home/hub/models--OpenGVLab--InternVL3-1B
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
/root/autodl-tmp/HiMem-Bridge-VLA/runs/transition_trigger/libero_closed_loop_trace_smoke/20260617_212202_trace_jsonl_smoke

result: client_status 0, 1 episode, 35/35 steps, max_steps_exhausted
trace: 35 rows, 3 scored rows, max score 0.5480, 0 triggers
```

80-step trace eval:

```text
run:
/root/autodl-tmp/HiMem-Bridge-VLA/runs/transition_trigger/libero_closed_loop_trace_eval/20260617_214140_long_trace_eval

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
/root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m pytest -q \
  tests/test_libero_client_config.py \
  tests/test_libero_transition_trace.py \
  tests/test_run_libero_smoke_script.py \
  tests/test_run_libero_eval_script.py \
  tests/test_write_libero_run_manifest.py
```

Trace summary tests:

```bash
/root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m pytest -q \
  tests/test_libero_transition_trace_summary.py
```

Trace summary command:

```bash
python scripts/summarize_libero_transition_trace.py <trace.jsonl> --format text
```

Runtime replay command:

```bash
python transition_trigger/scripts/evaluate_runtime_policy.py \
  --package-dir /root/autodl-tmp/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512 \
  --split test \
  --device cuda
```

## Next Decisions

1. Split the dirty tree into the commit groups above.
2. Decide LIBERO policy scope:
   keep `planner_threshold=0.70` for conservative integration only, or add an explicit
   LIBERO/demo calibration threshold such as `0.65`.
3. Keep paper claims anchored on the offline transition-trigger result unless LIBERO-specific
   calibration/training is added.
