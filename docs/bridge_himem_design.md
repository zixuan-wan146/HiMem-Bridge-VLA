# Bridge-HiMem Design

This document describes the active BridgeAttention integration surface. The current planner route is H32 single-token intent planning. Older H64 suffix queues and transition-trigger refresh logic are not part of this design.

## Model Path

```text
RGB views + prompt
  -> InternVL3 hidden states
  -> optional CoarsePlanner(H_t, s_t) -> one H32 plan token
  -> BridgeAdapter / BridgeAttention
  -> FlowMatchingActionHead
  -> 32-step action chunk
```

The baseline path remains a control: `InternVL3 fused tokens -> FlowMatchingActionHead`.

## Config Entry

Bridge-HiMem experiment files live under:

```text
configs/bridge_himem/base.yaml
configs/bridge_himem/experiments/*.yaml
```

New experiment knobs should go through YAML and `himem_bridge_vla/bridge_himem_config.py`. Do not hard-code experiment behavior in model or training scripts.

Validate configs before training:

```bash
python scripts/validate_bridge_himem_configs.py
```

## Fused Tokens

`fused_tokens` are the final InternVL3 hidden sequence produced from images and prompt. They are not planner tokens and not memory tokens.

```text
fused_only             baseline control
bridge_clean           action head sees only bridge tokens
bridge_residual        concat(fused_tokens, bridge_tokens)
bridge_gated_residual  concat(tanh(gate) * fused_tokens, bridge_tokens)
```

The clean planner integration should start from `bridge_clean` so the effect of the plan token is not mixed with a fused-token residual comparison.

## H32 Planner Token

When `coarse_planner.enabled` is true, `HiMemBridgeVLA` calls the planner during context augmentation:

```text
P_t = CoarsePlanner(fused_tokens, state)
```

The planner output shape is `[B, 1, D]`. BridgeAttention receives it as plan context. There is no cache, suffix offset, or transition-trigger refresh signal.

Important defaults:

```yaml
action_head:
  horizon: 32
coarse_planner:
  num_plan_steps: 1
  planning_horizon: 32
  input_memory: false
  placement: bridge_crosskv
```

## Memory Status

The active memory work is Dual-FIFO visual memory. The current step is standalone memory-side inference construction, not BridgeAttention consumption.

```text
BridgeAttention memory integration: not active in this step
coarse_planner.input_memory: false
```

See `docs/dual_fifo_visual_memory_design_zh.md` and `docs/dual_fifo_visual_memory_qa_zh.md`.

## Active Experiment Files

```text
baseline.yaml                  fused-token control
crosskv_clean.yaml             BridgeAttention baseline with memory config available but not wired into BridgeAttention
mixed_latent_clean.yaml        mixed-latent bridge baseline with memory config available but not wired into BridgeAttention
mixed_latent_skill.yaml        skill-token ablation, not active planner route
coarse_planner_crosskv.yaml    current H32 planner + bridge integration config
```
