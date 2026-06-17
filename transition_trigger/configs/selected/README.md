# Selected Transition Trigger Configs

This directory contains deployment-facing configs selected from completed ablations.

Current default:

```text
robomme_rmbench_w32_value_delta_transformer_d512.yaml
```

Remote runtime package:

```text
/root/autodl-tmp/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512
```

Use this config with `checkpoint.pt` from the runtime package for memory-write plus replan
integration. The same package also keeps the boundary-F1 upper checkpoint and a conservative W24
memory-write fallback.
