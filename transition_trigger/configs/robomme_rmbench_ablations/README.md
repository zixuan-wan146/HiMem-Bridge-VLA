# RoboMME + RMBench Transition Trigger Ablations

Generated explicit configs for the 4 x 4 x 2 x 2 training grid.

- `window_size`: 16, 24, 32, 48
- `feature_set`: value_mask, value_delta_mask, value_mask_domain, full
- `model.type`: ssm, transformer
- `model.d_model`: 256, 512

Batch sizes are set from RTX 4090 memory probes on the heaviest `window_size=48, feature_set=full` configs:

- `transformer, d_model=512`: 2816
- `transformer, d_model=256`: 5632
- `ssm, d_model=512`: 8704
- `ssm, d_model=256`: 16384

Each YAML is standalone and writes to `/root/autodl-tmp/runs/transition_trigger/robomme_rmbench_ablations/<run_name>`.
