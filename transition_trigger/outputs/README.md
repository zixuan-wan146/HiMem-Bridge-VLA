# Transition Trigger Outputs

Training and evaluation artifacts should be written under this directory.

Expected run layout:

```text
outputs/<run-name>/
  resolved_config.yaml
  best.pt
  train_history.json
  eval_metrics.json
```

Large checkpoints and logs should stay local to the data disk on remote machines and should not be
committed.
