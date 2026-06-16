# CALVIN ABC_D BoundaryHead TCN v1

## Goal

Train a standalone BoundaryHead on CALVIN LeRobot motion/state data. The model predicts only a scalar boundary logit for the question:

```text
Has the current low-level motion intent reached a termination point?
```

It does not predict a skill latent, does not use a planner, and does not consume vision or language embeddings at inference.

## Data

- Source repo: `CollisionCode/calvin_abc_d_lerobot_v2.1`
- Local root: `/root/autodl-tmp/datasets/calvin/lerobot/task_ABC_D`
- HF metadata reports `17870` episodes, but the repository API exposes `7870` actual parquet files under `data/`.
- Downloaded parquet files: `7870 / 7870` actual files from `/root/autodl-tmp/calvin_abc_d_actual_parquets.txt`
- Boundary sidecar: `/root/autodl-tmp/datasets/calvin/annotations/task_ABC_D_boundaries_actual.jsonl`
- Sidecar policy: one segment per actual downloaded parquet episode, with `end = num_rows - 1`
- Available windows after filtering:
  - total records: `229324`
  - train records: `206640`
  - validation records: `22684`
  - trajectories: `7870`
  - task ids: `389`

Class/window groups:

```text
positive       23610
ignore         30490
hard_negative 156456
easy_negative 18768
```

## Features

Per frame feature vector:

```text
[A, S, delta_A, delta_S, delta_gripper]
```

For this LeRobot conversion:

```text
action dim = 7
state dim  = 8
input dim  = 31
window     = 32 frames
```

## Labels

For each episode, the only supervised boundary is the final frame:

```text
y_t = positive if |t - episode_end| <= 2
ignore if 2 < |t - episode_end| <= 6
hard negative if 6 < |t - episode_end| <= 30
easy negative otherwise
```

Soft positive label:

```text
y_t = exp(-abs(t - episode_end) / 2.0)
```

## Model

`MotionStateBoundaryHead` causal TCN:

```yaml
hidden_dim: 768
kernel_size: 5
dilations: [1, 2, 4, 8, 16, 32]
dropout: 0.15
mlp_hidden_dim: 256
```

Training config:

```yaml
batch_size: 6144
epochs: 20
lr: 2.0e-4
weight_decay: 1.0e-4
epoch_size: 262144
positive_ratio: 0.5
hard_negative_ratio: 0.25
pos_weight: sqrt_neg_pos
```

GPU: RTX 4090 D 24GB. Memory probe with this config reached about `20.46GB`; observed training memory was about `21.9GB`.

## Results

Run dir:

```text
/root/autodl-tmp/HiMem-Bridge-VLA/motion_boundary/outputs/calvin_abc_d_tcn_v1
```

Best checkpoint:

```text
best.pt, epoch 4
```

Best validation summary:

```json
{
  "epoch": 4,
  "train_loss": 0.6320860025494598,
  "val_auprc": 0.6408783877841372,
  "train_samples": 206640,
  "val_samples": 22684,
  "pos_weight": 2.7263803467756444
}
```

Selected event metrics:

```text
threshold  precision  recall   f1
0.25       0.7305     0.9022   0.8073
0.65       0.8180     0.8679   0.8422
0.95       1.0000     0.0013   0.0025
```

Auto-selected thresholds:

```text
tau_p = 0.25
tau_w = 0.95
```

Interpretation:

- `tau_p = 0.25` is usable for high-recall replanning: recall is about `90.2%`.
- `tau_w = 0.95` satisfies the configured `95%` memory-write precision target, but recall is too low to be useful as a practical memory-write threshold.
- The best F1 in this grid is around threshold `0.65`, with F1 about `0.8422`, precision about `81.8%`, and recall about `86.8%`.

## Notes

This experiment validates that a motion/state-only BoundaryHead can learn a usable termination score from CALVIN low-level data. The conservative memory-write threshold needs further calibration, likely by relaxing the precision target or adding a separate local-maximum/segment-quality rule instead of relying on an extremely high raw probability threshold.
