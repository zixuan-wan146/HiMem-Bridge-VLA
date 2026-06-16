# BoundaryHead Next Experiment Plan

## Current Status

The first ABC_D run (`calvin_abc_d_tcn_v1`) proves that a motion/state-only TCN can learn a usable boundary score:

```text
best val AUPRC: 0.6409
best F1 grid point: threshold 0.65, F1 0.8422, precision 0.8180, recall 0.8679
replanning threshold: tau_p = 0.25, recall 0.9022, precision 0.7305
memory-write threshold under 0.95 precision target: tau_w = 0.95, recall 0.0013
```

Conclusion:

```text
The score is useful for replanning.
The current high-precision memory-write calibration is too conservative.
```

The next experiments should improve practical thresholding first, then test whether model capacity or history length is the main bottleneck.

## Data Invariants

Use the actual downloaded ABC_D parquet manifest, not the misleading metadata episode count.

```text
LeRobot root:
/root/autodl-tmp/datasets/calvin/lerobot/task_ABC_D

Actual parquet manifest:
/root/autodl-tmp/calvin_abc_d_actual_parquets.txt

Actual parquet count:
7870 / 7870

Boundary sidecar:
/root/autodl-tmp/datasets/calvin/annotations/task_ABC_D_boundaries_actual.jsonl
```

Input features remain:

```text
[A, S, delta_A, delta_S, delta_gripper]
```

The sidecar uses one weak boundary per episode:

```text
event_frame = episode_end = num_rows - 1
```

This is enough for a first BoundaryHead, but it should be treated as weak supervision rather than exact long-horizon skill boundaries.

## Success Criteria

Replanning:

```text
recall >= 0.90
precision >= 0.70
duplicate trigger rate near 0
mean delay in [-1, +1] frames
```

Memory-write:

```text
precision >= 0.85
recall >= 0.50
duplicate trigger rate near 0
```

Overall detector quality:

```text
best F1 >= 0.85
val AUPRC > 0.64
```

If memory-write precision `0.85` still causes recall to collapse, keep `tau_w` around the best-F1 region and add a practical write gate:

```text
b_t >= tau_w
local maximum
cooldown satisfied
segment length >= 8 to 10 frames
```

## Experiment A: Threshold Sweep Only

Purpose:

```text
Do not retrain. Re-evaluate the current best checkpoint with a denser threshold grid and a practical memory precision target.
```

Config:

```text
motion_boundary/configs/autodl_calvin_abc_d_tcn_v1_threshold_sweep.yaml
```

Checkpoint:

```text
/root/autodl-tmp/HiMem-Bridge-VLA/motion_boundary/outputs/calvin_abc_d_tcn_v1/best.pt
```

Command:

```bash
cd /root/autodl-tmp/HiMem-Bridge-VLA
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m motion_boundary.evaluate \
  --config motion_boundary/configs/autodl_calvin_abc_d_tcn_v1_threshold_sweep.yaml \
  --checkpoint /root/autodl-tmp/HiMem-Bridge-VLA/motion_boundary/outputs/calvin_abc_d_tcn_v1/best.pt \
  --output /root/autodl-tmp/HiMem-Bridge-VLA/motion_boundary/outputs/calvin_abc_d_tcn_v1_threshold_sweep/eval_metrics.json \
  --device cuda
```

Expected decision:

```text
Pick a practical tau_w from 0.55 to 0.80 if precision >= 0.85 and recall is not collapsed.
```

## Experiment B: Smaller TCN

Purpose:

```text
Test whether the v1 768-hidden TCN is overfitting. A smaller model may produce smoother, better-calibrated scores.
```

Config:

```text
motion_boundary/configs/autodl_calvin_abc_d_tcn_hidden384_w32.yaml
```

Main changes:

```text
window_size: 32
hidden_dim: 384
mlp_hidden_dim: 128
batch_size: 12288
lr: 2.5e-4
```

Command:

```bash
cd /root/autodl-tmp/HiMem-Bridge-VLA
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m motion_boundary.train \
  --config motion_boundary/configs/autodl_calvin_abc_d_tcn_hidden384_w32.yaml \
  --device cuda
```

Then evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m motion_boundary.evaluate \
  --config motion_boundary/configs/autodl_calvin_abc_d_tcn_hidden384_w32.yaml \
  --checkpoint /root/autodl-tmp/HiMem-Bridge-VLA/motion_boundary/outputs/calvin_abc_d_tcn_hidden384_w32/best.pt \
  --output /root/autodl-tmp/HiMem-Bridge-VLA/motion_boundary/outputs/calvin_abc_d_tcn_hidden384_w32/eval_metrics.json \
  --device cuda
```

Expected decision:

```text
Keep the smaller model if AUPRC stays close to v1 and memory-write precision/recall improves.
Reject it if AUPRC or best F1 drops materially.
```

## Experiment C: Longer History Window

Purpose:

```text
Test whether boundary detection needs longer motion history than 32 frames.
```

Config:

```text
motion_boundary/configs/autodl_calvin_abc_d_tcn_hidden768_w48.yaml
```

Main changes:

```text
window_size: 48
hidden_dim: 768
batch_size: 4096
```

The smaller batch is intentional. A 48-frame window with hidden 768 should remain near the 20GB GPU memory target on RTX 4090D.

Command:

```bash
cd /root/autodl-tmp/HiMem-Bridge-VLA
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m motion_boundary.train \
  --config motion_boundary/configs/autodl_calvin_abc_d_tcn_hidden768_w48.yaml \
  --device cuda
```

Then evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniforge3/envs/Evo1/bin/python -m motion_boundary.evaluate \
  --config motion_boundary/configs/autodl_calvin_abc_d_tcn_hidden768_w48.yaml \
  --checkpoint /root/autodl-tmp/HiMem-Bridge-VLA/motion_boundary/outputs/calvin_abc_d_tcn_hidden768_w48/best.pt \
  --output /root/autodl-tmp/HiMem-Bridge-VLA/motion_boundary/outputs/calvin_abc_d_tcn_hidden768_w48/eval_metrics.json \
  --device cuda
```

Expected decision:

```text
Keep the longer window only if it improves AUPRC/F1 or memory-write recall at the same precision.
If it only increases cost, stay with window 32.
```

## After A/B/C

If the best model still cannot satisfy practical memory-write criteria, run label ablations:

```text
positive_radius: 1, 2, 3
ignore_radius: 4, 6, 8
label_sigma: 1.5, 2.0, 3.0
hard_labels vs soft_labels
```

The most likely useful direction is sharpening the positive zone, because v1 already detects the boundary for replanning but does not produce enough high-confidence peaks for memory-write.
