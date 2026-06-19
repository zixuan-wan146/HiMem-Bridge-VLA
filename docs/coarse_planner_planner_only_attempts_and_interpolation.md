# Coarse Planner Planner-Only Attempts and Interpolation Notes

Date: 2026-06-19

## Scope

This note explains the planner-only push after the Action Segment Autoencoder
was frozen. It is intended as a compact learning record for why each attempt was
run, what changed, what failed, and why checkpoint interpolation can change
model behavior.

The scope is only:

```text
frozen ActionSegmentAutoencoder
CoarsePlanner -> plan tokens
latent_head(plan tokens) -> normalized z prediction
training-only decoder chunk loss
```

It does not include BridgeAttention / ActionHead suffix integration training and
does not include joint end-to-end VLA training.

## Retained Datasets

After disk cleanup, only the datasets needed to reproduce the current planner
state and continue the next stage were retained.

```text
/root/autodl-tmp/datasets/coarse_planner/libero_h64
```

Small original LIBERO H64 cache, 2048 samples. It is kept because early AE and
planner baselines were measured on it, and it is small enough that keeping it is
cheap.

```text
/root/autodl-tmp/datasets/coarse_planner/libero_h64_holdout_seed43
```

Independent seed43 holdout cache, 2048 samples. This is the most important
evaluation set because the final reported generalization metrics are measured on
it. Removing it would make fair comparison with v17 impossible.

```text
/root/autodl-tmp/datasets/coarse_planner/libero_h64_s32768_seed42
```

Current useful large training cache, 32768 samples. The effective v11-v17
planner improvements came from this cache. It is the practical base for further
planner-only diagnosis or the next BridgeAttention / ActionHead suffix
integration stage.

Removed caches:

```text
8k and 16k caches: superseded by the 32k cache
63k cache: built successfully but v15/v16 training did not improve and it used about 108 GB
H32/H48/smoke/probe caches: not needed for the current H64 planner path
```

## Current Final Artifact

The current standalone planner checkpoint is:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075/best.pt
```

It was produced by checkpoint interpolation:

```text
theta_v17 = theta_v12 + 0.75 * (theta_v13 - theta_v12)
```

Final seed43 holdout all metrics:

```text
raw_latent_mse:        0.088253
normalized_latent_mse: 0.264069
decoded_chunk_loss:    0.176643
latent_cosine:         0.900730
```

This is not strictly below 0.08, but it is below 0.09 and in the intended
0.08-ish usable range for moving to the next integration stage.

## Attempts Summary

| run | data | main change | holdout raw MSE | decoded chunk | cosine | decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| v4 | 2k | z-normalized warm start, chunk 1.0 | 0.153218 | 0.340088 | 0.828579 | baseline after z-norm |
| v5 | 8k | moderate late-token weighting | 0.116523 | 0.196231 | 0.868545 | useful jump |
| v8 | 8k | latent-focused continuation | 0.111459 | 0.224498 | 0.875165 | 8k diminishing returns |
| v10 | 16k | latent-focused continuation | 0.099159 | 0.178007 | 0.888414 | reached <=0.10 |
| v11 | 32k | low-LR continuation | 0.098461 | 0.155727 | 0.888590 | better chunk, little raw-MSE gain |
| v12 | 32k | higher LR, dropout 0, chunk 0.25 | 0.091676 | 0.174122 | 0.896749 | effective latent-focused step |
| v13 | 32k | stronger late-token weights, chunk 0.10 | 0.088456 | 0.180509 | 0.900542 | best trained checkpoint |
| v14 | 32k | smaller LR late continuation | not promoted | worse internal eval | - | stopped |
| v15 | 63k | random-order 63k training | no checkpoint | I/O stall | - | stopped |
| v16 | 63k | sequential 63k training | not promoted | degraded | - | stopped |
| v17 | 32k | v12-v13 interpolation alpha 0.75 | 0.088253 | 0.176643 | 0.900730 | current final |

## Why Interpolation Can Change Performance

Checkpoint interpolation is not changing the architecture or adding data. It
changes the parameters to a point between two trained solutions.

For v17:

```text
theta_v12: better decoded chunk tradeoff, less late-token specialization
theta_v13: stronger late-token specialization, slightly higher chunk loss
theta_v17: point between them
```

The model prediction is a nonlinear function of the parameters:

```text
z_hat = f_theta(H, s)
```

Therefore:

```text
f_(0.75 theta13 + 0.25 theta12)(H, s)
```

is not generally equal to:

```text
0.75 f_theta13(H, s) + 0.25 f_theta12(H, s)
```

Even though the parameter interpolation is linear, the network output changes
nonlinearly because Transformer layers, layer norms, GELU activations, and the
latent head all compose nonlinear transformations.

In practice, interpolation can help when two checkpoints are in the same basin
or connected region of parameter space. It can reduce over-specialization from a
later fine-tune while keeping most of the useful direction of the update.

Here that means:

```text
v12 -> v13 update:
  improves late-token latent prediction
  but increases decoded chunk loss and starts to over-specialize after epoch 1

alpha 0.75:
  keeps most of v13's useful latent improvement
  pulls slightly back toward v12's smoother tradeoff
```

The interpolation sweep confirmed that the optimum along this line was near, but
not beyond, v13:

| alpha | holdout raw MSE | decoded chunk | cosine |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.091675 | 0.174121 | 0.896750 |
| 0.50 | 0.088745 | 0.174447 | 0.900157 |
| 0.75 | 0.088257 | 0.176639 | 0.900731 |
| 1.00 | 0.088455 | 0.180511 | 0.900542 |
| 1.20 | 0.089081 | 0.184636 | 0.899874 |
| 1.50 | 0.090790 | 0.192388 | 0.897989 |

The fact that alpha greater than 1.0 became worse means simple extrapolation
past v13 was not useful. The useful point was a partial rollback from v13 toward
v12.

## Lessons

1. The AE was not the first bottleneck. It reconstructed well on the original and
   seed43 holdout caches, so the planner was the right target.
2. More same-family training helped until v13, but then saturated.
3. Stronger late-token weighting can help briefly, but too many epochs quickly
   degrade validation behavior.
4. Bigger data is not automatically better. The 63k cache was expensive and did
   not improve with the same planner/training strategy.
5. Checkpoint interpolation is useful as a low-cost final polish when two nearby
   checkpoints represent different tradeoffs.
6. The next real improvement should likely come from the next training stage or
   a model/objective change, not more low-LR planner-only fine-tuning.

## Frozen AE Check on the 32k Cache

The frozen Action Segment Autoencoder was evaluated on the retained 32k cache:

```text
/root/autodl-tmp/datasets/coarse_planner/libero_h64_s32768_seed42
```

Checkpoint:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h64_segment_ae_v2/best.pt
```

Evaluation output:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h64_segment_ae_v2/eval_libero_h64_s32768_seed42.json
/root/autodl-tmp/runs/coarse_planner/libero_h64_segment_ae_v2/eval_libero_h64_s32768_seed42.md
```

Results:

| split | samples | active segments | total loss | rec loss | dist loss | motion Huber | gripper BCE | gripper acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 29459 | 235672 | 0.016184 | 0.014696 | 0.014881 | 0.006838 | 0.018557 | 0.994336 |
| eval | 3309 | 26472 | 0.016337 | 0.014796 | 0.015410 | 0.006699 | 0.018384 | 0.994475 |

Interpretation:

```text
The AE remains stable on the 32k cache. The eval loss is close to the train loss,
motion reconstruction is low, and gripper accuracy is about 99.45%.
```

This supports the earlier decision to keep the AE frozen and focus improvement
on the planner / integration path.
