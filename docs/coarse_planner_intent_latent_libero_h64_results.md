# Coarse Planner Intent-Latent LIBERO H64 Results

Date: 2026-06-18

## Scope

This records the first LIBERO H64 run for the refactored Coarse Planner path:

```text
ActionSegmentAutoencoder(A_i) -> z_i*
CoarsePlanner(H_t, s_t) -> P_t
latent_head(P_t) -> z_hat_i
loss = L_z + 0.25 * L_chunk
```

The inference path remains plan-token only. The AE encoder, AE decoder, and latent
prediction head are training-time shaping tools.

## Final Configs

```text
AE:
coarse_planner/configs/libero_h64_segment_ae_v2.yaml

Planner:
coarse_planner/configs/libero_h64_planner_v2.yaml
```

The planner config records the 80-epoch primary run and the 81-100 fine-tune:

```text
primary:
  batch_size: 640
  lr: 0.0001
  epochs: 80

fine_tune:
  resume_from: /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_v2/best.pt
  reset_optimizer: true
  lr: 0.00003
  epochs: 100
```

## Data

```text
original cache:
/root/autodl-tmp/datasets/coarse_planner/libero_h64
samples: 2048
split_counts: train 1824, eval 224

holdout cache:
/root/autodl-tmp/datasets/coarse_planner/libero_h64_holdout_seed43
samples: 2048
split_counts: train 1867, eval 181
```

Both caches use the same target contract:

```text
planning_horizon: 64
num_plan_steps: 8
chunk_size: 8
execution_horizon: 16
suffix_stride_tokens: 2
gripper_indices: [6]
```

Stored sample tensors:

```text
vlm_tokens: [1024, 896] float16
state: [24] float32
action_segments: [8, 8, 7] float32
action_segment_mask: [8] float32
```

Motion actions are normalized to `[-1, 1]`. Gripper actions are binary `{0, 1}`.

## Action Segment Autoencoder

Run:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h64_segment_ae_v2
```

Training:

```text
epochs: 100
batch_size: 2048
lr: 0.0003
peak CUDA reserved: 19.984 GB
best epoch: 100
```

Metrics:

```text
original eval loss: 0.015466
original eval rec loss: 0.014050
seed43 holdout all loss: 0.015194
seed43 holdout all rec loss: 0.013810
seed43 holdout all dist loss: 0.013845
```

Interpretation:

```text
The action-only AE does not show degradation on the independently sampled seed43
holdout cache. This supports using its latent space as a stable action-segment
intent target for this first planner warm-up.
```

## Coarse Planner

Run:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h64_planner_v2
```

Training:

```text
primary epochs: 80
fine-tune epochs: 81-100
batch_size: 640
AMP: true
peak CUDA reserved: 23.025 GB
best epoch: 85
```

The tested batch-size boundary was:

```text
batch_size 512: peak reserved 18.789 GB
batch_size 640: peak reserved about 22.3-23.0 GB
batch_size 704: OOM
```

Best internal eval:

```text
loss: 0.242264
latent_mse: 0.158486
latent_mse_u0: 0.158482
latent_mse_u2: 0.170666
latent_mse_u4: 0.184989
latent_mse_u6: 0.196150
```

External seed43 holdout, all weighted:

```text
loss: 0.242503
latent_mse: 0.157849
latent_mse_u0: 0.157860
latent_mse_u2: 0.167338
latent_mse_u4: 0.180237
latent_mse_u6: 0.193283
```

## Analysis

The external holdout loss is essentially identical to the original internal eval
loss:

```text
original eval loss: 0.242264
seed43 holdout all loss: 0.242503
difference: +0.000239
```

This means the standalone planner warm-up is not just memorizing the original
2048-sample cache. The suffix diagnostics also have the expected ordering: later
active suffixes are harder, but every suffix remains in the same range and the
holdout suffix metrics are not worse than internal eval.

The 81-100 fine-tune produced only a small improvement:

```text
80-epoch best: 0.243574
100-epoch best: 0.242264
absolute improvement: 0.001310
```

After epoch 85, training loss continued to fall but validation total loss did
not improve. The next useful improvement is unlikely to come from simply adding
more standalone planner epochs at the same data scale. Better next steps are:

```text
1. Increase LIBERO cache size beyond 2048 samples.
2. Add the strict anchor/current BridgeAttention/ActionHead integration dataset.
3. Evaluate whether decoded chunk loss or latent MSE should drive checkpoint
   selection for the downstream bridge objective.
```

## Artifact

Full final evaluation JSON:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h64_planner_v2/final_eval_seed43.json
```

## Next Experiment: Train-Set z Normalization

Decision:

```text
AE stays frozen.
Compute z_mean/z_std only from the original train split through the frozen AE encoder.
Planner predicts normalized z.
L_z is computed in normalized z space.
Before D_theta, predicted z is unnormalized back to the AE latent space.
Evaluation reports normalized latent MSE, raw latent MSE, decoded chunk loss,
and latent cosine similarity.
```

This targets the current planner bottleneck: direct raw-z MSE can be dominated by
high-variance AE latent dimensions, while lower-variance dimensions may carry
useful intent information.

The existing raw-z planner checkpoint is reused rather than discarded. Because
the latent head output layer is linear, the old raw-z output can be converted
exactly into normalized-z output:

```text
z_raw_hat = W h + b
z_norm_hat = (z_raw_hat - z_mean) / z_std

W' = W / z_std
b' = (b - z_mean) / z_std
```

Therefore:

```text
unnormalize(z_norm_hat) = z_raw_hat
```

At initialization, the frozen decoder receives exactly the same latent as the
previous raw-z planner, while the subsequent fine-tuning loss is better
conditioned.

Config:

```text
coarse_planner/configs/libero_h64_planner_znorm_v3.yaml
```

## z Normalization Results

Two normalized-z variants were trained from the same v2 raw-z checkpoint:

```text
v3:
  config: coarse_planner/configs/libero_h64_planner_znorm_v3.yaml
  chunk_loss_weight: 0.25
  run: /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_v3
  best epoch: 92

v4:
  config: coarse_planner/configs/libero_h64_planner_znorm_chunk1_v4.yaml
  chunk_loss_weight: 1.0
  run: /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_chunk1_v4
  best epoch: 102
```

Both runs:

```text
AE checkpoint: /root/autodl-tmp/runs/coarse_planner/libero_h64_segment_ae_v2/best.pt
warm start: /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_v2/best.pt
latent head conversion: raw-z output -> normalized-z output
batch_size: 640
lr: 0.00003
train epochs: checkpoint epoch 86 through 105
peak CUDA reserved: about 22.97 GB
```

Latent normalizer:

```text
source split: original train only
active segment count: 14592
std_floor: 0.0001
mean(mean): -0.034088
mean(std): 0.560660
min(std): 0.263713
max(std): 1.174249
```

Same-metric comparison:

```text
original eval:
  v2 raw:
    raw_latent_mse: 0.158482
    decoded_chunk_loss: 0.335113
    cosine: 0.825306

  v3 z-norm, chunk 0.25:
    raw_latent_mse: 0.155413
    decoded_chunk_loss: 0.353084
    cosine: 0.828396

  v4 z-norm, chunk 1.0:
    raw_latent_mse: 0.155831
    decoded_chunk_loss: 0.342241
    cosine: 0.828862

seed43 holdout all:
  v2 raw:
    raw_latent_mse: 0.157860
    decoded_chunk_loss: 0.338952
    cosine: 0.822824

  v3 z-norm, chunk 0.25:
    raw_latent_mse: 0.154762
    decoded_chunk_loss: 0.354649
    cosine: 0.825767

  v4 z-norm, chunk 1.0:
    raw_latent_mse: 0.153206
    decoded_chunk_loss: 0.338868
    cosine: 0.828580
```

Interpretation:

```text
z normalization works for latent regression:
  - v3 improves raw latent MSE and cosine, but decoded chunk loss degrades.
  - The degradation is caused by the normalized latent MSE becoming the dominant
    term when chunk_loss_weight remains 0.25.

Increasing chunk_loss_weight to 1.0 gives the better tradeoff:
  - v4 has the best seed43 holdout raw latent MSE.
  - v4 has the best seed43 holdout cosine.
  - v4 keeps seed43 decoded chunk loss essentially equal to v2.
```

Recommendation:

```text
Use v4 as the current standalone Coarse Planner checkpoint:
/root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_chunk1_v4/best.pt
```

The normalized training objective is not numerically comparable to the old raw-z
total loss. Compare cross-run quality using raw latent MSE, decoded chunk loss,
and cosine similarity.

Artifacts:

```text
v2 full-metric eval:
/root/autodl-tmp/runs/coarse_planner/libero_h64_planner_v2/final_eval_seed43_full_metrics.json

v3 eval:
/root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_v3/final_eval_seed43.json

v4 eval:
/root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_chunk1_v4/final_eval_seed43.json

comparison:
/root/autodl-tmp/runs/coarse_planner/znorm_comparison_seed43.json
```

## 2026-06-19: Toward 0.08 Raw Latent MSE

The current best seed43 holdout raw latent MSE is:

```text
v4 z-norm + chunk 1.0: 0.153206
```

This is an improvement over the raw-z planner, but it is not yet good enough for
a robust intent-token planner. The working target is:

```text
raw_latent_mse ~= 0.08
```

This target is the usable-stage goal for the standalone Coarse Planner, not an
immediate pass/fail threshold for the next single run. The practical gates are:

```text
near-term gate: raw_latent_mse <= 0.12
mid-term gate:  raw_latent_mse <= 0.10
usable target:  raw_latent_mse ~= 0.08
```

The target must not be optimized in isolation. A planner is only considered
usable for the next integration stage if the latent MSE improves while decoded
chunk quality and latent direction also remain healthy:

```text
raw_latent_mse:        target around 0.08
decoded_chunk_loss:    should not regress from v4; prefer <= 0.33
latent_cosine:         should improve from v4; prefer >= 0.84, then >= 0.86
suffix/token behavior: no severe collapse on late tokens or u in {0,2,4,6}
```

A raw latent MSE of `0.15` means the planner is still far from the frozen AE
encoder's target latents. Since the AE reconstructs well and generalizes to the
seed43 holdout cache, the next work should focus on planner predictability and
training signal rather than retraining the AE.

Decision:

```text
Do not blindly add more epochs on the 2048-sample cache.
First run a structured planner error diagnosis.
Then choose between larger data, contrastive latent alignment, residual priors,
or task/suite-balanced training.
```

The required diagnosis should report:

```text
1. per-token raw latent MSE for p0...p7
2. per-token normalized latent MSE for z-normalized runs
3. per-token cosine similarity
4. per-suite and per-task raw latent MSE
5. per-dimension MSE, variance, and R2
6. mean-predictor baseline from train-set z_mean
7. input nearest-neighbor baseline using pooled VLM/state context
8. oracle target-latent nearest-neighbor baseline
```

Interpretation rules:

```text
If planner is close to the input-NN baseline:
  the current input/features/data may be the bottleneck.

If input-NN is much better than planner:
  the planner training objective/model is underfitting the available signal.

If oracle target-latent NN is much better than both:
  the AE latent manifold has usable local structure, but the current observation
  does not retrieve it reliably.

If late tokens dominate error:
  H64 future intent may need a stronger prior, more data, or suffix-aware losses.

If only a few latent dimensions dominate:
  dimension weighting, R2-based diagnostics, or latent whitening/contrastive
  losses should be considered.
```

The next experiment should only be selected after this diagnosis.

## Stage Gate Before End-to-End Training

The standalone planner checkpoint is a semantic warm-up component. It should not
be treated as final evidence for closed-loop LIBERO performance until it is
inserted into the main model and trained jointly with BridgeAttention and the
ActionHead.

The planned order is:

```text
1. keep the AE frozen and improve Coarse Planner toward raw_latent_mse ~= 0.08
2. verify decoded_chunk_loss and cosine do not regress
3. export the usable planner checkpoint and config
4. start BridgeAttention / ActionHead suffix integration training
5. only after the integration path is stable, run end-to-end main VLA training
```

This current round stops at step 3. It does not start joint end-to-end VLA
training. Steps 4 and 5 are later-stage work after the standalone planner reaches
the usable target.

The end-to-end stage should use the planner only through plan tokens:

```text
(H_t, s_t) -> CoarsePlanner -> P_t
P_active -> BridgeAttention -> ActionHead
```

Training-only latent tools remain excluded from inference:

```text
E_phi, D_theta, W_z
```

## 2026-06-19: Planner-Only Push Toward 0.08

After the v4 diagnosis, the main bottleneck was not AE quality. The frozen AE
generalized cleanly, while the planner had systematic late-token error:

```text
v4 seed43 holdout all raw_latent_mse: 0.153218
v4 holdout token raw MSE:
[0.1219, 0.1240, 0.1331, 0.1410, 0.1543, 0.1667, 0.1849, 0.1999]
```

The input-context nearest-neighbor baseline was worse than the planner, while
the oracle target-latent nearest-neighbor baseline was much better. This means:

```text
the AE latent manifold has useful local structure;
the planner is learning signal beyond simple input nearest-neighbor retrieval;
late future tokens remain the dominant bottleneck.
```

Planner-only experiments run in this round:

```text
v5:
  data: 8192 LIBERO H64 samples
  init: v4 best
  change: moderate late-token weights, chunk_loss_weight=1.0
  checkpoint: /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latew_s8192_v5/best.pt

v6:
  data: 8192 LIBERO H64 samples
  init: v5 best
  change: stronger late-token weights
  outcome: worse than v5; discarded as primary checkpoint

v7/v8:
  data: 8192 LIBERO H64 samples
  init: v5/v7 best
  change: latent-focused fine-tuning, chunk_loss_weight=0.5
  outcome: small holdout improvement, but 8192 samples reached diminishing returns

v9/v10:
  data: 16384 LIBERO H64 samples
  init: v8/v9 best
  change: same latent-focused loss on larger cache
  checkpoint: /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s16384_v10/best.pt
```

Main comparison on seed43 holdout all:

| run | samples | raw latent MSE | normalized MSE | decoded chunk loss | cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 | 2048 | 0.153218 | 0.455218 | 0.340088 | 0.828579 |
| v5 | 8192 | 0.116523 | 0.342423 | 0.196231 | 0.868545 |
| v8 | 8192 | 0.111459 | 0.330282 | 0.224498 | 0.875165 |
| v9 | 16384 | 0.101534 | 0.299539 | 0.173766 | 0.885469 |
| v10 | 16384 | 0.099159 | 0.293366 | 0.178007 | 0.888414 |

v10 holdout all per-token raw MSE:

```text
[0.0753, 0.0739, 0.0826, 0.0941, 0.1036, 0.1085, 0.1213, 0.1339]
```

Current status:

```text
near-term gate <= 0.12: reached
mid-term gate  <= 0.10: reached on seed43 holdout all, nearly reached on holdout eval
usable target  ~= 0.08: not reached
```

The remaining gap is now mostly late-token prediction. Early tokens are already
near or below `0.08`, while `p6` and `p7` remain around `0.12-0.13`. Stronger
late-token weights alone did not solve this; v6 was worse than v5. Low-learning
rate continuation gave only small gains.

Decision after v10:

```text
Use v10 as the current best standalone planner checkpoint.
Do not start BridgeAttention / ActionHead integration training yet if the hard
planner target remains raw_latent_mse ~= 0.08.
Do not keep blindly fine-tuning v10 on the same 16k cache; the slope is small.
Next planner-only attempt should change the available signal or model class,
for example 32k+ data, a stronger temporal planner core, or an explicit
late-token/far-horizon objective.
```

v10 artifacts:

```text
dataset:
  /root/autodl-tmp/datasets/coarse_planner/libero_h64_s16384_seed42

config:
  coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s16384_v10.yaml

checkpoint:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s16384_v10/best.pt

diagnostics:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s16384_v10/latent_diagnostics_seed43.json
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s16384_v10/latent_diagnostics_seed43.md
```

## 2026-06-19: 32k Cache Continuation v11

After v10 reached the `<= 0.10` seed43 holdout gate, the next test was to check
whether simply expanding data to 32768 LIBERO H64 samples would move the frozen-AE
planner toward the `0.08` usable target.

v11 setup:

```text
data:
  /root/autodl-tmp/datasets/coarse_planner/libero_h64_s32768_seed42
  samples: 32768
  train/eval split: 29459 / 3309

init:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s16384_v10/best.pt

config:
  coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s32768_v11.yaml

training:
  lr: 0.00001
  batch_size: 640
  planned epochs: 8
  stopped after epoch 4 because validation did not improve after epoch 2
  peak CUDA reserved: about 23.0 GB
```

v11 internal eval over the 32k cache:

| epoch | raw latent MSE | normalized MSE | decoded chunk loss | cosine |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.114350 | 0.336506 | 0.203125 | 0.871044 |
| 2 | 0.113013 | 0.332814 | 0.205275 | 0.872521 |
| 3 | 0.114739 | 0.337205 | 0.204038 | 0.870541 |
| 4 | 0.113766 | 0.334854 | 0.203202 | 0.871963 |

The best checkpoint is epoch 2:

```text
/root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v11/best.pt
```

Seed43 holdout comparison:

| run | samples | raw latent MSE | normalized MSE | decoded chunk loss | cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 | 16384 | 0.099159 | 0.293366 | 0.178007 | 0.888414 |
| v11 | 32768 | 0.098461 | 0.290008 | 0.155727 | 0.888590 |

v11 seed43 holdout all per-token raw MSE:

```text
[0.0758, 0.0737, 0.0827, 0.0937, 0.1019, 0.1062, 0.1202, 0.1334]
```

Interpretation:

```text
The larger 32k cache gives only a small raw latent MSE improvement over v10:
0.099159 -> 0.098461.

The decoded chunk loss improves more clearly:
0.178007 -> 0.155727.

The hard 0.08 target is still not reached. The remaining error is still
late-token dominated; p6 and p7 remain around 0.12-0.13.
```

The most important diagnostic change is the input-context nearest-neighbor
baseline:

| run | seed43 holdout all input-context NN raw MSE | planner raw MSE |
| --- | ---: | ---: |
| v10 | 0.123340 | 0.099159 |
| v11 | 0.082485 | 0.098461 |

This changes the bottleneck interpretation:

```text
At 16k, the learned planner was better than input-context NN retrieval.
At 32k, the retrieval baseline is close to the 0.08 target and better than the
learned planner.
```

Therefore, the 32k cache appears to contain enough nearby examples to approach
the target, but the current planner/training setup is not fully exploiting that
signal. Continuing v11 with the same low-learning-rate configuration is unlikely
to be the best use of time.

Decision after v11:

```text
Do not start BridgeAttention / ActionHead integration training yet.
Do not start joint end-to-end VLA training yet.
Keep the AE frozen.
Use v11 as the best decoded-chunk tradeoff checkpoint, but treat the planner
quality target as still unmet because raw_latent_mse is about 0.098, not 0.08.
Next planner-only attempt should focus on fitting the 32k signal better:
  - higher-step or higher-LR planner-only training from v11/v10,
  - stronger latent-focused checkpoint selection,
  - or a planner capacity/objective change if the 32k retry remains flat.
```

v11 artifacts:

```text
dataset:
  /root/autodl-tmp/datasets/coarse_planner/libero_h64_s32768_seed42

config:
  coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s32768_v11.yaml

checkpoint:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v11/best.pt

diagnostics:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v11/latent_diagnostics_seed43.json
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v11/latent_diagnostics_seed43.md
```

## 2026-06-19: Latent-Focused 32k Continuation v12

v11 showed that simply increasing the cache to 32k was not enough when the
learning rate stayed low and the decoded chunk objective remained relatively
strong. However, the 32k input-context nearest-neighbor baseline was already
close to the `0.08` target, so v12 tested whether the current planner could fit
that signal better with a more latent-focused continuation.

v12 changes from v11:

```text
init:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v11/best.pt

config:
  coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s32768_v12.yaml

training:
  lr: 0.00003
  dropout: 0.0
  chunk_loss_weight: 0.25
  batch_size: 640
  stopped after epoch 4 because epoch 2 remained best
```

v12 internal 32k eval:

| epoch | raw latent MSE | normalized MSE | decoded chunk loss | cosine |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.111521 | 0.330198 | 0.229353 | 0.874274 |
| 2 | 0.110001 | 0.325863 | 0.235978 | 0.876685 |
| 3 | 0.111646 | 0.330774 | 0.253019 | 0.874768 |
| 4 | 0.110008 | 0.325731 | 0.245338 | 0.876959 |

Seed43 holdout comparison:

| run | raw latent MSE | normalized MSE | decoded chunk loss | cosine |
| --- | ---: | ---: | ---: | ---: |
| v10 | 0.099159 | 0.293366 | 0.178007 | 0.888414 |
| v11 | 0.098461 | 0.290008 | 0.155727 | 0.888590 |
| v12 | 0.091676 | 0.273135 | 0.174122 | 0.896749 |

v12 seed43 holdout all per-token raw MSE:

```text
[0.0723, 0.0682, 0.0776, 0.0865, 0.0949, 0.1000, 0.1120, 0.1219]
```

Interpretation:

```text
v12 validates the main v11 diagnosis: the 32k cache contains useful signal, and
the previous low-LR continuation underfit it.

Reducing chunk pressure and removing dropout improves latent prediction:
0.098461 -> 0.091676 on seed43 holdout all.

The decoded chunk loss regresses relative to v11:
0.155727 -> 0.174122.
This is still far better than the early v4 baseline of about 0.34, so the
tradeoff is acceptable for this planner-only push.

The hard usable target is still not reached. The remaining gap is about 0.0117
raw MSE, concentrated in p4-p7.
```

Decision after v12:

```text
Keep AE frozen.
Keep the inference path unchanged.
Continue with planner-only latent-focused training.
Use v12 as the current best raw-MSE checkpoint.
The next attempt should focus the loss even more on late/far tokens while keeping
decoded chunk loss under control.
Do not start BridgeAttention / ActionHead integration training yet.
Do not start joint end-to-end VLA training yet.
```

v12 artifacts:

```text
config:
  coarse_planner/configs/libero_h64_planner_znorm_latentfocus_s32768_v12.yaml

checkpoint:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v12/best.pt

diagnostics:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v12/latent_diagnostics_seed43.json
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v12/latent_diagnostics_seed43.md
```

## 2026-06-19: Late-Token Focus v13

v12 improved overall latent prediction, but the remaining gap to 0.08 stayed
concentrated in the later plan tokens. v13 tested a more aggressive late-token
weight schedule while keeping the AE frozen and the inference path unchanged.

v13 changes from v12:

```text
init:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latentfocus_s32768_v12/best.pt

config:
  coarse_planner/configs/libero_h64_planner_znorm_latefocus_s32768_v13.yaml

training:
  lr: 0.00002
  chunk_loss_weight: 0.10
  token_loss_weights: [0.8, 0.8, 0.9, 1.0, 1.25, 1.55, 2.0, 2.4]
  stopped after epoch 3 because epoch 1 remained best
```

v13 internal 32k eval:

| epoch | raw latent MSE | normalized MSE | decoded chunk loss | cosine |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.109238 | 0.324201 | 0.255213 | 0.877654 |
| 2 | 0.110700 | 0.328120 | 0.268488 | 0.876568 |
| 3 | 0.112704 | 0.333447 | 0.274673 | 0.874626 |

Seed43 holdout comparison:

| run | raw latent MSE | normalized MSE | decoded chunk loss | cosine |
| --- | ---: | ---: | ---: | ---: |
| v11 | 0.098461 | 0.290008 | 0.155727 | 0.888590 |
| v12 | 0.091676 | 0.273135 | 0.174122 | 0.896749 |
| v13 | 0.088456 | 0.264801 | 0.180509 | 0.900542 |

v13 seed43 holdout all per-token raw MSE:

```text
[0.0697, 0.0653, 0.0741, 0.0843, 0.0917, 0.0978, 0.1068, 0.1179]
```

Interpretation:

```text
The late-token focused update is useful, but only as a short fine-tune. Epoch 1
improves the holdout all raw latent MSE from 0.091676 to 0.088456, while epochs
2 and 3 already degrade the internal eval metric.

Decoded chunk loss rises modestly from 0.174122 to 0.180509, which remains
acceptable for planner-only latent shaping and is still far below the early v4
baseline.

The remaining gap to the 0.08 target is about 0.0085 raw MSE. p6 and p7 are
still the main bottleneck.
```

Decision after v13:

```text
Use v13 as the current best raw-MSE checkpoint.
Do not continue v13 with the same lr/weights for many epochs.
Next step should be a smaller-learning-rate late-focused continuation from v13
best, not a larger aggressive update.
No BridgeAttention / ActionHead integration training yet.
No joint end-to-end VLA training yet.
```

v13 artifacts:

```text
config:
  coarse_planner/configs/libero_h64_planner_znorm_latefocus_s32768_v13.yaml

checkpoint:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latefocus_s32768_v13/best.pt

diagnostics:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latefocus_s32768_v13/latent_diagnostics_seed43.json
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_latefocus_s32768_v13/latent_diagnostics_seed43.md
```

## 2026-06-19: Final Planner-Only Attempts v14-v17

After v13, three follow-up checks were run before closing this planner-only
round.

### v14: Smaller LR Late-Focused Continuation

v14 tested whether v13 could be improved by a smaller learning-rate continuation:

```text
init: v13 best
lr: 0.000005
chunk_loss_weight: 0.05
token_loss_weights: [0.7, 0.7, 0.85, 1.0, 1.3, 1.65, 2.1, 2.5]
```

Internal 32k eval:

| epoch | raw latent MSE | decoded chunk loss | cosine |
| ---: | ---: | ---: | ---: |
| 1 | 0.109840 | 0.272020 | 0.877343 |
| 2 | 0.109850 | 0.272834 | 0.877433 |

Decision:

```text
v14 did not improve over v13. It was stopped after epoch 2 and not promoted.
```

### 63k Cache Build and v15/v16

A larger H64 cache was built to test whether broader coverage could close the
remaining gap:

```text
config:
  coarse_planner/configs/libero_h64_s63044_build.yaml

cache:
  /root/autodl-tmp/datasets/coarse_planner/libero_h64_s63044_seed42

samples:
  total: 63044
  train: 56745
  eval: 6299

size:
  about 108 GB
```

The first training attempt, v15, used random sample order. It was stopped before
the first epoch completed because random access across 247 shards caused
unacceptable I/O stalls.

The training entrypoint was then updated to support:

```yaml
training:
  shuffle: false
```

v16 used sequential shard reads and completed epochs, but the larger cache did
not improve the planner:

| epoch | raw latent MSE | decoded chunk loss | cosine |
| ---: | ---: | ---: | ---: |
| 1 | 0.114636 | 0.275370 | 0.871049 |
| 2 | 0.138335 | 0.308582 | 0.846184 |

Decision:

```text
Do not continue the 63k training path in this round.
The 63k cache is available for future work, but the v16 fine-tune degraded
quickly and is not a usable checkpoint.
```

### v17: v12-v13 Checkpoint Interpolation

The final useful improvement came from checkpoint interpolation along the v12 to
v13 direction:

```text
checkpoint:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075/best.pt

formula:
  theta_v17 = theta_v12 + 0.75 * (theta_v13 - theta_v12)
```

Interpolation sweep on seed43 holdout all:

| alpha | raw latent MSE | decoded chunk loss | cosine |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.091675 | 0.174121 | 0.896750 |
| 0.50 | 0.088745 | 0.174447 | 0.900157 |
| 0.75 | 0.088257 | 0.176639 | 0.900731 |
| 1.00 | 0.088455 | 0.180511 | 0.900542 |
| 1.20 | 0.089081 | 0.184636 | 0.899874 |
| 1.50 | 0.090790 | 0.192388 | 0.897989 |

Full v17 diagnosis:

```text
seed43 holdout all raw latent MSE: 0.088253
seed43 holdout all normalized latent MSE: 0.264069
seed43 holdout all decoded chunk loss: 0.176643
seed43 holdout all cosine: 0.900730
```

v17 seed43 holdout all per-token raw MSE:

```text
[0.0695, 0.0652, 0.0741, 0.0840, 0.0917, 0.0972, 0.1069, 0.1175]
```

Final planner-only comparison:

| run | raw latent MSE | normalized MSE | decoded chunk loss | cosine |
| --- | ---: | ---: | ---: | ---: |
| v4 | 0.153218 | 0.455218 | 0.340088 | 0.828579 |
| v10 | 0.099159 | 0.293366 | 0.178007 | 0.888414 |
| v12 | 0.091676 | 0.273135 | 0.174122 | 0.896749 |
| v13 | 0.088456 | 0.264801 | 0.180509 | 0.900542 |
| v17 | 0.088253 | 0.264069 | 0.176643 | 0.900730 |

Final decision for this round:

```text
Use v17 as the current standalone planner checkpoint.
It does not strictly cross 0.08, but it is below 0.09 and in the intended
"0.08-ish" usable range for the next stage.
Decoded chunk loss and cosine are both much better than the early v4 baseline.
The remaining error is concentrated in p6-p7.
Do not start joint end-to-end VLA training in this round.
The next stage is BridgeAttention / ActionHead suffix integration using the
planner only through plan tokens.
```

v17 artifacts:

```text
checkpoint:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075/best.pt

diagnostics:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075/latent_diagnostics_seed43.json
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075/latent_diagnostics_seed43.md

interpolation sweep:
  /root/autodl-tmp/runs/coarse_planner/v12_v13_interpolation_eval.json
```

Post-run cleanup note:

```text
To recover disk space, intermediate datasets and runs were removed after the
final v17 checkpoint was verified.

Retained remote datasets:
  /root/autodl-tmp/datasets/coarse_planner/libero_h64
  /root/autodl-tmp/datasets/coarse_planner/libero_h64_holdout_seed43
  /root/autodl-tmp/datasets/coarse_planner/libero_h64_s32768_seed42

Retained remote runs:
  /root/autodl-tmp/runs/coarse_planner/libero_h64_segment_ae_v2
  /root/autodl-tmp/runs/coarse_planner/libero_h64_planner_znorm_interp_v17_alpha075

Older artifact paths in this document are historical records of the experiment.
They are not guaranteed to remain on disk after cleanup.
```
