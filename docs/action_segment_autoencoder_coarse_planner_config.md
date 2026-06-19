# Action Segment Autoencoder + Coarse Planner Config

Date: 2026-06-18

Status: first LIBERO-focused training configuration. This document fixes only the
two pretraining components:

1. Action Segment Autoencoder
2. Coarse Planner

The goal is to first learn a stable action-segment latent space, then train the
Coarse Planner to predict future action intent tokens in that latent space.

## Global Timing

First version:

```yaml
global:
  benchmark: libero

  per_action_dim: 7
  motion_dim: 6
  gripper_dim: 1

  action_head_horizon: 16      # H_e
  planning_horizon: 64         # H_p
  num_plan_tokens: 8           # K
  token_span_steps: 8          # c = H_p / K
  tokens_per_action_chunk: 2   # q = H_e / c
```

Required constraints:

```text
H_p = 64
K = 8
c = H_p / K = 8

H_e = 16
q = H_e / c = 2

H_p % K == 0
H_e % c == 0
```

LIBERO action layout:

```yaml
action_layout:
  action_dim: 7
  motion_indices: [0, 1, 2, 3, 4, 5]
  gripper_indices: [6]
```

Before AE training:

- Motion dimensions use the current project's continuous action normalization.
- Gripper targets must be explicitly binarized before segment construction:

```text
g in {0, 1}
```

BCE targets for gripper must never contain continuous values.

## Component A: Action Segment Autoencoder

### Data Target

For each anchor time, slice the next 64 low-level actions into 8 segments:

```text
A[t : t + H_p - 1] -> [A_0, A_1, ..., A_7]
A_i: [8, 7]
```

Standalone AE training sees one segment at a time:

```text
A_i -> z_i -> A_hat_i
```

The AE is action-only. It must not condition on state, VLM tokens, language, memory,
or task metadata.

### Network Config

Use a deterministic Transformer autoencoder. Do not use VAE/KL in the first
version.

```yaml
action_segment_autoencoder:
  model_type: deterministic_transformer_autoencoder

  segment_length: 8
  action_dim: 7
  motion_dim: 6
  gripper_dim: 1

  hidden_dim: 128
  latent_dim: 128
  num_heads: 4
  dropout: 0.05
  activation: gelu
  norm_first: true

  encoder:
    input_proj: Linear(7, 128)
    use_cls_token: true
    position_embedding:
      type: learned
      length: 9
      dim: 128
    transformer:
      num_layers: 2
      num_heads: 4
      ffn_dim: 512
      dropout: 0.05
    latent_head:
      - LayerNorm(128)
      - Linear(128, 128)

  decoder:
    latent_proj: Linear(128, 128)
    time_queries:
      type: learned
      length: 8
      dim: 128
    position_embedding:
      type: learned
      length: 9
      dim: 128
    transformer:
      num_layers: 2
      num_heads: 4
      ffn_dim: 512
      dropout: 0.05
    output_heads:
      motion_head: Linear(128, 6)
      gripper_head: Linear(128, 1)
```

Encoder output:

```text
z_i = E_phi(A_i)
z_i: [128]
```

Decoder input:

```text
[z_i, q_0, q_1, ..., q_7]
```

`q_j` are learned timestep queries. The decoder can be implemented as an
encoder-style self-attention stack over one latent token plus 8 timestep queries.
Only the 8 timestep-query outputs are passed to the motion and gripper heads.

Decoder output:

```text
A_hat_i.motion:  [8, 6]
A_hat_i.gripper: [8, 1] logits
```

### AE Loss

```yaml
action_segment_autoencoder_loss:
  motion_loss: huber
  huber_delta: 1.0
  gripper_loss: bce_with_logits
  gripper_weight: 2.0

  distance_loss_weight: 0.1
  dct_low_frequency: 4
  endpoint_distance_weight: 0.5
  gripper_distance_weight: 0.25
```

Reconstruction:

```text
L_rec =
  Huber(A_hat.motion, A.motion)
  + 2.0 * BCEWithLogits(A_hat.gripper, A.gripper)
```

Action-segment distance:

```text
d_A(A_i, A_j) =
  ||DCT_1:4(A_i.motion) - DCT_1:4(A_j.motion)||_2
  + 0.5 * ||A_i.end.motion - A_j.end.motion||_2
  + 0.25 * 1[g_i != g_j]
```

Latent distance:

```text
d_z(z_i, z_j) = ||z_i - z_j||_2
```

Distance preservation:

```text
L_dist = Huber(
  d_z / (mean(d_z) + eps),
  d_A / (mean(d_A) + eps)
)
```

Pairwise distances are computed inside each training batch.

Total AE loss:

```text
L_AE = L_rec + 0.1 * L_dist
```

### AE Training

```yaml
action_segment_autoencoder_training:
  optimizer: AdamW
  learning_rate: 0.0003
  weight_decay: 0.0001
  batch_size: 512
  max_epochs: 80
  warmup_ratio: 0.05
  scheduler: cosine
  grad_clip_norm: 1.0
  amp: true

  validation_split: 0.05
  early_stop_patience: 12
  save_best_metric: val_total_loss

  checkpoint:
    save_encoder: true
    save_decoder: true
    save_config: true
    save_action_normalizer: true
```

Metrics:

```yaml
action_segment_autoencoder_metrics:
  - train_total_loss
  - train_rec_loss
  - train_dist_loss
  - val_total_loss
  - val_motion_huber
  - val_gripper_bce
  - val_gripper_accuracy
  - val_distance_spearman
```

After training:

```text
freeze(E_phi, D_theta)
```

## Component B: Coarse Planner

### Data Target

For each planner sample:

```text
A[t : t + H_p - 1] -> [A_0, ..., A_7]
z_i* = E_phi(A_i)
```

Planner input:

```text
H_t: VLM hidden tokens
s_t: robot state
```

LIBERO profile:

```yaml
coarse_planner_input:
  vlm_hidden_dim: 896
  state_dim: 7
```

`state_dim: 7` is the LIBERO planner profile when the planner consumes the
normalized 7D LIBERO proprio state directly. If a training path pads state before
the planner, this field must match the actual planner input tensor dimension.

Planner output:

```text
P_t = [p_0, ..., p_7]
P_t: [8, 896]
z_hat_i = W_z(p_i)
z_hat_i: [128]
```

### Network Config

```yaml
coarse_planner:
  model_type: query_cross_attention

  input:
    vlm_hidden_dim: 896
    state_dim: 7

  output:
    num_plan_tokens: 8
    plan_token_dim: 896
    latent_dim: 128

  planner_core:
    hidden_dim: 896
    num_layers: 4
    num_heads: 8
    ffn_mult: 4
    ffn_dim: 3584
    dropout: 0.05
    activation: gelu
    norm_first: true

  plan_queries:
    shape: [8, 896]
    init: normal_std_0.02

  state_projection:
    - LayerNorm(7)
    - Linear(7, 896)
    - GELU
    - Linear(896, 896)

  context:
    tokens: concat(vlm_tokens, state_token)

  output_norm:
    - LayerNorm(896)

  latent_prediction_head:
    - LayerNorm(896)
    - Linear(896, 512)
    - GELU
    - Dropout(0.05)
    - Linear(512, 128)
```

Deprecated components must not be kept:

```yaml
deprecated:
  remove:
    - coarse_action_head
    - coarse_actions
    - coarse_action_mask
    - smooth_l1_coarse_action_loss
```

### Planner Pretraining Loss

Standalone Coarse Planner pretraining supervises all 8 plan tokens. Do not apply
suffix masking to the loss in this stage.

Reason: suffix masking only changes which planner tokens receive latent loss. It
does not train BridgeAttention or ActionHead to consume variable-length suffixes.
If used as the standalone planner loss, it overweights later tokens:

```text
u=0 supervises p0..p7
u=2 supervises p2..p7
u=4 supervises p4..p7
u=6 supervises p6..p7
```

Planner pretraining should instead make every token a stable action-segment intent
token:

```text
L_z = (1 / K) * sum_i ||z_hat_i - z_i*||_2^2
A_hat_i = D_theta(z_hat_i)
L_chunk = (1 / K) * sum_i L_rec(A_hat_i, A_i)
L_planner = L_z + 0.25 * L_chunk
```

Config:

```yaml
coarse_planner_loss:
  latent_mse_weight: 1.0
  decoded_chunk_loss_weight: 0.25
  loss_on_active_suffix_only: false

  motion_loss: huber
  huber_delta: 1.0
  gripper_loss: bce_with_logits
  gripper_weight: 2.0
```

Suffix metrics may be logged as diagnostics, but they must not control the
standalone planner loss:

```yaml
coarse_planner_suffix_diagnostics:
  enabled: true
  consumed_tokens_set: [0, 2, 4, 6]
  metrics:
    - val_latent_mse_u0
    - val_latent_mse_u2
    - val_latent_mse_u4
    - val_latent_mse_u6
```

### Planner Training

```yaml
coarse_planner_training:
  freeze_vlm: true
  freeze_action_segment_encoder: true
  freeze_action_segment_decoder: true

  train_modules:
    - coarse_planner
    - latent_prediction_head

  optimizer: AdamW
  learning_rate: 0.0001
  weight_decay: 0.00001
  batch_size: 32
  grad_accum_steps: 4
  effective_batch_size: 128
  max_steps: 30000
  warmup_steps: 1500
  scheduler: cosine
  grad_clip_norm: 1.0
  amp: true

  validation_interval_steps: 1000
  checkpoint_interval_steps: 1000
  save_best_metric: val_planner_loss
```

Metrics:

```yaml
coarse_planner_metrics:
  - train_planner_loss
  - train_latent_mse
  - train_decoded_chunk_loss
  - val_planner_loss
  - val_latent_mse
  - val_decoded_motion_huber
  - val_decoded_gripper_bce
  - val_latent_distance_spearman
  - val_latent_mse_u0
  - val_latent_mse_u2
  - val_latent_mse_u4
  - val_latent_mse_u6
```

## Component Interface

AE checkpoint:

```yaml
component_interface:
  action_segment_autoencoder_ckpt: checkpoints/action_segment_ae/best.pt

  planner_target:
    z_target: encoder(action_segment)

  planner_auxiliary_decode:
    decoded_action_segment: decoder(predicted_z)
```

Training flow:

```text
A_i -> E_phi -> z_i*
(H_t, s_t) -> CoarsePlanner -> P_t
p_i -> W_z -> z_hat_i
z_hat_i -> D_theta -> A_hat_i
```

Inference flow:

```text
(H_t, s_t) -> CoarsePlanner -> P_t
P_active -> BridgeAttention -> ActionHead
```

Not used at inference:

```text
E_phi
D_theta
W_z
```

## Next-Stage Suffix Training

Plan suffix training belongs to BridgeAttention / ActionHead integration, not
standalone Coarse Planner pretraining.

The integration-stage training state should use an anchor/current split:

```text
P_tau = CoarsePlanner(H_tau, s_tau)
u in {0, 2, 4, 6}
t = tau + u * c
P_active = P_tau[u:K]
ActionHead(H_t, s_t, P_active, M_t)
```

Runtime queue semantics:

```text
N <- N + executed_control_steps
u = floor(N / c)
r = N mod c
P_active = P_tau[u:K]
```

Integration-stage config:

```yaml
plan_token_queue:
  enabled: true
  consume_by_executed_control_steps: true
  token_span_steps: 8
  tokens_per_action_chunk: 2
  use_plan_token_mask: true
  suffix_sampling: per_sample
  consumed_tokens_set: [0, 2, 4, 6]
```

By the end of the two-component pretraining stage, the required guarantees are:

1. AE latents reconstruct action segments.
2. Distinct action segments remain separable in latent space.
3. Coarse Planner predicts all 8 future action-segment latents from current VLM
   tokens and state.
4. Suffix metrics for `u=0,2,4,6` are observable, but suffix behavior is trained
   in the later BridgeAttention / ActionHead integration stage.
