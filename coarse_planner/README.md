# Standalone Coarse Planner

This module trains the Coarse Planner before it is attached to the full HiMem-Bridge-VLA model.

The intended pipeline is:

```text
cache InternVL3/VLA tokens per frame
build planner feature cache
train CoarsePlanner only
inspect loss and coarse trajectory quality
export checkpoint
load checkpoint into HiMemBridgeVLA and finetune Bridge/ActionHead
```

The planner still does not read memory. It only consumes:

```text
vlm_tokens: [L, D]
state:      [state_dim]
```

and predicts:

```text
plan_tokens:    [K, D]
coarse_actions: [K, action_dim]
```

## Feature Cache Format

Raw input sources for `build_dataset.py` are `.pt` or `.npz` files containing either one episode dict or:

```python
{
    "episodes": [
        {
            "episode_id": "task/episode_000",
            "vlm_tokens": ...,  # [T, L, D]
            "states": ...,      # [T, state_dim]
            "actions": ...,     # [T, action_dim]
            "frame_index": ..., # optional [T]
        },
    ],
}
```

The builder writes sharded planner samples:

```text
manifest.json
train/planner_samples_00000.pt
eval/planner_samples_00001.pt
```

Each sample contains `vlm_tokens`, `state`, `coarse_actions`, and `coarse_action_mask`.

## Smoke Dataset

Use the synthetic path only to verify the data/training pipeline:

```bash
python -m coarse_planner.build_dataset \
  --config coarse_planner/configs/synthetic_smoke.yaml \
  --synthetic-smoke \
  --output /root/autodl-tmp/datasets/coarse_planner/smoke
```

## From SimulationDataset

For real demonstrations, first export the exact token source the planner will see in the main model:

```bash
python -m coarse_planner.build_from_simulation \
  --config coarse_planner/configs/calvin_abc_d_smoke.yaml \
  --dry-run
```

Then build a small feature cache:

```bash
python -m coarse_planner.build_from_simulation \
  --config coarse_planner/configs/calvin_abc_d_smoke.yaml \
  --device cuda \
  --max-samples 128
```

The default token source is `feature.source=fused`, matching the main training path that calls InternVL3 with `return_cls_only=False`.
For layer ablations, set:

```yaml
feature:
  source: hidden_state
  hidden_state_layer: deep
```

State is stored separately in the cache and fused by `CoarsePlanner` through its state projection token. Do not pre-concatenate raw state into `vlm_tokens`.

## LIBERO Horizon Ablation

The LIBERO planner warm-up path uses the Evo-1 style InternVL3-1B embedder:

```text
OpenGVLab/InternVL3-1B
language_model.layers[:14]
return_cls_only = false
```

The cached `vlm_tokens` are the 14-layer VLM fused tokens for observation and language only. LIBERO proprioception is built as:

```text
state_raw = [obs/ee_states(6), obs/gripper_states(2)]
state = minmax_normalize(state_raw, Evo1_LIBERO/norm_stats.json)
state = pad_to_24(state)
```

The first horizon ablation keeps the chunk size fixed at 8 control steps:

```text
H=32, K=4
H=48, K=6
H=64, K=8
```

All three caches reuse the same sampled `(task, demo, timestep)` manifest, so the input distribution is identical and only the target horizon changes.

Dry-run the cache plan:

```bash
python -m coarse_planner.build_from_libero \
  --config coarse_planner/configs/libero_horizon_ablation_build.yaml \
  --dry-run
```

Build all three feature caches in one InternVL pass:

```bash
python -m coarse_planner.build_from_libero \
  --config coarse_planner/configs/libero_horizon_ablation_build.yaml \
  --device cuda
```

Run the full H32/H48/H64 standalone training and report:

```bash
COARSE_PLANNER_BATCH_SIZE=512 \
COARSE_PLANNER_EPOCHS=8 \
scripts/run_coarse_planner_libero_ablation.sh
```

The train logs include `cuda_peak_reserved_gb`; adjust `COARSE_PLANNER_BATCH_SIZE` so the 4090 run reserves about 20 GB without OOM.

## Training

```bash
python -m coarse_planner.train \
  --config coarse_planner/configs/default.yaml \
  --run-dir /root/autodl-tmp/runs/coarse_planner/libero_warmup
```

## Export

```bash
python -m coarse_planner.export \
  --checkpoint /root/autodl-tmp/runs/coarse_planner/libero_warmup/best.pt \
  --output /root/autodl-tmp/checkpoints/coarse_planner/libero_warmup.pt
```
