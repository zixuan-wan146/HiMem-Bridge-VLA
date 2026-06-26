---
license: cc-by-4.0
library_name: pytorch
pipeline_tag: robotics
tags:
  - robotics
  - vision-language-action
  - vla
  - memory
  - fast-weights
  - constant-vram
  - imitation-learning
  - test-time-training
---

# AURA: Action-Gated Memory for Robot Policies at Constant VRAM

<p align="center">
  <img src="https://huggingface.co/Kaikaku/aura/resolve/main/aura.gif" alt="AURA demo: a robot memory that never grows" width="640"/>
</p>

<p align="center"><a href="https://huggingface.co/Kaikaku/aura/resolve/main/aura.mp4">Watch the full video (MP4)</a></p>

AURA (Action-Utility Recurrent Adaptive Memory) is a bounded, recurrent memory layer for robot and vision-language-action (VLA) policies. Instead of a KV-cache that grows with every step, AURA carries one fixed-size fast-weight matrix W and a learned surprise gate that writes to it only when the current observation would change the policy's next action. The inference state is constant at 4,224 bytes for any episode length, so VRAM is O(1) in the horizon.

- Paper: https://arxiv.org/abs/2606.02775
- Interactive demo: https://huggingface.co/spaces/KAIKAKU/aura-demo
- Author: Josef Chen, KAIKAKU

## What it is

A policy that remembers a long episode usually pays for it with a state that grows every step (a KV-cache, O(T) memory and O(T) attention). AURA replaces that with a single fixed-shape fast-weight matrix updated test-time-training style. A learned gate reads each observation and opens only on action-relevant surprise (high inner prediction error against the current W, trained against a closed-loop action-prediction objective), so writes are sparse and the carried state never grows.

## Key results (measured, from the paper)

- Inference state is a constant 4,224 bytes (d_k = d_v = 32, batch 1, fp32), 6,061x smaller than a same-dimension growing KV-cache at 100,000 steps.
- Accuracy parity with the best fixed-size O(1) baseline at 4.98 to 9.19x fewer memory writes (5.19 to 6.13x on the hard, non-saturated task; the 9.19x point is N=64, near-saturated, n=3 seeds).
- The gate is selective: it fires 2.61x harder on real events than on distractors (write-prob 0.829 vs 0.318; single trained seed).
- Budget-matched random and periodic write schedules collapse to about 0.366, so the gain is the action-utility signal, not the write budget; a dense GRU that writes every step collapses to about 0.25 on the hard task (chance 0.125).
- Real policy: on OpenVLA-OFT 7B / LIBERO-Long (n=60 episodes/arm), AURA matches the ungated base (0.233 vs 0.233) and slightly exceeds an always-write KV residual (0.217) at 7.0x fewer writes (504 vs 3,541) and constant 4,224-byte state, while the KV arm grows to 906,496 bytes over the 520-step evaluation.

The defensible claim is parity of task success at a fraction of the writes and constant VRAM, not improved task success.

## How to use

AURA is a drop-in memory layer for a policy backbone. Replace the growing KV-cache with `AuraMemLayer`, feed per-frame features, and read the action plus the binary write gate.

```python
from memory_layer import AuraMemLayer, MemoryConfig

layer = AuraMemLayer(MemoryConfig(d_model=4096, d_key=32, d_val=32, d_action=7))
out = layer(z)                 # z: [batch, steps, d_model] features from your backbone
action = out["action"]         # predicted action chunk
wrote  = out["g"]              # 1 when AURA chose to write this step
print(layer.memory_bytes(1))   # 4224, constant for any horizon
```

Train the gate with two terms: an action information-bottleneck loss (rewarding writes that change the next action) plus a write-sparsity term (penalizing over-writing). See the paper for the exact objective and hyper-parameters.

## Files in this repo

- The trained checkpoint and `memory_layer.py` (the module above).
- Note on configuration: the headline 4,224-byte figure is the sweep config (d_k = d_v = 32). If a checkpoint ships with a different key/value dimension, its inference-state size scales as (d_k * d_v + d_v) * batch * 4 bytes; cite the matching config when quoting bytes.

## Limitations

- Parity, not superiority: accuracy gaps versus the best fixed-size baseline are within confidence intervals; AURA does not improve task success, it matches it at far fewer writes and constant VRAM.
- "O(1)" refers to the inference state; training-time activation memory is O(T), and the model weights are a separate, shared footprint.
- The paper instantiates an approximate-information-state (AIS) value-loss bound as a methodology demonstration; at this scale the instantiated bound is vacuous (the informative quantity is the small measured action-prediction error, epsilon mean 0.0021, q95 0.0076), and no formal optimality guarantee is claimed.
- Regime: a growing KV-cache is fine in datacenter batch-N serving (amortized across requests, resets between them); AURA targets the batch-1 embodied regime, where a single agent runs one continuous, non-resetting episode and the cache would grow without bound.

## Citation

```bibtex
@misc{chen2026aura,
  title         = {AURA: Action-Gated Memory for Robot Policies at Constant VRAM},
  author        = {Chen, Josef},
  year          = {2026},
  eprint        = {2606.02775},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.02775}
}
```

Not affiliated with NVIDIA or the OpenVLA authors; model names are trademarks of their respective owners.
