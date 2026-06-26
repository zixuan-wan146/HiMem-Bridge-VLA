# Reference Repositories

This folder stores official non-empty code or model repositories related to the reference papers.

## Primary Local Repos

- `memer/`
  - Official MemER repository.
  - Use for keyframe selection, episodic keyframe clustering, high-level VLM prompt contract, and the subtask interface to a low-level policy.
- `robomme_benchmark/`
  - Official RoboMME benchmark repository.
  - Use for memory task taxonomy and evaluation protocol.
- `robomme_policy_learning/`
  - Official RoboMME MME-VLA policy learning repository, cloned with `GIT_LFS_SKIP_SMUDGE=1`.
  - Use for concrete implementations of symbolic, perceptual, and recurrent memory variants, plus context/modulation/expert integration mechanisms.

## Checked But No Official Non-Empty Repo Found Yet

- MEM / `2603.03596`
  - Paper and project page are available, but no official code repository was found in the current pass.
- EchoVLA / `2511.18112`
  - Paper is available, but no official code repository was found in the current pass.

Naming note:

```text
MEM   = Physical Intelligence multi-scale memory paper; no official code found here.
MemER = Stanford experience retrieval paper; official code is in `memer/`.
```

## Secondary / Parked

- `aura/`
  - Hugging Face repository for AURA, cloned with `GIT_LFS_SKIP_SMUDGE=1`.
  - Kept locally only as a secondary reference. It is not part of the current main design baseline.
