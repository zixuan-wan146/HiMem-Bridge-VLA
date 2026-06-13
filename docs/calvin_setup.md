# CALVIN Setup for HiMem-Bridge-VLA

This setup keeps CALVIN paths project-relative and adapts it to the existing HiMem-Bridge-VLA
simulation dataset loader. If the repository itself is on a data disk, these relative paths stay on
that disk.

## Expected Paths

```bash
datasets/calvin/
  lerobot/task_D_D/                       # LeRobot-format CALVIN training data
  original/task_D_D/                      # optional original CALVIN data
  annotations/task_D_D_boundaries.jsonl   # generated boundary sidecar
```

The dataset config is:

```bash
configs/datasets/calvin.yaml
```

It enables `adapter: calvin`, which accepts both canonical LeRobot keys
(`observation.state`, `action`, `observation.images.*`) and the common
RoboTron/StarVLA CALVIN keys (`state`, `actions`, `image`, `wrist_image`).

## Boundary Sidecar

Generate the memory/boundary sidecar from the original CALVIN annotation file:

```bash
python scripts/prepare_calvin_boundaries.py \
  --auto-lang-ann datasets/calvin/original/task_D_D/training/lang_annotations/auto_lang_ann.npy \
  --output datasets/calvin/annotations/task_D_D_boundaries.jsonl
```

If using a LeRobot D-D conversion where each episode is one task segment,
generate per-episode boundaries directly from LeRobot metadata:

```bash
python scripts/prepare_calvin_boundaries.py \
  --lerobot-root datasets/calvin/lerobot/task_D_D \
  --output datasets/calvin/annotations/task_D_D_boundaries.jsonl
```

The sidecar stores segment-level labels:

```json
{"segment_id": 0, "start": 10, "end": 50, "task": "open_drawer", "skill_id": 3, "language": "open drawer"}
```

The dataset adapter derives per-frame training labels online:

```text
boundary = frame_idx == segment_end
progress = (frame_idx - segment_start) / (segment_end - segment_start)
skill_id = task id from CALVIN language.task
```

No future frames, future video, or optical flow are used.

## Validation

After the LeRobot-format data is present:

```bash
python scripts/validate_training_dataset.py \
  --dataset-config configs/datasets/calvin.yaml \
  --dataset-base-dir .
```

For a metadata-only check before videos are available:

```bash
python scripts/validate_training_dataset.py \
  --dataset-config configs/datasets/calvin.yaml \
  --dataset-base-dir . \
  --no-require-videos
```

## Training Entry

Use the normal HiMem-Bridge-VLA train script with the CALVIN config:

```bash
python scripts/train.py \
  --dataset_config_path configs/datasets/calvin.yaml \
  --dataset_config_base_dir . \
  --horizon 8 \
  --per_action_dim 7 \
  --state_dim 8
```

## Evaluation Entry

CALVIN eval is separated from training code:

```text
evaluations/calvin/
  calvin_client.py          websocket rollout client
  calvin_client_config.py   environment/profile config
  calvin_action_protocol.py server action parsing and gripper mapping
  calvin_eval_summary.py    sequence metrics and result JSON
```

The eval client expects an installed CALVIN checkout with `calvin_models/conf` and the official
ABC->D validation dataset. By default it looks under:

```bash
HIMEM_CALVIN_ROOT=datasets/calvin/runtime
HIMEM_CALVIN_DATASET_PATH=datasets/calvin/runtime/dataset/task_ABC_D
```

Start the HiMem-Bridge-VLA server first, then run a dry-run to inspect resolved paths:

```bash
HIMEM_CALVIN_DRY_RUN=1 \
HIMEM_CALVIN_PROFILE=configs/calvin_profiles/smoke.env \
scripts/run_calvin_eval.sh
```

Run one smoke sequence:

```bash
HIMEM_CALVIN_PROFILE=configs/calvin_profiles/smoke.env \
HIMEM_CALVIN_RUN_DIR=run_outputs/himem_runs/calvin_smoke_001 \
CALVIN_PYTHON=run_outputs/calvin_env/bin/python \
scripts/run_calvin_eval.sh
```

Run the default full evaluation:

```bash
HIMEM_CALVIN_PROFILE=configs/calvin_profiles/full_eval.env \
HIMEM_CALVIN_RUN_DIR=run_outputs/himem_runs/calvin_eval_001 \
CALVIN_PYTHON=run_outputs/calvin_env/bin/python \
scripts/run_calvin_eval.sh
```

The run directory layout is `logs/`, `videos/`, `results/`, and `run_manifest.json`. Videos are
disabled by default because 1000 CALVIN sequences can produce a large artifact set; enable them
with `HIMEM_CALVIN_SAVE_VIDEO=1`.

Important eval knobs:

- `HIMEM_CALVIN_RESET_MEMORY_SCOPE=sequence` resets HiMem once at the start of each five-subtask
  CALVIN sequence. Use `subtask` only for a stricter per-instruction reset ablation.
- `HIMEM_CALVIN_GRIPPER_MODE=openvla` maps a model gripper output in `[0,1]` to CALVIN's signed
  gripper action. Use `passthrough` only if the checkpoint already denormalizes directly to the
  CALVIN environment sign convention.
