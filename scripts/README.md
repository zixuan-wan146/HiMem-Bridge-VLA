# Scripts

Scripts are grouped by responsibility. Prefer these entry points over hand-written command chains. The active path no longer includes transition-trigger conversion, runtime, or trace-summary scripts.

## Quality Gates

- `check_repo.sh`: consolidated local gate for requirements, runtime checks, unit tests, config validation, smoke planning, compile checks, and whitespace checks.
- `preflight.py`: repository, checkpoint, dataset, LIBERO result, and run-manifest checks.
- `audit_requirements.py`: dependency policy audit.
- `validate_bridge_himem_configs.py`: Bridge-HiMem YAML schema and `extends` validation.
- `validate_training_configs.py`: training profile validation.
- `validate_training_dataset.py`: dataset structure validation before training.
- `smoke_direct_bridge_inference.py`: deterministic direct bridge-attn action-head inference smoke. Defaults to the final H32/D896 architecture; use `--preset tiny` for quick CPU checks. Pass `--progress-planner-checkpoint <best.pt>` to smoke the progress planner checkpoint -> plan token -> direct bridge action path.
- `smoke_direct_bridge_token_cache_training.py`: one-step direct bridge-attn training smoke over synthetic or cached replay-token batches. Use `--preset auto --manifest <cache>` to read a real cache and infer dimensions from the batch. Pass `--progress-planner-checkpoint <best.pt>` to smoke checkpoint-produced plan tokens during the training loss/backward path.

## Training And Model Assets

- `train_stage1.py`: active Stage1 trajectory-window token-cache training launcher. The implementation lives under `himem_bridge_vla/training/stage1/`; the script only starts the CLI.
- `train.py`: legacy mixed training entry for historical paths. It does not own active Stage1 trajectory-window training.
- `himem_server.py`: websocket inference server.
- `download_libero_checkpoint.sh`: download the LIBERO checkpoint to the data disk.
- `start_himem_server.sh`: start the HiMem-Bridge-VLA websocket server with checkpoint preflight.

## Benchmark Data

- `download_rmbench_tasks.py`: download the official nine RMBench `demo_clean` task folders.
- `inspect_benchmarks.py`: inspect local LIBERO, LIBERO-Plus, and RMBench assets and emit a JSON inventory.
- `build_libero_memory_replay_index.py`: write a deterministic LIBERO JSONL index for current frames, short-memory history, and future action chunks. The index keeps `long_capacity=0`; progress-state long memory is not stored in replay rows.
- `build_rmbench_norm_stats.py`: compute RMBench min/max stats for state normalization and action denormalization.
- `build_rmbench_memory_replay_index.py`: write a deterministic RMBench JSONL index for current frames, short-memory history, and future action chunks. The index keeps `long_capacity=0`; progress-state long memory is not stored in replay rows.
- `build_memory_replay_token_cache.py`: encode replay-index frames into replay-token shards plus a manifest. Use `--encoder internvl3 --include-vlm-hidden-states --hidden-state-layers 3 6 9 12` for final direct bridge training caches; use `--encoder image_stats` only for IO smoke tests.
- `plan_rmbench_eval.py`: inspect RMBench eval prerequisites and write direct/socket eval commands for the official `script/eval_policy.py` stack.
- `install_rmbench_policy_adapter.py`: install the checked-in `HiMemBridgeVLA` policy adapter into an official RMBench checkout.
- `run_rmbench_eval.sh`: reproducible RMBench eval wrapper. It installs the adapter, writes a run manifest, writes a command plan, then runs official direct eval task by task.

Token caches are read with `MemoryTokenCacheDataset` and `collate_memory_token_cache_samples`. They expose current visual tokens, optional current VLM hidden-state layers, recent short visual tokens, `short_steps`, `short_mask`, state, and future action chunks for IO checks and direct bridge training.

There are two different cache families:

- `*_progress_vl_embedding_warmup_cache`: pooled VL embedding windows for progress-state planner warm-up.
- `memory_replay_visual_token_cache`: replay-token shards for short memory and direct bridge-attn action-head training. Final direct bridge caches include `current_hidden_states`.

Do not pass progress warm-up caches into the direct bridge token-cache smoke; the script validates the manifest format and will reject them.

For active Stage1 direct bridge-attn training from cached replay tokens, use `scripts/train_stage1.py` or `python -m himem_bridge_vla.training.stage1.cli`. The Stage1 loader uses trajectory windows from `MemoryTokenCacheTrajectoryDataset`, not random frame-level batches, so the frozen progress planner state is advanced chronologically through burn-in and loss windows.

Example:

```bash
python scripts/inspect_benchmarks.py \
  --data-root "$AUTODL_TMP" \
  --output run_outputs/benchmark_inventory.json \
  --allow-missing

python scripts/build_libero_memory_replay_index.py \
  --libero-root "$AUTODL_TMP/libero/datasets" \
  --output run_outputs/libero_memory_replay.jsonl

python scripts/build_rmbench_norm_stats.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --output run_outputs/rmbench_norm_stats.json \
  --metadata-output run_outputs/rmbench_norm_stats.metadata.json

python scripts/build_rmbench_memory_replay_index.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --output run_outputs/rmbench_memory_replay.jsonl

python scripts/build_memory_replay_token_cache.py \
  --benchmark LIBERO \
  --data-root "$AUTODL_TMP/libero/datasets" \
  --index run_outputs/libero_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/libero_memory_replay" \
  --encoder image_stats \
  --max-samples 2

python scripts/build_memory_replay_token_cache.py \
  --benchmark LIBERO \
  --data-root "$AUTODL_TMP/libero/datasets" \
  --index run_outputs/libero_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/libero_memory_replay_internvl3_hidden_l3_6_9_12" \
  --encoder internvl3 \
  --include-vlm-hidden-states \
  --hidden-state-layers 3 6 9 12 \
  --storage-dtype bfloat16 \
  --device cuda

python scripts/build_memory_replay_token_cache.py \
  --benchmark LIBERO \
  --data-root "$AUTODL_TMP/libero/datasets" \
  --index run_outputs/libero_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/libero_memory_replay_image_stats_hidden_smoke" \
  --encoder image_stats \
  --image-stats-hidden-dim 896 \
  --image-stats-tokens-per-view 32 \
  --include-vlm-hidden-states \
  --hidden-state-layers 3 6 9 12 \
  --max-samples 2 \
  --max-samples-per-shard 2 \
  --storage-dtype float32

python scripts/smoke_direct_bridge_token_cache_training.py \
  --preset auto \
  --manifest "$AUTODL_TMP/token_caches/libero_memory_replay" \
  --device cpu \
  --steps 1 \
  --batch-size 1

python scripts/smoke_direct_bridge_token_cache_training.py \
  --preset final \
  --manifest "$AUTODL_TMP/token_caches/libero_memory_replay_image_stats_hidden_smoke" \
  --device auto \
  --steps 1 \
  --batch-size 1 \
  --action-horizon 32 \
  --memory-entry-tokens 16 \
  --progress-planner-checkpoint "$AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt"

python scripts/smoke_direct_bridge_inference.py \
  --preset final \
  --device auto \
  --progress-planner-checkpoint "$AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt"

python scripts/plan_rmbench_eval.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --output run_outputs/rmbench_eval_plan.md \
  --mode direct \
  --tasks observe_and_pickup press_button

HIMEM_RMBENCH_TASKS=press_button \
HIMEM_RMBENCH_RUN_DIR=run_outputs/rmbench_smoke \
HIMEM_RMBENCH_PLAN_ONLY=1 \
HIMEM_SERVER_URI=ws://127.0.0.1:9000 \
bash scripts/run_rmbench_eval.sh
```

## LIBERO Runs

- `setup_libero_env.sh`: create or validate the LIBERO simulation environment.
- `run_libero_smoke.sh`: minimal smoke evaluation.
- `run_libero_eval.sh`: full LIBERO evaluation.
- `plan_libero_run.py`: generate reproducible server/eval/report commands before running.
- `init_libero_experiment.py`: create a tracked experiment skeleton.
- `libero_profile.sh`: safe parser for `configs/libero_profiles/*.env`.

## Reporting

- `write_libero_run_manifest.py`: write run metadata before evaluation starts.
- `write_rmbench_run_manifest.py`: write run metadata before RMBench evaluation starts.
- `summarize_libero_results.py`: summarize result JSON files.
- `check_libero_metrics.py`: gate candidate runs against thresholds or baseline.
- `report_libero_runs.py`: build inventory, summary, metric gate, and report index.

## Maintenance

- `export_unpushed_commits.sh`: export local commits as portable patches when remote push is blocked.
