# Scripts

Scripts are grouped by responsibility. Prefer these entry points over hand-written command chains. The active path no longer includes transition-trigger conversion, runtime, or trace-summary scripts.

## Quality Gates

- `check_repo.sh`: consolidated local gate. By default it skips training smoke tests; set `HIMEM_CHECK_INCLUDE_TRAINING=1` only when training smoke is intentionally in scope.
- `preflight.py`: repository, checkpoint, dataset, LIBERO result, and run-manifest checks.
- `audit_requirements.py`: dependency policy audit.
- `validate_bridge_himem_configs.py`: Bridge-HiMem YAML schema and `extends` validation.
- `validate_training_configs.py`: training profile validation.
- `validate_training_dataset.py`: dataset structure validation before training.

## Training And Model Assets

- `train.py`: main training entry point.
- `train_memory_token_cache_adapter.py`: historical memory-side smoke training over visual-token cache. Current work does not run or extend this entry.
- `himem_server.py`: websocket inference server.
- `download_libero_checkpoint.sh`: download the LIBERO checkpoint to the data disk.
- `start_himem_server.sh`: start the HiMem-Bridge-VLA websocket server with checkpoint preflight.

## Benchmark Data

- `download_rmbench_tasks.py`: download the official nine RMBench `demo_clean` task folders.
- `inspect_benchmarks.py`: inspect local LIBERO, LIBERO-Plus, and RMBench assets and emit a JSON inventory.
- `build_libero_memory_replay_index.py`: write a deterministic LIBERO JSONL index for current frames, short-memory history, and future action chunks.
- `build_rmbench_norm_stats.py`: compute RMBench min/max stats for state normalization and action denormalization.
- `build_rmbench_memory_replay_index.py`: write a deterministic RMBench JSONL index for current frames, short-memory history, and future action chunks.
- `build_memory_replay_token_cache.py`: encode replay-index frames into visual-token shards plus a manifest. Use `--encoder internvl3` for real caches and `--encoder image_stats` only for IO smoke tests.
- `plan_rmbench_eval.py`: inspect RMBench eval prerequisites and write direct/socket eval commands for the official `script/eval_policy.py` stack.
- `install_rmbench_policy_adapter.py`: install the checked-in `HiMemBridgeVLA` policy adapter into an official RMBench checkout.
- `run_rmbench_eval.sh`: reproducible RMBench eval wrapper. It installs the adapter, writes a run manifest, writes a command plan, then runs official direct eval task by task.

Token caches are read in training code with `MemoryTokenCacheDataset` and `collate_memory_token_cache_samples`; the builder only writes the cache.

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
