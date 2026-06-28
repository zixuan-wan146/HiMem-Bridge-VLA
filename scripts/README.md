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

- `train/stage1/libero.py`: active LIBERO Stage1 trajectory-window token-cache training launcher. The command delegates to `src/himem_bridge_vla/training/stage1/libero/` for LIBERO-specific config/contract and `src/himem_bridge_vla/training/stage1/common/` for the shared loop.
- `serve/serve_policy.py`: active websocket inference server entry.
- `download_libero_checkpoint.sh`: download the LIBERO checkpoint to the data disk.
- `start_himem_server.sh`: start the HiMem-Bridge-VLA websocket server with checkpoint preflight.

## Benchmark Data

- `download_rmbench_tasks.py`: download the official nine RMBench `demo_clean` task folders.
- `inspect_benchmarks.py`: inspect local LIBERO, LIBERO-Plus, and RMBench assets and emit a JSON inventory.
- `build_libero_episode_replay_index.py`: write an episode-first LIBERO JSON index. Each episode stores replan nodes plus the complete set of required visual frame indices, including short-memory frames such as `t-8` that are not themselves replan nodes.
- `build_libero_episode_feature_cache.py`: materialize an episode-level processed feature cache from the episode-first index. It stores visual tokens, current VLM hidden-state layers, planner summaries, states, actions, and node metadata; it does not duplicate raw image arrays from HDF5.
- `build_rmbench_norm_stats.py`: compute RMBench min/max stats for state normalization and action denormalization.
- `build_rmbench_memory_replay_index.py`: write a deterministic RMBench JSONL index for current frames, short-memory history, and future action chunks. The index keeps `long_capacity=0`; progress-state long memory is not stored in replay rows.
- `build_memory_replay_token_cache.py`: encode RMBench replay-index frames into replay-token shards plus a manifest. LIBERO Stage1 uses `build_libero_episode_feature_cache.py` instead.
- `plan_rmbench_eval.py`: inspect RMBench eval prerequisites and write direct/socket eval commands for the official `script/eval_policy.py` stack.
- `install_rmbench_policy_adapter.py`: install the checked-in `HiMemBridgeVLA` policy adapter into an official RMBench checkout.
- `run_rmbench_eval.sh`: reproducible RMBench eval wrapper. It installs the adapter, writes a run manifest, writes a command plan, then runs official direct eval task by task.

RMBench replay-token caches are read with `MemoryTokenCacheDataset` and `collate_memory_token_cache_samples`. They expose current visual tokens, optional current VLM hidden-state layers, recent short visual tokens, `short_steps`, `short_mask`, state, and future action chunks for IO checks.

There are two different cache families:

- `*_progress_vl_embedding_warmup_cache`: pooled VL embedding windows for progress-state planner warm-up.
- `libero_episode_feature_cache`: active LIBERO Stage1 cache. It stores episode-level nodes, current hidden states, planner summaries, short-memory visual tokens, states, and actions.
- `memory_replay_visual_token_cache`: replay-token shards retained for RMBench and low-level IO smoke checks.

Do not pass progress warm-up caches into the direct bridge token-cache smoke; the script validates the manifest format and will reject them.

For active Stage1 direct bridge-attn training, use `scripts/train/stage1/libero.py` or `python -m himem_bridge_vla.training.stage1.libero.cli`. The Stage1 loader requires `libero_episode_feature_cache` and advances the frozen progress planner state chronologically through each episode's replan nodes.

Example:

```bash
python scripts/eval/inspect_benchmarks.py \
  --data-root "$AUTODL_TMP" \
  --output run_outputs/benchmark_inventory.json \
  --allow-missing

python scripts/cache/build_libero_episode_replay_index.py \
  --libero-root "$AUTODL_TMP/libero/datasets" \
  --suites libero_10 \
  --output run_outputs/libero_10_episode_replay.json

python scripts/cache/build_rmbench_norm_stats.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --output run_outputs/rmbench_norm_stats.json \
  --metadata-output run_outputs/rmbench_norm_stats.metadata.json

python scripts/cache/build_rmbench_memory_replay_index.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --output run_outputs/rmbench_memory_replay.jsonl

python scripts/cache/build_libero_episode_feature_cache.py \
  --episode-index run_outputs/libero_10_episode_replay.json \
  --libero-root "$AUTODL_TMP/libero/datasets" \
  --output-root "$AUTODL_TMP/token_caches/libero_10_episode_feature_internvl3_hidden_l3_6_9_12_stride16" \
  --encoder internvl3 \
  --include-vlm-hidden-states \
  --hidden-state-layers 3 6 9 12 \
  --storage-dtype bfloat16 \
  --device cuda

python scripts/cache/build_memory_replay_token_cache.py \
  --benchmark RMBench \
  --data-root "$AUTODL_TMP/benchmarks/RMBench" \
  --index run_outputs/rmbench_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/rmbench_memory_replay_image_stats_smoke" \
  --encoder image_stats \
  --max-samples 2 \
  --max-samples-per-shard 2 \
  --storage-dtype float32

python scripts/quality/smoke_direct_bridge_inference.py \
  --preset final \
  --device auto \
  --progress-planner-checkpoint "$AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt"

python scripts/eval/plan_rmbench_eval.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --output run_outputs/rmbench_eval_plan.md \
  --mode direct \
  --tasks observe_and_pickup press_button

HIMEM_RMBENCH_TASKS=press_button \
HIMEM_RMBENCH_RUN_DIR=run_outputs/rmbench_smoke \
HIMEM_RMBENCH_PLAN_ONLY=1 \
HIMEM_SERVER_URI=ws://127.0.0.1:9000 \
bash scripts/eval/run_rmbench_eval.sh
```

## LIBERO Runs

- `setup_libero_env.sh`: create or validate the LIBERO simulation environment.
- `run_libero_smoke.sh`: minimal smoke evaluation.
- `run_libero_eval.sh`: full LIBERO evaluation.
- `plan_libero_run.py`: generate reproducible server/eval/report commands before running.
- `init_libero_experiment.py`: create a tracked experiment skeleton.
- `libero_profile.sh`: safe parser for `configs/runtime/libero_profiles/*.env`.

## Reporting

- `write_libero_run_manifest.py`: write run metadata before evaluation starts.
- `write_rmbench_run_manifest.py`: write run metadata before RMBench evaluation starts.
- `summarize_libero_results.py`: summarize result JSON files.
- `check_libero_metrics.py`: gate candidate runs against thresholds or baseline.
- `report_libero_runs.py`: build inventory, summary, metric gate, and report index.

## Maintenance

- `export_unpushed_commits.sh`: export local commits as portable patches when remote push is blocked.
