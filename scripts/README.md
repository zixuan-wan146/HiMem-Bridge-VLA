# Scripts

Scripts are grouped by responsibility. Prefer these entry points over hand-written command chains.

## Quality Gates

- `check_repo.sh`: consolidated local gate.
- `preflight.py`: repository, checkpoint, dataset, LIBERO result, and run-manifest checks.
- `audit_requirements.py`: dependency policy audit.
- `validate_bridge_himem_configs.py`: Bridge-HiMem YAML schema and `extends` validation.
- `validate_training_dataset.py`: dataset structure validation before training.

## Training And Model Assets

- `train.py`: main training entry point.
- `himem_server.py`: websocket inference server.
- `download_libero_checkpoint.sh`: download the LIBERO checkpoint to the data disk.
- `start_himem_server.sh`: start the HiMem-Bridge-VLA websocket server with checkpoint preflight.

## Transition Trigger Data

- `download_rmbench_tasks.py`: download the official nine RMBench `demo_clean` task folders.
- `convert_rmbench_to_transition_trigger.py`: convert RMBench HDF5 trajectories and
  `language_annotation.json` durations into motion-boundary parquet plus boundary sidecars.
- `convert_robomme_h5_to_transition_trigger.py`: convert RoboMME H5 tasks with native
  `info/is_subgoal_boundary` supervision into the same motion-boundary parquet format.

## LIBERO Runs

- `setup_libero_env.sh`: create or validate the LIBERO simulation environment.
- `run_libero_smoke.sh`: minimal smoke evaluation.
- `run_libero_eval.sh`: full LIBERO evaluation.
- `plan_libero_run.py`: generate reproducible server/eval/report commands before running.
- `init_libero_experiment.py`: create a tracked experiment skeleton.
- `libero_profile.sh`: safe parser for `configs/libero_profiles/*.env`.

## Reporting

- `write_libero_run_manifest.py`: write run metadata before evaluation starts.
- `summarize_libero_results.py`: summarize result JSON files.
- `check_libero_metrics.py`: gate candidate runs against thresholds or baseline.
- `report_libero_runs.py`: build inventory, summary, metric gate, and report index.

## Maintenance

- `export_unpushed_commits.sh`: export local commits as portable patches when remote push is blocked.
