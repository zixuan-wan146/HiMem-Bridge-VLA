# Project Layout

HiMem-Bridge-VLA now uses a `src/` Python package layout. Importable project code lives under `src/himem_bridge_vla/`; repository-root directories such as `configs/`, `scripts/`, `tests/`, `docs/`, and `evaluations/` remain non-package project assets.

The migration follows `Structure.md`: core contracts are separated from model, data, training, runtime, benchmark, evaluation, and diagnostics code. Script entry points are grouped under `scripts/train/`, `scripts/cache/`, `scripts/serve/`, `scripts/eval/`, `scripts/report/`, `scripts/quality/`, `scripts/setup/`, and `scripts/maintenance/`. Historical root-level script names were removed instead of kept as stale compatibility wrappers.

Large datasets, checkpoints, caches, logs, and run outputs must stay outside git on the remote data disk. Repository paths in code and configs should remain project-relative.
