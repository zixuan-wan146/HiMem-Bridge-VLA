# Legacy Workspace Cleanup

Use this note only when retiring an older sibling checkout. The active workspace is `HiMem-Bridge-VLA`,
which contains BridgeAttention, HiMem, YAML configs, CALVIN tooling, documentation, and tests.

## Current Check

As of the latest local check:

- The active checkout has a clean git working tree.
- The active branch is aligned with its remote.
- Files present only in the legacy checkout are git metadata under `.git/`.
- Files present only in the active checkout include:
  - `himem_bridge_vla/bridge_himem_config.py`
  - `himem_bridge_vla/experiment_config.py`
  - `himem_bridge_vla/reproducibility.py`
  - `himem_bridge_vla/model/bridge/`
  - `himem_bridge_vla/model/himem/`
  - `configs/bridge_himem/`
  - `docs/`
  - Bridge/HiMem/CALVIN tests

## Recommendation

Content-wise, a legacy checkout can be removed after confirming no one still needs its local git history.
Because it has a separate `.git`, deletion should be explicit rather than bundled with ordinary
HiMem-Bridge-VLA edits.

Safer archive option:

```bash
cd workspace-parent
tar -czf legacy_checkout_$(date +%Y%m%d).tar.gz legacy_checkout
rm -rf legacy_checkout
```

Direct removal:

```bash
rm -rf legacy_checkout
```

After removal, run Bridge-HiMem checks from:

```bash
cd HiMem-Bridge-VLA
python3 scripts/validate_bridge_himem_configs.py
python3 -m pytest -q
```
