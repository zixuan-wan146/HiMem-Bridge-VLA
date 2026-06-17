# RoboMME Data Plan For Transition Trigger


## Decision

Use RoboMME before RoboCasa365 for the first `transition_trigger` training loop.

Reason: RoboMME H5 data contains direct subgoal transition supervision:

- `info/is_subgoal_boundary`
- `info/simple_subgoal`
- `info/simple_subgoal_online`
- `info/grounded_subgoal`
- `info/grounded_subgoal_online`
- `info/is_completed`

This avoids manual boundary annotation for the first prototype.

## Remote Paths

All practical work is on the remote data disk:

```text
/root/autodl-tmp/benchmarks/robomme_benchmark
/root/autodl-tmp/datasets/robomme_data_h5
/root/autodl-tmp/logs/robomme_data_download.log
/root/autodl-tmp/logs/robomme_data_download_monitor.log
/root/autodl-tmp/logs/robomme_data_download_status.env
```

The download should avoid keeping two full copies of the H5 data. Use a local-dir style download
into `/root/autodl-tmp/datasets/robomme_data_h5`, with Hugging Face cache metadata on the data disk.

## Dataset Fields

Use the H5 fields as follows:

```text
state:
  obs/eef_state
  obs/joint_state
  obs/gripper_state
  obs/is_gripper_close

action:
  action/eef_action
  action/joint_action
  action/waypoint_action

label:
  info/is_subgoal_boundary

optional text:
  info/simple_subgoal_online
  info/grounded_subgoal_online
```

## Training Label Construction

For each episode:

```text
boundary_label[t] = bool(info/is_subgoal_boundary[t])
```

Initial positive/negative sampling:

```text
positive windows: windows centered near boundary_label[t] = true
negative windows: windows far from any boundary
terminal event:   track info/is_completed separately from internal subgoal boundaries
```

The first adapter should export the current `transition_trigger` sidecar shape:

```json
{
  "episode_id": "episode_000123",
  "task": "robomme_task",
  "subtask_id": 2,
  "subtask_name": "simple_subgoal_online text",
  "start": 120,
  "end": 183,
  "is_terminal": false,
  "label_source": "robomme/info/is_subgoal_boundary"
}
```

## Current Remote Setup Status

RoboMME repository and docs are cloned on the remote. Python 3.11 was installed by `uv`, but the
first `uv sync` failed during a network transfer for a SAPIEN wheel. This is separate from the H5
data download and can be retried after the data is local.

Known setup files:

```text
/root/autodl-tmp/scripts/robomme_setup.sh
/root/autodl-tmp/scripts/robomme_setup_monitor.sh
/root/autodl-tmp/logs/robomme_setup.log
/root/autodl-tmp/logs/robomme_setup_monitor.log
```

The active environment setup uses a local ManiSkill fallback because direct `uv sync` GitHub fetches
were unstable:

```text
/root/autodl-tmp/scripts/robomme_env_setup_local_maniskill.sh
/root/autodl-tmp/scripts/robomme_env_setup_monitor.sh
/root/autodl-tmp/logs/robomme_env_setup.log
/root/autodl-tmp/logs/robomme_env_setup_monitor.log
/root/autodl-tmp/sources/ManiSkill-07be6fbc66350ddca200abfb0a11b692f078f7fd
```

`pyproject.toml` in the remote RoboMME clone is temporarily patched so `mani-skill` points to that
local source path. The original remote file is backed up as:

```text
/root/autodl-tmp/benchmarks/robomme_benchmark/pyproject.toml.robomme_remote_backup
```

## Download Policy

Do:

- Download RoboMME H5 data to `/root/autodl-tmp/datasets/robomme_data_h5`.
- Keep Hugging Face metadata/cache under `/root/autodl-tmp/hf-cache`.
- Avoid a second full dataset copy in the global HF cache.
- Keep a monitor process running while downloading.
- Use `PIP_NO_CACHE_DIR=1` and `uv --no-cache` for the active setup/download scripts.

Do not:

- Download RoboCasa365 for this phase.
- Run RoboMME evaluation, training, or data generation during the download step.
- Store large artifacts on the system disk.

## Active Download Scripts

Current download and monitor entry points:

```text
/root/autodl-tmp/scripts/robomme_data_download.py
/root/autodl-tmp/scripts/robomme_data_download.sh
/root/autodl-tmp/scripts/robomme_data_download_monitor.sh
/root/autodl-tmp/logs/robomme_data_download.log
/root/autodl-tmp/logs/robomme_data_download_monitor.log
/root/autodl-tmp/logs/robomme_data_download_status.env
```

The active downloader uses `huggingface_hub.snapshot_download(local_dir=...)`. With current
Hugging Face Hub behavior, large files are written into the target local directory and only small
metadata files are kept in cache. Monitor both:

```text
du -sh /root/autodl-tmp/datasets/robomme_data_h5
du -sh /root/autodl-tmp/hf-cache
```
