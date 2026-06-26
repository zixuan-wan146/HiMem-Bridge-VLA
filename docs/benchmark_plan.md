# Benchmark 调研与推进计划

状态说明：本文保留 LIBERO / LIBERO-Plus / RMBench 的本机数据状态、工程入口和 eval planning 信息。文中 2026-06-23 的 memory replay / token cache 内容只代表 short visual-memory 数据路径和历史 smoke 工程；long memory 的当前主设计已经改为 planner-coupled task-progress state，见 `docs/progress_state_planner_design_zh.md`。

---

当前准备关注三个 benchmark：LIBERO、LIBERO-Plus、RMBench。本文记录官方信息、本机数据状态、我们已有工程入口，以及 progress-state planner warmup 需要的数据准备工作。

## 1. LIBERO

官方定位：LIBERO 是面向 lifelong robot learning 的 manipulation benchmark，包含 procedural generation pipeline、130 个任务、四类 task suite，并提供 human teleoperation demonstrations。官方 GitHub 说明了四个 suite：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、LIBERO-100；其中 LIBERO-100 又拆成 LIBERO-90 / LIBERO-10。

官方入口：

```text
https://github.com/Lifelong-Robot-Learning/LIBERO
https://libero-project.github.io/
```

本机状态：

```text
$AUTODL_TMP/libero/datasets/libero_spatial  10 demo files, about 5.9G
$AUTODL_TMP/libero/datasets/libero_object   10 demo files, about 7.0G
$AUTODL_TMP/libero/datasets/libero_goal     10 demo files, about 6.0G
$AUTODL_TMP/libero/datasets/libero_10       10 demo files, about 13G
```

当前已经具备：

```text
coarse_planner/build_from_libero.py
scripts/build_libero_memory_replay_index.py
evaluations/libero/libero_client_4tasks.py
configs/libero_profiles/smoke.env
configs/libero_profiles/full_eval.env
scripts/setup_libero_env.sh
scripts/run_libero_smoke.sh
scripts/run_libero_eval.sh
scripts/plan_libero_run.py
scripts/report_libero_runs.py
```

用途：

- 数据准备：可以继续用 `coarse_planner/build_from_libero.py` 生成 H32 feature cache。
- eval：可以用现有 websocket server + LIBERO client 跑 smoke/full eval。
- memory 后续验证：LIBERO 是双视角单臂，memory compression 默认 `n_m=1`。
- memory replay：可以用 `scripts/build_libero_memory_replay_index.py` 生成轻量 JSONL index，固定当前帧、短期历史帧、action chunk 范围和 mask。
- frame replay：`himem_bridge_vla/dataset/memory_replay_frames.py` 可以根据 index row 回读当前图像、短期历史图像、state 和 future action chunk。
- replay dataset：`himem_bridge_vla/dataset/memory_replay_dataset.py` 提供 PyTorch-compatible dataset 和 collate，输出当前图像、短期历史图像、state、future actions 和 mask。
- visual token cache：`scripts/build_memory_replay_token_cache.py` 可以把 replay dataset 中的图像预先编码成按 view 分组的 visual tokens，写成 shard + manifest。真实训练默认使用 InternVL3 visual tower；测试和 smoke 可使用 `image_stats` encoder。
- token cache dataset：`himem_bridge_vla/dataset/memory_token_cache.py` 提供 `MemoryTokenCacheDataset` 和 `collate_memory_token_cache_samples`，可以从 shard 回读 visual tokens、state、future actions、`short_steps`、`short_mask`，并为每个样本构造 short `MemoryReadResult`。
- memory context：`himem_bridge_vla/training/memory_context.py` 可以把 token-cache batch 中的 `short_memory` 压缩成 batched `memory_context` 和 `memory_context_mask`，作为历史 smoke 数据接口。

缺口：

- 现有本机数据是常用 40-task VLA eval 子集，不是完整 LIBERO-100/130 task 全集。
- memory replay 和 token cache 已经有可执行入口；progress-state planner warmup 将复用其中的图像、state、future action chunk 和 mask 数据协议。
- low-level 图像是否每个执行 step 都用于训练，由 replay index 的 `stride`、`short_offsets` 和 token cache 生成命令共同决定。

## 2. LIBERO-Plus

官方论文定位：LIBERO-Plus 用于 VLA robustness analysis，在 LIBERO 类任务上加入七类扰动：

```text
object layout
camera viewpoint
robot initial state
language instruction
lighting condition
background texture
sensor noise
```

官方论文入口：

```text
https://arxiv.org/abs/2510.13626
```

截至 2026-06-23 的重新检索结果：arXiv 页面显示论文 v3 于 2025-12-26 修订，页面没有直接列出官方代码或数据仓库。当前仍不能把 LIBERO-Plus 当成已经可执行的本地 benchmark。

需要注意：`LIBERO+`、`LIBERO-PRO`、`LIBERO-Para` 这类名称相近资源不能默认等价于这里的 LIBERO-Plus。`scripts/inspect_benchmarks.py` 现在会把 exact LIBERO-Plus root 和这些 name-similar candidates 分开报告；只有 exact root 存在时才认为 LIBERO-Plus 可用。

本机状态：

```text
未在 `$AUTODL_TMP` 下发现 exact LIBERO-Plus 数据或代码目录。若后续发现 `LIBERO+` / `LIBERO-PRO` / `LIBERO-Para` 等目录，需要先人工确认是否就是目标论文 benchmark，不能直接接入。
```

用途：

- 更适合作为 robustness eval，而不是第一阶段训练数据。
- 可用于检查 memory 是否缓解 camera / initial state / sensor noise 扰动下的闭环不稳定。

缺口：

- 需要先确认是否有官方代码和数据发布入口。
- 需要确定扰动配置是否能复用现有 LIBERO eval client，还是需要单独的 environment wrapper。
- 在资源确认前，不应把 LIBERO-Plus 写进可执行 eval pipeline。

## 3. RMBench

官方定位：RMBench 是基于 RoboTwin 的 memory-dependent manipulation benchmark，包含 9 个 manipulation tasks，覆盖不同层级的记忆复杂度。官方仓库说明其基于 RoboTwin 2.0，推荐下载 assets 和 data，并在 `demo_clean` setting 下使用数据。

官方入口：

```text
https://rmbench.github.io/
https://github.com/RoboTwin-Platform/RMBench
```

本机状态：

```text
$AUTODL_TMP/benchmarks/RMBench           about 25G
$AUTODL_TMP/benchmarks/RMBench/data/rmbench_9tasks_manifest.json
repo_id: TianxingChen/RMBench
file_count: 1809
skip_video: false
```

已下载 9 个任务：

```text
observe_and_pickup
rearrange_blocks
put_back_block
swap_blocks
swap_T
blocks_ranking_try
press_button
cover_blocks
battery_try
```

每个任务当前都有：

```text
50 hdf5 files
50 trajectory files
50 instruction files
50 video files
```

抽样 HDF5 结构：

```text
observation/front_camera/rgb
observation/head_camera/rgb
observation/left_camera/rgb
observation/right_camera/rgb
third_view_rgb
joint_action/vector        [T, 14]
endpose/left_endpose       [T, 7]
endpose/right_endpose      [T, 7]
```

用途：

- RMBench 是后续 memory 机制的关键 benchmark，因为任务本身要求历史信息。
- 双臂/多摄像机设置默认 memory compression 使用 `n_m=2`。
- RMBench 双臂 action/state 协议和 LIBERO 单臂不同，后续任何训练或真实 eval 都不能直接复用 LIBERO 单臂处理；当前只整理数据读取、manifest、plan-only 检查。

缺口：

- 已实现轻量 RMBench HDF5 reader：`himem_bridge_vla/dataset/rmbench.py`。它不依赖官方仿真环境，只负责解码 encoded RGB bytes，读取多视角图像、14 维 joint action、双臂 endpose/gripper，并组合出 state vector。
- 已实现 RMBench normalization stats builder：`scripts/build_rmbench_norm_stats.py`。它输出与现有 `NormalizationStats` 兼容的 `norm_stats.json`，metadata 单独输出，避免训练端把 metadata 误当成 robot key。
- 已实现 RMBench memory replay index builder：`scripts/build_rmbench_memory_replay_index.py`。它按 low-level step 写出当前帧、短期 memory offset、action chunk 范围和 mask；当前只生成轻量 JSONL index，不缓存图像或 visual tokens。
- frame replay：`himem_bridge_vla/dataset/memory_replay_frames.py` 可以根据 index row 回读三相机图像、state 和 14 维 future action chunk。
- replay dataset：`himem_bridge_vla/dataset/memory_replay_dataset.py` 可复用同一套 index/frame reader，为后续 visual token cache 和数据检查提供稳定 batch 协议。
- visual token cache：`scripts/build_memory_replay_token_cache.py` 可复用同一套 replay dataset 生成三相机 visual token shard。RMBench / 双臂默认后续 memory compression 使用 `n_m=2`，但 cache 本身保存的是视觉塔输出 tokens，不在 cache 阶段做 learned-query 压缩。
- token cache dataset：`MemoryTokenCacheDataset` 可以读取 RMBench 三相机 token shard，并输出 `future_actions` 的 padded batch 和 `action_mask`。这一步只解决离线数据读取，不等价于训练脚本。
- 需要定义我们模型的 action/state protocol：是否直接使用 14 维 joint action，还是映射到 per-arm 7 维动作。当前只记录该缺口，不实现训练或真实 eval。
- 已实现 RMBench eval plan builder：`scripts/plan_rmbench_eval.py`。它读取本地 RMBench root，检查官方 `script/eval_policy.py` / `script/eval_policy_client.py` / `script/policy_model_server.py` / policy config / task env / data 目录，并生成 direct 或 socket 模式的评估命令。
- 官方 direct 模式入口是 `script/eval_policy.py`，会 import `policy/<policy_name>/deploy_policy.py` 中的 `get_model`、`eval`、`reset_model`。
- 官方 socket 模式入口是 `script/policy_model_server.py` + `script/eval_policy_client.py`，server 侧持有模型，client 侧跑环境并通过 TCP 调用 `reset_model` / `get_action` / `update_obs`。
- 已实现 RMBench policy adapter 源码：`evaluations/rmbench/policy/HiMemBridgeVLA/`。它把 RMBench observation 转成我们 server/protocol 需要的三相机图像、state、prompt、action mask，再把 32-step action chunk 转成 RMBench `qpos` action。
- 已实现安装脚本：`scripts/install_rmbench_policy_adapter.py`。它会把 adapter 复制到官方 RMBench 仓库的 `policy/HiMemBridgeVLA/` 下，供 `script/eval_policy.py` import。
- 已实现 RMBench eval run wrapper：`scripts/run_rmbench_eval.sh`。它会安装 adapter、写 run manifest、写 eval plan，然后逐任务调用官方 `script/eval_policy.py`。
- 已实现 RMBench run manifest：`scripts/write_rmbench_run_manifest.py`，记录任务列表、server URI、action/state protocol、policy 名称和 git/environment metadata。
- 已有 token cache batch 到 `memory_context` / `memory_context_mask` 的共享构造入口；该入口只作为历史 smoke 数据接口保留。

## 4. 统一推进任务

第一阶段只做可复现准备：

```text
python scripts/inspect_benchmarks.py --data-root "$AUTODL_TMP" --output run_outputs/benchmark_inventory.json --allow-missing
python scripts/build_libero_memory_replay_index.py --libero-root "$AUTODL_TMP/libero/datasets" --output run_outputs/libero_memory_replay.jsonl
python scripts/build_rmbench_norm_stats.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_norm_stats.json --metadata-output run_outputs/rmbench_norm_stats.metadata.json
python scripts/build_rmbench_memory_replay_index.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_memory_replay.jsonl
python scripts/validate_bridge_himem_configs.py
python -m pytest -q
```

第二阶段补数据 adapter：

```bash
python scripts/build_memory_replay_token_cache.py \
  --benchmark LIBERO \
  --data-root "$AUTODL_TMP/libero/datasets" \
  --index run_outputs/libero_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/libero_memory_replay" \
  --encoder internvl3

python scripts/build_memory_replay_token_cache.py \
  --benchmark RMBench \
  --data-root "$AUTODL_TMP/benchmarks/RMBench" \
  --index run_outputs/rmbench_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/rmbench_memory_replay" \
  --encoder internvl3
```

数据代码读取 cache 的最小入口：

```python
from torch.utils.data import DataLoader

from himem_bridge_vla.dataset import MemoryTokenCacheDataset
from himem_bridge_vla.dataset import collate_memory_token_cache_samples

dataset = MemoryTokenCacheDataset("$AUTODL_TMP/token_caches/libero_memory_replay")
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_memory_token_cache_samples)
```

第三阶段只保留 eval 计划检查：

```text
LIBERO existing eval smoke/full
RMBench eval command planner
RMBench policy adapter install + plan-only check
LIBERO-Plus resource locator
```

RMBench eval plan 命令：

```bash
python scripts/plan_rmbench_eval.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --output run_outputs/rmbench_eval_plan.md \
  --policy-name HiMemBridgeVLA \
  --task-config demo_clean \
  --tasks observe_and_pickup rearrange_blocks put_back_block swap_blocks swap_T blocks_ranking_try press_button cover_blocks battery_try \
  --ckpt-setting himem_bridge_vla \
  --seed 0 \
  --gpu-id 0 \
  --mode direct
```

安装 adapter：

```bash
python scripts/install_rmbench_policy_adapter.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --force
```

固定运行入口：

```bash
HIMEM_RMBENCH_TASKS=press_button \
HIMEM_RMBENCH_RUN_DIR=run_outputs/rmbench_smoke \
HIMEM_RMBENCH_PLAN_ONLY=1 \
HIMEM_SERVER_URI=ws://127.0.0.1:9000 \
bash scripts/run_rmbench_eval.sh
```

其中 `HIMEM_RMBENCH_PLAN_ONLY=1` 表示 plan-only 检查模式：安装 adapter、写 manifest、写 eval plan。
