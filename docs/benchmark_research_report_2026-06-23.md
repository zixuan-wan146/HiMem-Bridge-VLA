# Benchmark 调研报告：LIBERO / LIBERO-Plus / RMBench

日期：2026-06-23

本文是本轮 benchmark 调研的独立报告，和 `docs/benchmark_plan.md` 分工不同：

- 本文回答：这些 benchmark 是什么、本机资源是什么状态、我们怎么用、有哪些任务、后续缺什么。
- `docs/benchmark_plan.md` 回答：工程推进顺序和可执行命令。

当前边界：不新增训练接口，不跑真实 checkpoint + server 的 RMBench 端到端 eval，不实现 LIBERO-Plus robustness wrapper。

## 1. 总览

| Benchmark | 当前定位 | 本机状态 | 当前可用程度 | 我们当前怎么用 |
| --- | --- | --- | --- | --- |
| LIBERO | 单臂 manipulation / VLA 常用 eval 与 demo 数据来源 | 已有 40-task 常用子集 | 可用于数据 replay、token cache、现有 LIBERO eval 入口 | 作为双视角单臂 memory replay 基准 |
| LIBERO-Plus | LIBERO 类任务上的 VLA robustness analysis | exact root 缺失 | 暂不可执行 | 只保留调研结论和资源定位检查 |
| RMBench | memory-dependent 双臂 / 多相机 manipulation benchmark | 已有 9 tasks 数据 | 可用于数据读取、norm stats、memory replay、eval plan-only | 作为后续 memory 能力主 benchmark |

## 2. LIBERO

### 2.1 是什么

LIBERO 是面向 lifelong robot learning 的 manipulation benchmark。官方论文描述其包含 procedural generation pipeline、4 类 task suite、共 130 个 benchmark tasks，并提供 human teleoperation demonstrations。

官方入口：

```text
https://arxiv.org/abs/2306.03310
https://libero-project.github.io/
https://github.com/Lifelong-Robot-Learning/LIBERO
```

核心 suite：

- `LIBERO-Spatial`
- `LIBERO-Object`
- `LIBERO-Goal`
- `LIBERO-100`

其中常见 VLA 设置里经常用到 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 这 40 个任务。

### 2.2 本机资源状态

本机 inventory 结果：

```text
$AUTODL_TMP/libero/datasets/libero_spatial  10 demo files
$AUTODL_TMP/libero/datasets/libero_object   10 demo files
$AUTODL_TMP/libero/datasets/libero_goal     10 demo files
$AUTODL_TMP/libero/datasets/libero_10       10 demo files
$AUTODL_TMP/libero/datasets/libero_90       missing
$AUTODL_TMP/libero/datasets/libero_100      missing
```

结论：

- 本机已有的是常用 40-task 子集。
- 不是完整 LIBERO-90 / LIBERO-100 / 130-task 全集。

### 2.3 我们怎么用

当前 LIBERO 适合承担三个角色：

1. 单臂 / 双视角 memory replay 数据源。
2. visual token cache IO 和 mask 逻辑的稳定验证集。
3. 保留现有 LIBERO websocket eval client，用于以后恢复闭环 eval 时复用。

当前已经具备的工程入口：

```text
scripts/build_libero_memory_replay_index.py
himem_bridge_vla/dataset/memory_replay_frames.py
himem_bridge_vla/dataset/memory_replay_dataset.py
scripts/build_memory_replay_token_cache.py
himem_bridge_vla/dataset/memory_token_cache.py
evaluations/libero/libero_client_4tasks.py
scripts/run_libero_smoke.sh
scripts/run_libero_eval.sh
scripts/plan_libero_run.py
```

memory 设置：

- 视角：通常是 `base` / `wrist` 两个视角。
- 每个 memory entry 默认压缩为 `n_m=1` 个 token。
- short memory 默认使用 low-level step offset，例如 `{t-32, t-16}`。

### 2.4 后续缺口

当前不推进训练接口，因此缺口只记录，不实现：

- 若要覆盖完整官方 benchmark，需要补 LIBERO-90 / LIBERO-100 数据。
- 若恢复训练，需要确认 replay index 的 `stride`、`short_offsets`、action horizon 是否和训练频率一致。
- 若恢复 eval，需要确认 checkpoint、server 协议、LIBERO profile 和动作尺度一致。

## 3. LIBERO-Plus

### 3.1 是什么

LIBERO-Plus 是面向 VLA robustness analysis 的 LIBERO 类扩展。论文关注在常规 LIBERO 风格任务上加入扰动，评估 VLA 模型在看似高分背后的脆弱性。

官方论文入口：

```text
https://arxiv.org/abs/2510.13626
```

论文中关注的扰动维度包括：

- object layout
- camera viewpoint
- robot initial state
- language instruction
- lighting condition
- background texture
- sensor noise

### 3.2 exact root 缺失是什么意思

`exact root` 是我们在本机用于定位“目标 benchmark 的真实代码/数据根目录”的路径。

对于 LIBERO-Plus，当前 `scripts/inspect_benchmarks.py` 默认检查：

```text
$AUTODL_TMP/libero_plus
```

同时脚本认为下面这些名字属于 exact-name candidate：

```text
libero_plus
libero-plus
LIBERO-Plus
LIBERO_PLUS
```

如果这些目录都不存在，inventory 会报告：

```text
libero_plus.exists = false
libero_plus.status = missing
```

这就是“LIBERO-Plus exact root 缺失”的意思。

它不是说 LIBERO-Plus 论文不存在，也不是说未来不能做。它只表示：

1. 远端数据盘上没有被确认的 LIBERO-Plus 代码/数据目录。
2. 当前不能把 LIBERO-Plus 当作一个已经下载、已经可执行的本地 benchmark。
3. 当前不应该写 wrapper 去假设它的环境 API、扰动配置和数据结构。

### 3.3 为什么不把名称相近的资源直接当作 LIBERO-Plus

公开资料里还有一些名称接近但目标不同的资源，例如：

```text
LIBERO+
LIBERO-PRO
LIBERO-Para
```

这些名字相近，但不能默认等价于 `LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models`。

例如 `LIBERO-PRO` 是另一篇 robustness / generalized evaluation 方向的工作，公开结果里能看到其独立论文和代码入口，但这不等于 LIBERO-Plus。

所以脚本把它们分成两类：

- exact root：可以作为 LIBERO-Plus 使用的目标目录。
- related candidates：名字相似，但默认不能当成 LIBERO-Plus。

这可以避免把不同 benchmark 混接进同一个 eval pipeline，导致结果不可解释。

### 3.4 本机资源状态

本机 inventory 结果：

```text
$AUTODL_TMP/libero_plus       missing
related_candidates           none found
```

结论：

- 当前没有可执行的 LIBERO-Plus 本地资源。
- 当前只保留资源定位检查，不进入 eval pipeline。

### 3.5 后续缺口

如果以后要做 LIBERO-Plus，需要先解决：

1. 确认官方代码/数据是否发布。
2. 确认下载后的根目录和 `inspect_benchmarks.py --libero-plus-root` 指向一致。
3. 确认其 perturbation config 是否能复用 LIBERO 环境，还是需要单独 wrapper。
4. 确认任务列表、相机设置、action/state 协议和现有 LIBERO client 的兼容性。

当前不做 LIBERO-Plus robustness wrapper。

## 4. RMBench

### 4.1 是什么

RMBench 是 memory-dependent robotic manipulation benchmark，目标是系统评估策略在历史依赖任务中的 memory capability。官方论文描述其包含 9 个 manipulation tasks，覆盖不同层级的 memory complexity。

官方入口：

```text
https://arxiv.org/abs/2603.01229
https://rmbench.github.io/
https://github.com/RoboTwin-Platform/RMBench
```

RMBench 对我们更关键，因为它天然要求历史信息，而不是只把 memory 当作额外上下文。

### 4.2 本机资源状态

本机 inventory 结果：

```text
$AUTODL_TMP/benchmarks/RMBench
$AUTODL_TMP/benchmarks/RMBench/data/rmbench_9tasks_manifest.json
repo_id: TianxingChen/RMBench
file_count: 1809
skip_video: false
```

已确认 9 个任务：

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

抽样 HDF5 结构包含：

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

### 4.3 我们怎么用

当前 RMBench 适合承担四个角色：

1. 多相机 memory replay 数据源。
2. 双臂 action/state 协议调研对象。
3. memory-dependent benchmark 的后续主评估对象。
4. eval plan-only / manifest / adapter 安装路径验证对象。

当前已经具备的工程入口：

```text
himem_bridge_vla/dataset/rmbench.py
scripts/build_rmbench_norm_stats.py
scripts/build_rmbench_memory_replay_index.py
himem_bridge_vla/dataset/memory_replay_frames.py
himem_bridge_vla/dataset/memory_replay_dataset.py
scripts/build_memory_replay_token_cache.py
himem_bridge_vla/dataset/memory_token_cache.py
scripts/plan_rmbench_eval.py
evaluations/rmbench/policy/HiMemBridgeVLA/
scripts/install_rmbench_policy_adapter.py
scripts/write_rmbench_run_manifest.py
scripts/run_rmbench_eval.sh
```

memory 设置：

- 视角：双臂/三摄像机场景默认取 `scene` / `left` / `right` 类似设置，具体 view name 由 reader/adapter 统一。
- 每个 memory entry 默认压缩为 `n_m=2` 个 token。
- action：数据中存在 14 维 `joint_action/vector`。
- state：当前 reader 会组合左右臂 endpose / gripper 等状态信息。

### 4.4 后续缺口

当前不跑真实 eval、不新增训练接口，因此缺口只记录：

- 需要最终确认 action/state protocol：直接预测 14 维 joint action，还是拆成 per-arm 7 维。
- 如果恢复真实 eval，需要确认 RMBench 官方环境、server protocol、action scale、checkpoint action dim 是否一致。
- 如果恢复训练，需要确认 replay index、visual token cache 和主模型训练 batch 协议的接入方式。

## 5. 当前建议

当前合理推进顺序：

1. 保持 LIBERO 和 RMBench 的 inventory / replay index / token cache / mask / manifest 可复现。
2. 保持 `check_repo.sh` 默认不跑训练 smoke，避免误触当前不做的训练接口。
3. RMBench 只跑 plan-only 检查，不跑真实 checkpoint + server eval。
4. LIBERO-Plus 在 exact root 缺失前，不写 wrapper、不纳入可执行 pipeline。

当前不建议做：

- 不把 LIBERO-Plus 和 LIBERO-PRO / LIBERO+ / LIBERO-Para 混用。
- 不用 RMBench 的 14 维 action 直接假设成 LIBERO action 协议。
- 不把 `image_stats` token cache 当成真实训练视觉特征。
- 不把小型动作回归 adapter 当成正式训练方案。

## 6. 一句话结论

LIBERO 和 RMBench 的本机数据状态已经清楚，离线 replay / cache / mask / manifest 路径已经具备；LIBERO-Plus 目前只是论文目标明确，但本机 exact code/data root 缺失，因此只能保留为待资源确认的 robustness benchmark，不能进入实际 eval pipeline。
