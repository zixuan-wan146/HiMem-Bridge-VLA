# 2026-06-23 工作总结

本文记录本轮对 HiMem-Bridge-VLA 的主要整理结果。重点是当前方案、核心代码、可复现入口和 benchmark 状态；中间临时尝试、重复同步、命令细节不展开。

## 1. 当前边界

本轮最后确认的边界如下：

- KFS 暂缓，不实现 keyframe selector。
- 当前 memory 方案统一为 Dual-FIFO Visual Token Memory，不再使用版本号命名。
- memory entry 只保留 visual tokens、$\tau_i$、$\eta_i$，不保留 `metadata_i`。
- 当前不推进正式 Bridge/action-head 训练接入。
- 当前不新增 token cache / memory context 的训练接口。
- 当前不做真实 checkpoint + HiMem server 的 RMBench 端到端 eval smoke。
- 当前不做 LIBERO-Plus robustness eval wrapper。

也就是说，本轮工作的核心是：把 memory 侧和 benchmark 侧的基础工程整理清楚，让数据、缓存、mask、manifest、plan-only 检查可以复现；训练和真实闭环 eval 暂时不继续扩展。

## 2. Memory 侧核心实现

当前 `himem_bridge_vla/model/himem/` 下只保留 Dual-FIFO visual memory 主线：

- `VisualMemoryEntry`：记录每个历史 entry 的多视角 visual tokens、写入时刻 $\tau_i$、记忆类型 $\eta_i$。
- `DualFifoVisualMemory`：维护 short FIFO 和 long FIFO。
- short memory：按 low-level step offset 确定性读取，例如默认 `short_offsets=[32,16]`。
- long memory：当前先支持 external/oracle keyframe 写入，KFS 以后再接。
- padding entry：缺失 entry 用 `None` / mask 表示，压缩后 padding token 保持全零。
- `VisualMemoryCompressor`：使用 view-aware learned-query compression，把每个 memory entry 压成固定数量 memory tokens。

关键约定：

- LIBERO / 双视角单臂默认每个 memory entry 压成 `n_m=1` 个 token。
- 双臂 / 三摄像机 benchmark，例如 RMBench，默认每个 memory entry 压成 `n_m=2` 个 token。
- padding slot 不作为可学习 null token 使用；下游通过 `memory_context_mask` 屏蔽。

旧的检索式 HiMem / boundary writer / controller / segment accumulator 已经清理，避免和当前 Dual-FIFO 方案并列存在。

## 3. Memory Context 与 Mask

新增了 token-cache batch 到 memory context 的共享构造入口：

```python
from himem_bridge_vla.training import build_token_cache_memory_context

memory_batch = build_token_cache_memory_context(batch, memory_compressor)
model_kwargs = memory_batch.as_model_kwargs()
```

输出包含：

- `memory_context`：形状为 `[B, T_m, D]` 的压缩 memory tokens。
- `memory_context_mask`：形状为 `[B, T_m]` 的有效 token mask。

处理逻辑：

- 有效 entry 先经过 `VisualMemoryCompressor` 压缩。
- 缺失 entry 的输出 token 保持全零。
- `memory_context_mask=False` 的位置在 Bridge-Attention condition path 中被 mask 掉。

这个接口现在只作为数据到 memory context 的稳定路径保留，不继续扩展训练接口。

## 4. Benchmark 状态

### 4.1 LIBERO

本机已确认有常用 40-task VLA eval 子集：

```text
$AUTODL_TMP/libero/datasets/libero_spatial  10 demo files
$AUTODL_TMP/libero/datasets/libero_object   10 demo files
$AUTODL_TMP/libero/datasets/libero_goal     10 demo files
$AUTODL_TMP/libero/datasets/libero_10       10 demo files
```

已经补齐的工程入口：

- `scripts/build_libero_memory_replay_index.py`
- `himem_bridge_vla/dataset/memory_replay_frames.py`
- `himem_bridge_vla/dataset/memory_replay_dataset.py`
- `scripts/build_memory_replay_token_cache.py`
- `himem_bridge_vla/dataset/memory_token_cache.py`

LIBERO 当前用途：

- 作为双视角单臂数据源验证 memory replay、visual token cache 和 short memory mask。
- 后续如果恢复训练，再从这些稳定数据入口往主训练链路接。

注意：当前本机 LIBERO 数据不是完整 LIBERO-100/130 全集。

### 4.2 LIBERO-Plus

调研结论：

- LIBERO-Plus 是面向 VLA robustness analysis 的 benchmark，关注 object layout、camera viewpoint、robot initial state、language instruction、lighting、background texture、sensor noise 等扰动。
- 截至本轮调研，没有在本机发现 exact LIBERO-Plus 数据或官方代码目录。
- `LIBERO+`、`LIBERO-PRO`、`LIBERO-Para` 这类名称相近资源不能自动视为 LIBERO-Plus。

当前工程处理：

- `scripts/inspect_benchmarks.py` 会区分 exact LIBERO-Plus root 和 name-similar candidates。
- 只有 exact root 存在时才把 LIBERO-Plus 视为可用 benchmark。
- 当前不做 LIBERO-Plus robustness wrapper。

### 4.3 RMBench

本机已确认有 RMBench 9-task 数据：

```text
$AUTODL_TMP/benchmarks/RMBench
repo_id: TianxingChen/RMBench
tasks:
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

每个任务当前都有 hdf5、trajectory、instruction、video 文件。抽样 HDF5 中包含多相机 RGB、14 维 joint action、左右臂 endpose。

已经补齐的工程入口：

- `himem_bridge_vla/dataset/rmbench.py`
- `scripts/build_rmbench_norm_stats.py`
- `scripts/build_rmbench_memory_replay_index.py`
- `scripts/plan_rmbench_eval.py`
- `evaluations/rmbench/policy/HiMemBridgeVLA/`
- `scripts/install_rmbench_policy_adapter.py`
- `scripts/write_rmbench_run_manifest.py`
- `scripts/run_rmbench_eval.sh`

当前 RMBench 用途：

- 作为多摄像机 / 双臂 / memory-dependent benchmark 的主要后续验证对象。
- 当前只做数据读取、norm stats、memory replay index、token cache、manifest、eval plan-only 检查。
- 真实 checkpoint + server 的端到端 eval smoke 暂不推进。

仍需以后明确的问题：

- action/state protocol：直接使用 14 维 joint action，还是拆成 per-arm 7 维动作。
- 如果以后恢复真实 eval，需要再确认仿真环境、动作尺度、server 协议和 checkpoint 维度是否完全一致。

## 5. 可复现入口

当前推荐的非训练检查入口：

```bash
python scripts/inspect_benchmarks.py --data-root "$AUTODL_TMP" --output run_outputs/benchmark_inventory.json --allow-missing
python scripts/build_libero_memory_replay_index.py --libero-root "$AUTODL_TMP/libero/datasets" --output run_outputs/libero_memory_replay.jsonl
python scripts/build_rmbench_norm_stats.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_norm_stats.json --metadata-output run_outputs/rmbench_norm_stats.metadata.json
python scripts/build_rmbench_memory_replay_index.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_memory_replay.jsonl
python scripts/build_memory_replay_token_cache.py --benchmark LIBERO --data-root "$AUTODL_TMP/libero/datasets" --index run_outputs/libero_memory_replay.jsonl --output-root "$AUTODL_TMP/token_caches/libero_memory_replay" --encoder image_stats --max-samples 2
python scripts/build_memory_replay_token_cache.py --benchmark RMBench --data-root "$AUTODL_TMP/benchmarks/RMBench" --index run_outputs/rmbench_memory_replay.jsonl --output-root "$AUTODL_TMP/token_caches/rmbench_memory_replay" --encoder image_stats --max-samples 2
HIMEM_RMBENCH_PLAN_ONLY=1 HIMEM_RMBENCH_TASKS=press_button bash scripts/run_rmbench_eval.sh
```

`image_stats` 只用于 IO smoke，不是训练视觉特征。

`scripts/check_repo.sh` 已调整为默认不跑训练 smoke 测试。若以后确实要把训练 smoke 也纳入检查，需要显式设置：

```bash
HIMEM_CHECK_INCLUDE_TRAINING=1 scripts/check_repo.sh
```

当前默认检查更适合本轮边界：代码、配置、benchmark inventory、dry-run / plan-only、compile 和 whitespace。

## 6. 文档状态

本轮主要文档更新：

- `docs/dual_fifo_visual_memory_design_zh.md`：当前 memory 方案说明。
- `docs/dual_fifo_visual_memory_qa_zh.md`：对 $\tau_i$、$\eta_i$、padding mask、type embedding、推理频率、replan stride 等问题的中文解释。
- `docs/benchmark_plan.md`：LIBERO / LIBERO-Plus / RMBench 的状态和推进计划。
- `docs/engineering_reproducibility.md`：工程边界、检查入口、数据盘布局和可复现约定。
- `to-do/6-23.md`：当天任务状态和暂不做事项。

文档口径已经统一：KFS、正式训练接口、真实 RMBench eval smoke、LIBERO-Plus wrapper 都不在当前推进范围内。

## 7. 当前结论

本轮任务已经把核心工程状态收口到一个清晰边界：

- memory 侧代码不再混用旧方案。
- LIBERO 和 RMBench 的离线数据读取、memory replay、token cache、mask 和 manifest 入口已经形成。
- LIBERO-Plus 已经明确为资源未确认状态，不进入可执行 pipeline。
- 仓库检查默认不触发训练 smoke，符合当前“暂不做训练接口”的要求。

后续如果要继续推进，需要先重新确认是否恢复训练或真实 eval；在确认之前，不应新增训练接口、不应跑 checkpoint/server 闭环 eval。
