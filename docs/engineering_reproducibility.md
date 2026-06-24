# 工程化与可复现约定

本文记录当前 HiMem-Bridge-VLA 核心代码的工程边界和可复现入口。目标是减少隐式状态，让训练、eval、benchmark 检查都能从固定入口复现。

## 核心边界

```text
himem_bridge_vla/bridge_himem_config.py   Bridge / memory / planner 配置 schema
himem_bridge_vla/experiment_config.py     训练和模型共用配置解析
himem_bridge_vla/model/bridge/            BridgeAttention
himem_bridge_vla/model/himem/             Dual-FIFO Visual Token Memory
himem_bridge_vla/model/planner/           H32 Coarse Planner
himem_bridge_vla/model/himem_bridge_vla.py VLM + Bridge + Planner + ActionHead 主模型
scripts/train.py                          训练入口
scripts/himem_server.py                   websocket 推理服务
evaluations/libero/                       LIBERO eval client 和结果统计
```

当前 memory 只保留 Dual-FIFO visual memory。旧的检索式 memory bank、boundary writer、segment accumulator 已经删除，避免和当前方案混用。

## 可复现检查

每次重要训练或评估前，先跑轻量检查：

```bash
python scripts/validate_bridge_himem_configs.py
python scripts/validate_training_configs.py
python scripts/inspect_benchmarks.py --data-root "$AUTODL_TMP" --output run_outputs/benchmark_inventory.json --allow-missing
python scripts/build_libero_memory_replay_index.py --libero-root "$AUTODL_TMP/libero/datasets" --output run_outputs/libero_memory_replay.jsonl
python scripts/build_rmbench_norm_stats.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_norm_stats.json --metadata-output run_outputs/rmbench_norm_stats.metadata.json
python scripts/build_rmbench_memory_replay_index.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_memory_replay.jsonl
python scripts/build_memory_replay_token_cache.py --benchmark LIBERO --data-root "$AUTODL_TMP/libero/datasets" --index run_outputs/libero_memory_replay.jsonl --output-root "$AUTODL_TMP/token_caches/libero_memory_replay" --encoder image_stats --max-samples 2
python scripts/build_memory_replay_token_cache.py --benchmark RMBench --data-root "$AUTODL_TMP/benchmarks/RMBench" --index run_outputs/rmbench_memory_replay.jsonl --output-root "$AUTODL_TMP/token_caches/rmbench_memory_replay" --encoder image_stats --max-samples 2
python scripts/install_rmbench_policy_adapter.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --force
python scripts/plan_rmbench_eval.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_eval_plan.md --mode direct --tasks observe_and_pickup press_button
HIMEM_RMBENCH_DRY_RUN=1 bash scripts/run_rmbench_eval.sh
HIMEM_RMBENCH_PLAN_ONLY=1 HIMEM_RMBENCH_TASKS=press_button bash scripts/run_rmbench_eval.sh
```

上面两个 token cache 命令使用 `image_stats`，只用于检查数据读取、mask、shard 和 manifest 是否通。正式视觉塔缓存后续再单独安排；当前不新增训练入口，也不把 cache 接入 Bridge/action-head 训练。

生成 cache 后，数据读取统一走：

```python
from torch.utils.data import DataLoader

from himem_bridge_vla.dataset import MemoryTokenCacheDataset
from himem_bridge_vla.dataset import collate_memory_token_cache_samples

dataset = MemoryTokenCacheDataset("$AUTODL_TMP/token_caches/libero_memory_replay")
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_memory_token_cache_samples)
```

`collate_memory_token_cache_samples` 会保留 current visual tokens、short visual tokens、`short_steps`、`short_mask`、state、future actions 和 `action_mask`。其中 `short_memory` 字段已经按样本构造成 `MemoryReadResult`，可直接送入 memory compressor；Bridge-Attention 训练消费仍按当前计划暂缓。

memory context 构造的共享入口：

```python
from himem_bridge_vla.training import build_token_cache_memory_context

memory_batch = build_token_cache_memory_context(batch, memory_compressor)
model_kwargs = memory_batch.as_model_kwargs()
```

`model_kwargs` 包含 `memory_context` 和 `memory_context_mask`。`memory_context_mask=False` 的 padding token 会保持全零，并在 Bridge-Attention 的 condition path 中被 mask 掉，避免 padding slot 被当作有效 null token。当前只保留这个数据到 memory context 的共享构造入口；不继续扩展训练接口。

完整仓库检查：

```bash
PYTHON="$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/check_repo.sh
```

默认检查不运行训练 smoke 测试。如果后续重新决定把训练 smoke 纳入质量门，需要显式打开：

```bash
HIMEM_CHECK_INCLUDE_TRAINING=1 PYTHON="$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/check_repo.sh
```

如果只做代码级验证：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" -m pytest -q
git diff --check
```

## 实验快照

训练入口应继续使用 `himem_bridge_vla.reproducibility.write_experiment_snapshot` 记录：

```text
resolved_config.json
environment.json
reproducibility.json
```

这些文件需要包含：

- resolved config。
- git commit / dirty 状态。
- Python / torch / CUDA / package 版本。
- 安全环境变量，例如 `HF_ENDPOINT`、`HF_HOME`、`CUDA_VISIBLE_DEVICES`。
- 运行命令和当前工作目录。

## 数据盘约定

远端数据盘根目录：

```text
$AUTODL_TMP
```

推荐布局：

```text
$AUTODL_TMP/HiMem-Bridge-VLA      repo
$AUTODL_TMP/libero/datasets       LIBERO demonstrations
$AUTODL_TMP/benchmarks/RMBench    RMBench repo, assets, data
$AUTODL_TMP/runs                  training/eval outputs
$AUTODL_TMP/checkpoints           checkpoints
$AUTODL_TMP/token_caches          replay visual token caches
$AUTODL_TMP/hf-home               Hugging Face cache
```

不要把大数据、模型权重、训练产物放进 git 工作区。

## 当前推进顺序

1. 保持 H32 planner 和 BridgeAttention 主线可测。
2. 保持 Dual-FIFO visual memory 独立可测，暂不接 BridgeAttention。
3. 为 LIBERO / RMBench / LIBERO-Plus 建立统一 benchmark inventory 和状态文档。
4. LIBERO / RMBench 已有 memory replay index builder、frame reader、PyTorch-compatible dataset/collate、visual token cache builder、token cache dataset 和 memory 侧训练 smoke adapter。
5. RMBench 已有轻量 HDF5 reader、normalization stats builder、memory replay index builder、token cache smoke 入口、cache dataset 读取路径和 eval command planner。
6. RMBench 已有 official policy adapter、安装脚本、manifest writer 和 run wrapper；当前只保留 plan-only / dry-run 级别检查，不推进真实 checkpoint + server 的端到端 eval smoke。
7. 暂不推进正式 Bridge/action-head 训练接入、训练接口扩展、LIBERO-Plus robustness eval wrapper。
