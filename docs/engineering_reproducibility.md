# 工程化与可复现约定

本文记录当前 HiMem-Bridge-VLA 核心代码的工程边界和可复现入口。目标是减少隐式状态，让训练、eval、benchmark 检查都能从固定入口复现。

## 核心边界

```text
himem_bridge_vla/bridge_himem_config.py    Bridge / memory / planner 配置 schema
himem_bridge_vla/experiment_config.py      训练和模型共用配置解析
himem_bridge_vla/model/bridge/             legacy bridge modules
himem_bridge_vla/model/himem/              short visual-token memory support
himem_bridge_vla/model/planner/            progress-state planner + legacy H32 baseline
himem_bridge_vla/model/himem_bridge_vla.py 现有主模型入口；当前设计阶段先处理 planner warmup
himem_bridge_vla/dataset/libero_progress_warmup.py   LIBERO progress warm-up cache / dataset
himem_bridge_vla/dataset/rmbench_progress_warmup.py  RMBench progress warm-up cache / dataset
scripts/train.py                           训练入口
scripts/himem_server.py                    websocket 推理服务
evaluations/libero/                        LIBERO eval client 和结果统计
evaluations/rmbench/                       RMBench policy adapter 和 eval plan helpers
```

当前 memory 分工：

```text
short memory: independent visual-token context
long memory: planner-coupled recurrent task-progress state
```

旧的检索式 memory bank、boundary writer、segment accumulator、Dual-FIFO long visual FIFO 已经不再作为当前主线。

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

上面两个 token cache 命令使用 `image_stats`，只用于检查数据读取、mask、shard 和 manifest 是否通。

生成 cache 后，短期视觉记忆的数据读取统一走：

```python
from torch.utils.data import DataLoader

from himem_bridge_vla.dataset import MemoryTokenCacheDataset
from himem_bridge_vla.dataset import collate_memory_token_cache_samples

dataset = MemoryTokenCacheDataset("$AUTODL_TMP/token_caches/libero_memory_replay")
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_memory_token_cache_samples)
```

`collate_memory_token_cache_samples` 会保留 current visual tokens、short visual tokens、`short_steps`、`short_mask`、state、future actions 和 `action_mask`。这些路径现在主要服务于 short visual memory 和数据 IO 检查。

旧的 token-cache batch 到 `memory_context` / `memory_context_mask` 构造入口仍可作为 smoke 工具保留，但不再定义 long memory。新的 long memory 应由 progress-state planner recurrent update 得到。

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

## Progress-State Warm-Up 入口

LIBERO 和 RMBench 的 action protocol 不同，progress warm-up cache 分开构建，不走混合通用脚本。

LIBERO H32/R16 cache：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/build_libero_progress_vl_embedding_cache.py \
  --index run_outputs/libero_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w4" \
  --segment-ae-checkpoint "$AUTODL_TMP/runs/coarse_planner/libero_h32_intent_ae_v1/best.pt" \
  --horizon 32 \
  --replan-stride 16 \
  --burnin-replan-steps 8 \
  --loss-replan-steps 4 \
  --vl-batch-size 192 \
  --storage-dtype bfloat16
```

已有 LIBERO step cache 重新切 window 时使用：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/rewindow_progress_warmup_cache.py \
  --source-cache "$AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16" \
  --output-root "$AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w4" \
  --loss-replan-steps 4
```

RMBench 14-dim action intent AE：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/train_rmbench_action_segment_autoencoder.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --index run_outputs/rmbench_memory_replay.jsonl \
  --norm-stats run_outputs/rmbench_norm_stats.json \
  --robot-key rmbench \
  --output-dir "$AUTODL_TMP/runs/progress_warmup/rmbench_h32_intent_ae_v1" \
  --batch-size 4096 \
  --max-steps 1000 \
  --samples-per-epoch 32768 \
  --eval-interval 50
```

RMBench H32/R16 cache：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/build_rmbench_progress_vl_embedding_cache.py \
  --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" \
  --index run_outputs/rmbench_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/rmbench_progress_vl_embedding_h32_r16_w4" \
  --segment-ae-checkpoint "$AUTODL_TMP/runs/progress_warmup/rmbench_h32_intent_ae_v1/best.pt" \
  --norm-stats run_outputs/rmbench_norm_stats.json \
  --robot-key rmbench \
  --horizon 32 \
  --replan-stride 16 \
  --burnin-replan-steps 8 \
  --loss-replan-steps 4 \
  --vl-batch-size 192 \
  --storage-dtype bfloat16
```

Progress-state planner warm-up：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/train_progress_state_planner.py \
  --cache-manifest "$AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w4" \
  --output-dir "$AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1" \
  --device cuda \
  --batch-size 12800 \
  --max-steps 1000 \
  --samples-per-epoch 12800 \
  --sampling-alpha 0.5 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --num-workers 0 \
  --seed 42 \
  --val-fraction 0.1 \
  --eval-interval 10 \
  --log-interval 5 \
  --ckpt-interval 100
```

RMBench window=8 cache 把 `--loss-replan-steps 4` 改为 `8`，输出目录改成 `rmbench_progress_vl_embedding_h32_r16_w8`。训练命令只需要把 `--cache-manifest` 和 `--output-dir` 换成对应 RMBench 路径。

## 已产出 Warm-Up Artifacts

当前可直接复用的 cache：

```text
LIBERO W=4:
  /root/autodl-tmp/token_caches/libero_progress_vl_embedding_h32_r16_w4
  step_count=18199
  window_count=12199

RMBench W=4:
  /root/autodl-tmp/token_caches/rmbench_progress_vl_embedding_h32_r16_w4
  step_count=16676
  window_count=15326

RMBench W=8:
  /root/autodl-tmp/token_caches/rmbench_progress_vl_embedding_h32_r16_w8
  step_count=16676
  window_count=13526
```

当前可直接复用的 checkpoints：

```text
RMBench intent AE:
  /root/autodl-tmp/runs/progress_warmup/rmbench_h32_intent_ae_v1/best.pt
  best step: 950
  val_loss: 0.015030

LIBERO W=4 progress warm-up:
  /root/autodl-tmp/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt
  best step: 310
  val_loss: 0.017872

RMBench W=4 progress warm-up:
  /root/autodl-tmp/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt
  best step: 590
  val_loss: 0.001225

RMBench W=8 progress warm-up:
  /root/autodl-tmp/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w8_bs6656_epval_v1/best.pt
  stopped after: step_000700.pt
  best step: 660
  val_loss: 0.001016
```

训练日志解析结果保存在对应 run 目录：

```text
train_history_from_log.json
early_stop_summary.json
```
