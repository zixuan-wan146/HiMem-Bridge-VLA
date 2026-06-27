# 工程化与可复现约定

本文记录当前 HiMem-Bridge-VLA 核心代码的工程边界和可复现入口。目标是减少隐式状态，让训练、eval、benchmark 检查都能从固定入口复现。

## 核心边界

```text
himem_bridge_vla/bridge_himem_config.py    Bridge / memory / planner 配置 schema
himem_bridge_vla/experiment_config.py      训练和模型共用配置解析
himem_bridge_vla/training/stage1/          active LIBERO Stage1 trajectory-window token-cache training
himem_bridge_vla/model/bridge/             legacy bridge modules
himem_bridge_vla/model/himem/              short visual-token memory support
himem_bridge_vla/model/planner/            progress-state planner + legacy H32 baseline
himem_bridge_vla/model/action_head/flow_matching.py direct bridge-attn flow-matching action head
himem_bridge_vla/model/himem_bridge_vla.py 主模型入口；direct bridge 模式直接连接 VLM hidden states、short memory、plan slots 和 state
himem_bridge_vla/dataset/libero_progress_warmup.py   LIBERO progress warm-up cache / dataset
himem_bridge_vla/dataset/rmbench_progress_warmup.py  RMBench progress warm-up cache / dataset
scripts/train/stage1/libero.py                    active LIBERO Stage1 训练入口
scripts/serve/serve_policy.py                    websocket 推理服务
evaluations/legacy/libero/                 legacy LIBERO eval client 和结果统计
evaluations/legacy/rmbench/                legacy RMBench policy adapter 和 eval plan helpers
```

当前 memory 分工：

```text
short memory: independent visual-token context
long memory: planner-coupled recurrent task-progress state
```

旧的检索式 memory bank、boundary writer、segment accumulator、Dual-FIFO long visual FIFO 代码路径已经删除或降级为历史背景，不再作为可运行入口。

## 可复现检查

每次重要训练或评估前，先跑轻量检查：

```bash
python scripts/quality/validate_bridge_himem_configs.py
python scripts/quality/validate_training_configs.py
python scripts/eval/inspect_benchmarks.py --data-root "$AUTODL_TMP" --output run_outputs/benchmark_inventory.json --allow-missing
python scripts/cache/build_libero_memory_replay_index.py --libero-root "$AUTODL_TMP/libero/datasets" --output run_outputs/libero_memory_replay.jsonl
python scripts/cache/build_rmbench_norm_stats.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_norm_stats.json --metadata-output run_outputs/rmbench_norm_stats.metadata.json
python scripts/cache/build_rmbench_memory_replay_index.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_memory_replay.jsonl
python scripts/cache/build_memory_replay_token_cache.py --benchmark LIBERO --data-root "$AUTODL_TMP/libero/datasets" --index run_outputs/libero_memory_replay.jsonl --output-root "$AUTODL_TMP/token_caches/libero_memory_replay" --encoder image_stats --max-samples 2
python scripts/cache/build_memory_replay_token_cache.py --benchmark RMBench --data-root "$AUTODL_TMP/benchmarks/RMBench" --index run_outputs/rmbench_memory_replay.jsonl --output-root "$AUTODL_TMP/token_caches/rmbench_memory_replay" --encoder image_stats --max-samples 2
python scripts/quality/smoke_direct_bridge_inference.py --preset final
python scripts/setup/install_rmbench_policy_adapter.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --force
python scripts/eval/plan_rmbench_eval.py --rmbench-root "$AUTODL_TMP/benchmarks/RMBench" --output run_outputs/rmbench_eval_plan.md --mode direct --tasks observe_and_pickup press_button
HIMEM_RMBENCH_DRY_RUN=1 bash scripts/eval/run_rmbench_eval.sh
HIMEM_RMBENCH_PLAN_ONLY=1 HIMEM_RMBENCH_TASKS=press_button bash scripts/eval/run_rmbench_eval.sh
```

上面两个 token cache 命令使用 `image_stats`，只用于检查数据读取、mask、shard 和 manifest 是否通。

构建最终 direct bridge 训练 cache 时，需要同时保存 current VLM hidden-state token layers：

```bash
python scripts/cache/build_memory_replay_token_cache.py \
  --benchmark LIBERO \
  --data-root "$AUTODL_TMP/libero/datasets" \
  --index run_outputs/libero_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/libero_memory_replay_internvl3_hidden_l3_6_9_12" \
  --encoder internvl3 \
  --include-vlm-hidden-states \
  --hidden-state-layers 3 6 9 12 \
  --storage-dtype bfloat16 \
  --device cuda
```

这个 cache 会同时写入：

```text
current_tokens_by_view      # visual tower tokens, used for short-memory IO checks and non-hidden-state smoke
short_tokens_by_view        # visual tower tokens, used as short memory
current_hidden_states       # selected language/VLM hidden-state token layers for direct bridge raw features
prompt                      # instruction used to produce current_hidden_states
executed_actions
executed_action_mask
future_actions
```

生成 cache 后，短期视觉记忆的数据读取统一走：

```python
from torch.utils.data import DataLoader

from himem_bridge_vla.dataset import MemoryTokenCacheDataset
from himem_bridge_vla.dataset import collate_memory_token_cache_samples

dataset = MemoryTokenCacheDataset("$AUTODL_TMP/token_caches/libero_memory_replay")
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_memory_token_cache_samples)
```

`collate_memory_token_cache_samples` 会保留 current visual tokens、short visual tokens、`short_steps`、`short_mask`、state、future actions 和 `action_mask`。这些路径现在主要服务于 short visual memory 和数据 IO 检查。

direct bridge-attn 训练烟测可以直接使用 visual-token cache：

```text
dataset_type: memory_token_cache
dataset_config_path: $AUTODL_TMP/token_caches/libero_memory_replay/manifest.json
bridge_himem_config: configs/models/bridge_himem/base.yaml
horizon: 32
```

这个路径使用 `collate_direct_bridge_token_cache_samples`，会构造：

```text
fused_tokens
vlm_hidden_states        # optional, preferred for final direct bridge raw-layer training
memory_context
memory_context_mask
short_memory_time_ids
planner_vl_summary       # required for active Stage1; VLM last-valid-token hidden state
states
actions
action_mask
```

其中 short memory 每个历史时刻固定 pack 到 `memory_entry_tokens=16`，两个历史时刻合计 32 tokens。

注意：最终 direct bridge 训练 cache 必须使用 `--encoder internvl3 --include-vlm-hidden-states --hidden-state-layers 3 6 9 12`。这会同时缓存 InternVL3 visual-tower tokens、selected language/VLM hidden-state token layers，以及 `planner_vl_summary`。`current_tokens_by_view` 供 short memory 与 IO 检查使用；`current_hidden_states` 会在 collate 后变成 `vlm_hidden_states`，并由 `scripts/train/stage1/libero.py` 传给 direct bridge action head。`planner_vl_summary` 会直接喂给冻结的 progress planner。只包含 `current_tokens_by_view` 的 cache 仅用于 smoke 或降级调试，不作为最终训练输入。

Stage1 action-side policy 训练必须走 trajectory-window cache 入口：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/train/stage1/libero.py \
  --config configs/training/stage1/libero/libero_10_direct_progress_w4.yaml
```

该入口会拒绝 frame-level random batch，因为 frozen W4 ProgressPlanner 内部维护递推 long memory：

```text
M_k = f(M_{k-1}, x_k)
```

每个 batch item 是一个 episode 内连续 replan window：先 burn-in 更新 `M_k` 但不计 loss，再在 loss window 上计算 masked flow-matching velocity loss。

注意 cache 类型不要混用：

```text
progress-state planner warm-up:
  format = *_progress_vl_embedding_warmup_cache
  content = pooled VL embeddings, state, executed action summary, target intent z_k

direct bridge / short memory:
  format = memory_replay_visual_token_cache
  content = current visual tokens, optional current VLM hidden states, short visual tokens, state, future actions
```

`scripts/quality/smoke_direct_bridge_token_cache_training.py --manifest` 只接受 `memory_replay_visual_token_cache`。已有的 LIBERO/RMBench progress warm-up cache 不能作为 direct bridge visual-token cache 使用。

visual-token cache 训练烟测：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/quality/smoke_direct_bridge_token_cache_training.py \
  --preset auto \
  --manifest "$AUTODL_TMP/token_caches/libero_memory_replay" \
  --device cpu \
  --steps 1 \
  --batch-size 1
```

带 progress-state planner checkpoint 的 direct bridge 训练烟测：

```bash
python scripts/cache/build_memory_replay_token_cache.py \
  --benchmark LIBERO \
  --data-root "$AUTODL_TMP/libero/datasets" \
  --index run_outputs/libero_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/libero_memory_replay_image_stats_hidden_smoke" \
  --encoder image_stats \
  --image-stats-hidden-dim 896 \
  --image-stats-tokens-per-view 32 \
  --include-vlm-hidden-states \
  --hidden-state-layers 3 6 9 12 \
  --max-samples 2 \
  --max-samples-per-shard 2 \
  --storage-dtype float32

"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/quality/smoke_direct_bridge_token_cache_training.py \
  --preset final \
  --manifest "$AUTODL_TMP/token_caches/libero_memory_replay_image_stats_hidden_smoke" \
  --device auto \
  --steps 1 \
  --batch-size 1 \
  --action-horizon 32 \
  --memory-entry-tokens 16 \
  --progress-planner-checkpoint "$AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt"
```

这里的 visual-token cache hidden dim 必须和 progress checkpoint 一致。LIBERO checkpoint 是 `hidden_dim=896`，因此不能用 8 维 `image_stats` IO smoke cache 来跑这条命令。

这个 smoke 会检查 visual-token cache batch 中的：

```text
fused_tokens
vlm_hidden_states
memory_context
executed_actions
executed_action_mask
states
actions
```

并验证 progress checkpoint 生成的 `[B, 1, 896]` plan token 可以进入 direct bridge action head 完成一次 flow-matching loss backward。

direct bridge 推理烟测可以加载 progress-state planner warm-up checkpoint，让 checkpoint 生成真实 plan token，再进入 direct bridge action head：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/quality/smoke_direct_bridge_inference.py \
  --preset final \
  --device auto \
  --progress-planner-checkpoint "$AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt"
```

该烟测检查：

```text
progress checkpoint format
ProgressStatePlanner checkpoint config loading
progress planner token shape: [B, 1, 896]
direct bridge action shape: [B, 32, action_dim]
finite action values
action_mask keeps masked dimensions fixed at zero
```

完整仓库检查：

```bash
PYTHON="$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/quality/check_repo.sh
```

如果只做代码级验证：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" -m pytest -q
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/quality/smoke_direct_bridge_inference.py --preset final
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/quality/smoke_direct_bridge_token_cache_training.py --preset final --manifest "$AUTODL_TMP/token_caches/libero_memory_replay_image_stats_hidden_smoke" --device auto --steps 1 --batch-size 1 --action-horizon 32 --memory-entry-tokens 16 --progress-planner-checkpoint "$AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt"
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/quality/smoke_direct_bridge_inference.py --preset final --progress-planner-checkpoint "$AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt"
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
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/cache/build_libero_progress_vl_embedding_cache.py \
  --index run_outputs/libero_memory_replay.jsonl \
  --output-root "$AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w4" \
  --horizon 32 \
  --replan-stride 16 \
  --burnin-replan-steps 8 \
  --loss-replan-steps 4 \
  --vl-batch-size 192 \
  --storage-dtype bfloat16
```

已有 LIBERO step cache 重新切 window 时使用：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/cache/rewindow_progress_warmup_cache.py \
  --source-cache "$AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w8" \
  --output-root "$AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w4" \
  --loss-replan-steps 4
```

RMBench 14-dim action intent AE：

```bash
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/train/train_rmbench_action_segment_autoencoder.py \
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
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/cache/build_rmbench_progress_vl_embedding_cache.py \
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
"$AUTODL_TMP/miniforge3/envs/Evo1/bin/python" scripts/train/train_progress_state_planner.py \
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
  $AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w4
  step_count=18199
  window_count=12199

LIBERO W=8:
  $AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w8
  step_count=18199
  window_count=5429

RMBench W=4:
  $AUTODL_TMP/token_caches/rmbench_progress_vl_embedding_h32_r16_w4
  step_count=16676
  window_count=15326

RMBench W=8:
  $AUTODL_TMP/token_caches/rmbench_progress_vl_embedding_h32_r16_w8
  step_count=16676
  window_count=13526
```

当前可直接复用的 checkpoints：

```text
RMBench intent AE:
  $AUTODL_TMP/runs/progress_warmup/rmbench_h32_intent_ae_v1/best.pt
  best step: 950
  val_loss: 0.015030

LIBERO W=4 progress warm-up:
  $AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt
  best step: 310
  val_loss: 0.017872

LIBERO W=8 progress warm-up:
  $AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_bs6656_epval_v1/best.pt
  best step: 280
  val_loss: 0.021811

RMBench W=4 progress warm-up:
  $AUTODL_TMP/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt
  best step: 590
  val_loss: 0.001225

RMBench W=8 progress warm-up:
  $AUTODL_TMP/runs/progress_warmup/rmbench_progress_state_planner_h32_r16_w8_bs6656_epval_v1/best.pt
  stopped after: step 700
  per-step checkpoints: pruned after summary generation
  best step: 660
  val_loss: 0.001016
```

训练日志解析结果保存在对应 run 目录：

```text
train_history_from_log.json
early_stop_summary.json
```
