# HiMem-Bridge-VLA 工程结构约定

这份约定的目的很简单：实验参数、模型实现、训练入口、评估工具和产物记录不要互相污染。当前研究主线是 short visual memory + progress-state planner。旧的 H64 suffix queue、transition trigger、Dual-FIFO long visual memory 已经从 active path 中移除。

## 目录职责

```text
configs/
  models/bridge_himem/     Bridge-HiMem 共享默认值
  experiments/bridge_himem/ Bridge-HiMem 实验 overlay
  datasets/                训练数据配置
  runtime/libero_profiles/ LIBERO smoke/full-eval 环境 profile

src/himem_bridge_vla/      package code: core、config、dataset、model、runtime、benchmark helpers
evaluations/legacy/        legacy LIBERO/RMBench 兼容代码和官方 policy adapter
scripts/                  train/server/repo gate、preflight、下载、评估编排、报告工具
tests/                    轻量单测，不下载模型权重
docs/                     当前设计说明和工程约定
referen-repo/             历史已跟踪 reference repositories，保留原路径避免大规模 rename
reference-repo/           新增 source-only 外部参考快照，例如 VLA-Adapter
```

大产物不进 git，统一放在远端数据盘：

```text
$AUTODL_TMP/runs/        训练、评估和 runtime 输出
$AUTODL_TMP/datasets/    数据集和转换后的 cache
$AUTODL_TMP/checkpoints/ 模型 checkpoint
$AUTODL_TMP/hf-home/     Hugging Face cache
```

repo 内的 `run_outputs/` 只作为临时评测输出目录使用，应保持 git ignored。

## 当前文档入口

```text
README.md
Plan.md
docs/current_project_state.md
docs/progress_state_planner_design_zh.md
docs/engineering_reproducibility.md
docs/benchmark_plan.md
docs/bridge_himem_design.md
docs/direct_bridge_attention_design_zh.md
docs/vla_adapter_bridge_attention_notes_zh.md
configs/README.md
scripts/README.md
```

旧 transition-trigger、H64 suffix planner、Dual-FIFO long visual memory、H32 action-latent planner 主线文档已经删除或降级。后续如果恢复某条路线，应新建明确标记的新设计，而不是复活旧文档里的默认假设。

## 配置规则

- 新 Bridge-HiMem 实验只改 `configs/experiments/bridge_himem/*.yaml`，不要在模型里写死实验参数。
- 共享默认值只改 `configs/models/bridge_himem/base.yaml`。
- Progress-state planner 配置应使用清晰的 experiment 名称，避免复用历史实验语义。
- 修改 YAML 后先跑 `python scripts/quality/validate_bridge_himem_configs.py`。

## 分工边界

- `src/himem_bridge_vla/core/`：全项目共享 contract、常量、错误、路径工具。
- `src/himem_bridge_vla/bridge_himem_config.py`：配置 schema、继承、校验、兼容旧字段。
- `src/himem_bridge_vla/experiment_config.py`：训练/模型共用 config 解析。
- `src/himem_bridge_vla/model/bridge`：legacy bridge modules 和 bridge token 生成。
- `src/himem_bridge_vla/model/himem`：short visual-token memory 相关结构；旧 long visual FIFO 不再作为主线。
- `src/himem_bridge_vla/model/planner`：progress-state planner、progress state updater、condition builder，以及 action segment autoencoder。
- `src/himem_bridge_vla/model/himem_bridge_vla.py`：主模型入口；direct bridge 模式连接 VLM hidden states、short memory、progress planner plan token、state 和 flow-matching action head。
- `src/himem_bridge_vla/dataset/action_segments.py`：future action segment 切分和 segment mask。
- `src/himem_bridge_vla/runtime/`：benchmark-neutral runtime contract、feature extraction、memory builder、inference engine、websocket server。
- `src/himem_bridge_vla/benchmarks/`：LIBERO/RMBench obs、state、action、history、runner 和 adapter 逻辑。
- `scripts/serve/serve_policy.py`：当前 websocket 推理服务入口。
- `src/himem_bridge_vla/training/stage1/`：active LIBERO Stage1 episode-level fixed-replan-node feature-cache 训练逻辑。
- `scripts/train/stage1/libero.py`：active LIBERO Stage1 训练入口。
