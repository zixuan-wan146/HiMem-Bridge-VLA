# HiMem-Bridge-VLA 工程结构约定

这份约定的目的很简单：实验参数、模型实现、训练入口、评估工具和产物记录不要互相污染。当前研究主线是 short visual memory + progress-state planner。旧的 H64 suffix queue、transition trigger、Dual-FIFO long visual memory 已经从 active path 中移除。

## 目录职责

```text
configs/
  bridge_himem/            Bridge-HiMem 共享默认值和实验 overlay
  datasets/                训练数据配置
  deepspeed/               DeepSpeed 配置
  libero_profiles/         LIBERO smoke/full-eval 环境 profile

himem_bridge_vla/          package code: config、dataset、model、runtime helpers
evaluations/libero/        LIBERO client、action 协议、result summary
evaluations/rmbench/       RMBench policy adapter 和 eval plan helpers
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

- 新 Bridge-HiMem 实验只改 `configs/bridge_himem/experiments/*.yaml`，不要在模型里写死实验参数。
- 共享默认值只改 `configs/bridge_himem/base.yaml`。
- Progress-state planner 配置应使用清晰的 experiment 名称，避免复用历史实验语义。
- 修改 YAML 后先跑 `python scripts/validate_bridge_himem_configs.py`。

## 分工边界

- `bridge_himem_config.py`：配置 schema、继承、校验、兼容旧字段。
- `experiment_config.py`：训练/模型共用 config 解析。
- `model/bridge`：legacy bridge modules 和 bridge token 生成。
- `model/himem`：short visual-token memory 相关结构；旧 long visual FIFO 不再作为主线。
- `model/planner`：progress-state planner、progress state updater、condition builder，以及 action segment autoencoder。
- `model/himem_bridge_vla.py`：主模型入口；direct bridge 模式连接 VLM hidden states、short memory、progress planner plan token、state 和 flow-matching action head。
- `dataset/action_segments.py`：future action segment 切分和 segment mask。
- `scripts/himem_server.py`：模型服务和 server protocol；当前路径不加载 transition trigger。
- `himem_bridge_vla/training/stage1/`：active LIBERO Stage1 trajectory-window token-cache 训练逻辑。
- `scripts/train_stage1.py`：active Stage1 训练入口；旧 `scripts/train.py` 是混合历史入口，不作为当前主线。
