# HiMem-Bridge-VLA 工程结构约定

这份约定的目的很简单：实验参数、模型实现、训练入口、评估工具和产物记录不要互相污染。当前研究主线是 short visual memory + progress-state planner。旧的 H64 suffix queue、transition trigger、Dual-FIFO long visual memory 已经从 active path 中移除。

## 目录职责

```text
configs/
  bridge_himem/            Bridge-HiMem 共享默认值和实验 overlay
  datasets/                训练数据配置
  deepspeed/               DeepSpeed 配置
  libero_profiles/         LIBERO smoke/full-eval 环境 profile

coarse_planner/            legacy H32 action-intent cache、AE、planner baseline
himem_bridge_vla/          package code: config、dataset、model、runtime helpers
evaluations/libero/        LIBERO client、action 协议、result summary
evaluations/rmbench/       RMBench policy adapter 和 eval plan helpers
scripts/                  train/server/repo gate、preflight、下载、评估编排、报告工具
tests/                    轻量单测，不下载模型权重
docs/                     当前设计说明和工程约定
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
coarse_planner/README.md
configs/README.md
scripts/README.md
```

旧 transition-trigger、H64 suffix planner、Dual-FIFO long visual memory、H32 action-latent planner 主线文档已经删除或降级。后续如果恢复某条路线，应新建明确标记的新设计，而不是复活旧文档里的默认假设。

## 配置规则

- 新 Bridge-HiMem 实验只改 `configs/bridge_himem/experiments/*.yaml`，不要在模型里写死实验参数。
- 共享默认值只改 `configs/bridge_himem/base.yaml`。
- Legacy H32 standalone planner 配置只放在 `coarse_planner/configs/`。
- New progress-state planner 配置应使用新的 experiment 名称，不要复用旧 `coarse_planner_crosskv` 语义。
- 修改 YAML 后先跑 `python scripts/validate_bridge_himem_configs.py`。

## 分工边界

- `bridge_himem_config.py`：配置 schema、继承、校验、兼容旧字段。
- `experiment_config.py`：训练/模型共用 config 解析。
- `model/bridge`：legacy bridge modules 和 bridge token 生成。
- `model/himem`：short visual-token memory 相关结构；旧 long visual FIFO 不再作为主线。
- `model/planner`：新增 progress-state planner、progress state updater、condition builder；legacy `CoarsePlanner` 保留为 baseline。
- `coarse_planner/`：standalone cache、AE、H32 action-latent baseline 训练和评估。
- `model/himem_bridge_vla.py`：现有主模型入口；当前设计阶段先实现 progress-state planner warmup，不扩展动作端接口。
- `dataset/action_segments.py`：future action segment 切分和 segment mask。
- `scripts/himem_server.py`：模型服务和 server protocol；当前路径不加载 transition trigger。
- `scripts/train.py`：训练流程、日志、checkpoint；新增结构必须通过 config 接入。
