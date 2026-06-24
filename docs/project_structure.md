# HiMem-Bridge-VLA 工程结构约定

这份约定的目的很简单：实验参数、模型实现、训练入口、评估工具和产物记录不要互相污染。当前代码主线是 H32 single-token coarse planner + BridgeAttention 集成。旧的 H64 suffix queue 和 transition trigger 方案已经从 active path 中移除。

## 目录职责

```text
configs/
  bridge_himem/            Bridge-HiMem 共享默认值和实验 overlay
  datasets/                训练数据配置
  deepspeed/               DeepSpeed 配置
  libero_profiles/         LIBERO smoke/full-eval 环境 profile

coarse_planner/            H32 cache、action-intent AE、planner 训练和评估
himem_bridge_vla/          package code: config、dataset、model、runtime helpers
evaluations/libero/        LIBERO client、action 协议、result summary
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
docs/engineering_reproducibility.md
docs/benchmark_plan.md
docs/bridge_himem_design.md
docs/coarse_planner_design.md
docs/dual_fifo_visual_memory_design_zh.md
docs/dual_fifo_visual_memory_qa_zh.md
coarse_planner/README.md
configs/README.md
scripts/README.md
```

旧 transition-trigger 和 H64 suffix planner 文档已经删除。后续如果恢复那条路线，应新建明确标记的新设计，而不是复活旧文档里的默认假设。

## 配置规则

- 新 Bridge-HiMem 实验只改 `configs/bridge_himem/experiments/*.yaml`，不要在模型里写死实验参数。
- 共享默认值只改 `configs/bridge_himem/base.yaml`。
- H32 standalone planner 配置只放在 `coarse_planner/configs/`。
- 当前 H32 planner 路线必须保持 `coarse_planner.num_plan_steps=1` 和 `action_head.horizon=32`。
- `coarse_planner.input_memory` 当前必须为 `false`。
- 修改 YAML 后先跑 `python scripts/validate_bridge_himem_configs.py`。

## 分工边界

- `bridge_himem_config.py`：配置 schema、继承、校验、兼容旧字段。
- `experiment_config.py`：训练/模型共用 config 解析。
- `model/bridge`：BridgeAttention 和 bridge token 生成。
- `model/himem`：当前 Dual-FIFO Visual Token Memory 的 entry、FIFO 读写、mask 和压缩模块。
- `model/planner`：只实现 Coarse Planner，不读取 memory，不维护 plan session。
- `coarse_planner/`：standalone cache、AE、planner warm-up、评估和导出。
- `model/himem_bridge_vla.py`：连接 VLM、bridge、planner 和 action head；BridgeAttention memory 接入前不保留旧 runtime memory。
- `dataset/action_segments.py`：future action segment 切分和 segment mask。
- `scripts/himem_server.py`：模型服务和 server protocol；当前路径不加载 transition trigger。
- `scripts/train.py`：训练流程、日志、checkpoint，不新增模型结构。
