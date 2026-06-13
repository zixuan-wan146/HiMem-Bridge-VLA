# HiMem-Bridge-VLA 工程结构约定

这份约定的目的很简单：实验参数、模型实现、训练入口、评估工具和产物记录不要互相污染。
目录不一定一步到位迁移，但新增代码应按下面的边界放置。

## 目录职责

```text
configs/
  bridge_himem/
    base.yaml              Bridge-HiMem 共享默认值
    experiments/*.yaml     只写实验差异，使用 extends 继承 base
  datasets/*.yaml          训练数据配置
  deepspeed/*.json         DeepSpeed 配置
  libero_profiles/*.env    LIBERO smoke/full-eval 环境 profile
  calvin_profiles/*.env    CALVIN smoke/full-eval 环境 profile

himem_bridge_vla/
  bridge_himem_config.py   Bridge-HiMem YAML schema、extends 合并、参数校验
  experiment_config.py     训练和模型共用的最终配置解析
  reproducibility.py       seed、deterministic、run snapshot
  training_config.py       训练 CLI 参数和路径校验
  dataset/                 数据集结构、路径解析、样本读取
  model/                   可训练模型模块，只消费已经解析好的 config

evaluations/libero/         LIBERO client、action 协议、result summary
evaluations/calvin/         CALVIN client、action 协议、result summary
evaluations/metaworld/      legacy MetaWorld client
scripts/                   train/server/repo gate、preflight、下载、评估编排、报告工具
tests/                     轻量单测，不下载模型权重
docs/                      设计说明和工程约定
```

## 配置规则

- 新实验只能新增 `configs/bridge_himem/experiments/*.yaml`，不要在 `himem_bridge_vla/model` 里写死实验参数。
- 共享默认值只改 `configs/bridge_himem/base.yaml`。
- A/B clean 实验必须共享 memory writer、segment accumulator、VLM raw layers 和 action head 设置。
- `bridge.variant` 和 `memory.placement` 必须一致；这由 `BridgeHiMemConfig.validate()` 强制检查。
- 修改 YAML 后先跑：

```bash
python3 scripts/validate_bridge_himem_configs.py
```

## 复现规则

每次训练启动后，`save_dir` 会写：

- `resolved_config.json`：CLI + YAML `extends` 合并后的最终配置。
- `reproducibility.json`：命令、cwd、Python、平台、git commit/branch/dirty、seed、实验名。

复现实验时不要只看原始 YAML，因为 CLI 可能覆盖 seed、device、batch size、save_dir 等字段。
应以 run 目录里的 `resolved_config.json` 为准。

推荐训练入口：

```bash
python scripts/train.py \
  --dataset_config_path configs/datasets/simulation.yaml \
  --bridge_himem_config configs/bridge_himem/experiments/crosskv_clean.yaml \
  --seed 42 \
  --save_dir /root/autodl-tmp/himem_runs/crosskv_clean_seed42
```

需要更强复现时加 `--deterministic`。这会更慢，并且某些 CUDA kernel 仍可能有环境差异。

## 分工边界

- `bridge_himem_config.py`：只关心配置 schema、继承、校验、兼容旧字段。
- `experiment_config.py`：只负责把训练/模型需要的最终 config 解析出来。
- `reproducibility.py`：只负责 seed 和 run snapshot，不碰模型逻辑。
- `model/bridge`：只实现 BridgeAttention 和 bridge token 生成。
- `model/himem`：只实现 memory writer、segment accumulator、episode bank。
- `model/himem_bridge_vla.py`：只负责把 VLM、bridge、memory、action head 连接起来。
- `scripts/train.py`：只负责训练流程、日志、checkpoint，不新增模型结构。

这几个边界以后要尽量守住。否则最容易回到“参数散在脚本里、模型里、YAML 里各一份”的状态。
