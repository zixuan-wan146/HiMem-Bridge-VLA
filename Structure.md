# HiMem-Bridge-VLA 项目结构重构设计

## 0. 设计目标

本项目不应该继续通过补丁式脚本维护训练、评测、runtime 和 benchmark 协议。目标是建立一个清晰、可测试、可扩展的科研/开源项目结构，使以下几类逻辑互相隔离：

1. 模型结构：`HiMemBridgeVLA`、action head、progress planner、InternVL3。
2. 离线数据：LIBERO/RMBench demo 读取、replay index、token cache、progress warmup cache。
3. 训练流程：Stage1、progress warmup，其中通用训练 loop 和 benchmark-specific 数据/配置分开。
4. Runtime 推理：feature extraction、short memory、planner summary、action inference、websocket server。
5. Benchmark 协议：LIBERO/RMBench 各自的 obs、state、action、view、history、eval runner。
6. Evaluation 结果处理：metrics、summary、manifest、report。
7. Diagnostics：cache-vs-runtime、图像方向、gripper、planner summary、short memory probe。
8. Scripts：只作为命令行入口，不承载核心逻辑。

最终数据流应为：

```text
Benchmark Env
    ↓
BenchmarkAdapter
    ↓
PolicyRequest
    ↓
RuntimeInferenceEngine
    ↓
FeatureExtractor + MemoryBuilder
    ↓
HiMemBridgeVLA
    ↓
PolicyActionChunk
    ↓
BenchmarkAdapter
    ↓
Env Action
```

---

## 1. 总体目录结构

建议采用 `src/` layout：

```text
HiMem-Bridge-VLA/
├── pyproject.toml
├── README.md
├── .gitignore
├── .pre-commit-config.yaml
│
├── src/
│   └── himem_bridge_vla/
│       ├── __init__.py
│       ├── core/
│       ├── model/
│       ├── data/
│       ├── training/
│       ├── runtime/
│       ├── benchmarks/
│       ├── evaluation/
│       ├── diagnostics/
│       ├── configs/
│       └── utils/
│
├── scripts/
│   ├── train/
│   ├── cache/
│   ├── serve/
│   ├── eval/
│   ├── report/
│   ├── quality/
│   ├── diagnose/
│   ├── setup/
│   ├── maintenance/
│   └── legacy/
│
├── configs/
│   ├── model/
│   ├── training/
│   ├── data/
│   ├── runtime/
│   └── eval/
│
├── tests/
│   ├── core/
│   ├── model/
│   ├── data/
│   ├── training/
│   ├── runtime/
│   ├── benchmarks/
│   ├── evaluation/
│   └── diagnostics/
│
├── docs/
│   ├── architecture/
│   ├── benchmark_contracts/
│   ├── training/
│   ├── evaluation/
│   └── engineering/
│
├── evaluations/
│   └── legacy/
│
├── local_data/
├── run_outputs/
└── tools/
```

说明：

* `HiMem-Bridge-VLA/` 是 repository root，也就是项目根目录。
* `src/himem_bridge_vla/` 是 Python source package，也就是可 import 的源码主包。
* `scripts/` 只保留命令行入口，不写核心逻辑。
* `evaluations/` 不再承载新逻辑，只保留 legacy 兼容入口。

---

## 2. `src/himem_bridge_vla/core/`

职责：全项目共享的稳定类型、常量、错误类型、注册器。

```text
src/himem_bridge_vla/core/
├── __init__.py
├── types.py
├── constants.py
├── errors.py
├── registry.py
└── paths.py
```

### 负责内容

* `BenchmarkSpec`
* `PolicyRequest`
* `PolicyResponse`
* `PolicyActionChunk`
* `RuntimeFeatures`
* `ImageBundle`
* `ActionChunk`
* 全局常量
* 自定义异常
* repo/data path resolver

示例：

```python
@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    view_names: tuple[str, ...]
    state_dim: int
    action_dim: int
    replan_stride: int
    short_memory_offsets: tuple[int, ...]
```

```python
@dataclass(frozen=True)
class PolicyRequest:
    benchmark: str
    prompt: str
    images_by_view: dict[str, np.ndarray]
    state: np.ndarray
    action_dim: int
    short_memory_images_by_offset: dict[int, dict[str, np.ndarray]] | None
    executed_actions: np.ndarray | None
    executed_action_mask: np.ndarray | None
    reset_memory: bool = False
    robot_key: str | None = None
```

### 不负责内容

* 不放模型结构。
* 不放 benchmark obs/action 逻辑。
* 不放训练 loop。
* 不放 server 实现。

---

## 3. `src/himem_bridge_vla/model/`

职责：模型结构本身。

```text
src/himem_bridge_vla/model/
├── __init__.py
├── himem_bridge_vla.py
│
├── internvl3/
│   ├── __init__.py
│   ├── embedder.py
│   └── feature_types.py
│
├── action_head/
│   ├── __init__.py
│   ├── flow_matching.py
│   ├── direct_bridge_blocks.py
│   └── losses.py
│
├── planner/
│   ├── __init__.py
│   ├── progress_state.py
│   ├── action_segment_autoencoder.py
│   └── losses.py
│
└── bridge/
    ├── __init__.py
    └── adapters.py
```

### 负责内容

* `HiMemBridgeVLA`
* InternVL3 embedder wrapper
* flow-matching action head
* direct bridge-attention blocks
* progress-state planner
* action segment autoencoder
* 模型内部 forward / predict_action
* 模型内部 loss 或辅助 loss

### 不负责内容

* 不负责 LIBERO/RMBench obs 解析。
* 不负责 eval runner。
* 不负责 websocket server。
* 不负责 cache 文件读写。
* 不负责 benchmark-specific action decode。

模型层只接受 tensor contract：

```text
current_visual_tokens
vlm_hidden_states
planner_vl_summary
short_memory_tokens
short_memory_mask
short_memory_time_ids
state
executed_actions
executed_action_mask
```

不要让模型层直接感知 LIBERO 或 RMBench。

---

## 4. `src/himem_bridge_vla/data/`

职责：离线数据读取、replay index、token cache、progress warmup cache、normalization。

```text
src/himem_bridge_vla/data/
├── __init__.py
│
├── readers/
│   ├── __init__.py
│   ├── libero_reader.py
│   └── rmbench_reader.py
│
├── replay/
│   ├── __init__.py
│   ├── index.py
│   ├── frame_reader.py
│   └── samples.py
│
├── token_cache/
│   ├── __init__.py
│   ├── common.py
│   ├── builder.py
│   ├── dataset.py
│   ├── collate.py
│   ├── manifest.py
│   ├── encoders.py
│   ├── libero.py
│   └── rmbench.py
│
├── progress_cache/
│   ├── __init__.py
│   ├── common.py
│   ├── builder.py
│   ├── dataset.py
│   ├── collate.py
│   ├── manifest.py
│   ├── libero.py
│   └── rmbench.py
│
└── normalization/
    ├── __init__.py
    ├── stats.py
    └── minmax.py
```

## 4.1 `data/readers/`

负责：

* 读取 LIBERO HDF5 demo。
* 读取 RMBench demo。
* 返回 frame、action、state、images。
* 不负责在线 eval loop。

## 4.2 `data/replay/`

负责：

* 构建 memory replay index。
* 定义 current step。
* 定义 future action chunk。
* 定义 short memory step。
* 读取 replay row 对应的 current frame 和 short frames。

## 4.3 `data/token_cache/`

负责 Stage1 token cache。

### common 文件负责

* token cache manifest schema。
* token cache shard IO。
* token packing。
* 通用 dataset/collate 接口。
* 通用 visual token encoder。

### `token_cache/libero.py`

负责 LIBERO Stage1 token cache：

* LIBERO view names。
* LIBERO replay row 到 current/short images 的映射。
* LIBERO state/action normalization 适配。
* LIBERO planner_vl_summary 构建。
* LIBERO cache manifest extra metadata。

### `token_cache/rmbench.py`

负责 RMBench Stage1 token cache：

* RMBench view names。
* RMBench replay row 到 current/short images 的映射。
* RMBench state/action normalization 适配。
* RMBench planner_vl_summary 构建。
* RMBench cache manifest extra metadata。

### Stage1 token cache 必须包含

```text
current_visual_tokens
current_hidden_states
planner_vl_summary
short_tokens_by_view
short_steps
short_mask
current_state
executed_actions
executed_action_mask
future_actions
action_valid_count
```

注意：`planner_vl_summary` 必须和 progress warmup 使用同一种定义，不能让 Stage1 fallback 到 `mean(raw visual tokens)` 作为主路径。

## 4.4 `data/progress_cache/`

负责 progress planner warmup cache。

### common 文件负责

* progress warmup cache manifest schema。
* progress warmup dataset。
* progress warmup collate。
* target intent encode。
* common window construction。

### `progress_cache/libero.py`

负责 LIBERO progress warmup cache：

* LIBERO demo step 采样。
* LIBERO `planner_vl_summary` 生成。
* LIBERO executed action segment。
* LIBERO target intent。

### `progress_cache/rmbench.py`

负责 RMBench progress warmup cache：

* RMBench demo step 采样。
* RMBench `planner_vl_summary` 生成。
* RMBench executed action segment。
* RMBench target intent。

---

## 5. `src/himem_bridge_vla/training/`

职责：训练流程。需要区分 common training loop 和 benchmark-specific dataset/config。

```text
src/himem_bridge_vla/training/
├── __init__.py
│
├── common/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── distributed.py
│   ├── logging.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── seed.py
│
├── progress_warmup/
│   ├── __init__.py
│   ├── common/
│   │   ├── config.py
│   │   ├── loop.py
│   │   ├── loss.py
│   │   ├── validators.py
│   │   └── cli_base.py
│   │
│   ├── libero/
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── dataset.py
│   │   └── defaults.py
│   │
│   └── rmbench/
│       ├── cli.py
│       ├── config.py
│       ├── dataset.py
│       └── defaults.py
│
└── stage1/
    ├── __init__.py
    ├── common/
    │   ├── config.py
    │   ├── loop.py
    │   ├── loss.py
    │   ├── batch_contract.py
    │   ├── validators.py
    │   └── cli_base.py
    │
    ├── libero/
    │   ├── cli.py
    │   ├── config.py
    │   ├── dataset.py
    │   ├── contract.py
    │   └── defaults.py
    │
    └── rmbench/
        ├── cli.py
        ├── config.py
        ├── dataset.py
        ├── contract.py
        └── defaults.py
```

---

## 5.1 `training/common/`

负责所有训练共享工具：

* checkpoint 保存/加载。
* single-card accelerator glue。
* optimizer param groups。
* scheduler。
* logging。
* seed / reproducibility。

---

## 5.2 `training/progress_warmup/common/`

负责 progress warmup 的通用训练逻辑：

* progress planner warmup loop。
* progress warmup loss。
* batch validation。
* optimizer/scheduler 调用。
* checkpoint 管理。
* 与 benchmark 无关的训练过程。

不负责：

* LIBERO cache 怎么读。
* RMBench cache 怎么读。
* benchmark-specific defaults。

---

## 5.3 `training/progress_warmup/libero/`

负责 LIBERO progress warmup 训练入口和配置适配：

* LIBERO progress warmup config 默认值。
* LIBERO progress warmup dataset 路径解析。
* LIBERO-specific sanity check。
* 调用 common loop。

---

## 5.4 `training/progress_warmup/rmbench/`

负责 RMBench progress warmup 训练入口和配置适配：

* RMBench progress warmup config 默认值。
* RMBench progress warmup dataset 路径解析。
* RMBench-specific sanity check。
* 调用 common loop。

---

## 5.5 `training/stage1/common/`

负责 Stage1 通用训练逻辑：

* Stage1 trajectory-window training loop。
* Stage1 flow-matching loss。
* Stage1 batch contract validation。
* frozen progress planner state chronological update。
* checkpoint 保存。
* optimizer/scheduler。
* common CLI base。

这里不直接写 LIBERO/RMbench 数据路径和 view/action 维度。

---

## 5.6 `training/stage1/libero/`

负责 LIBERO Stage1 训练：

* LIBERO Stage1 config schema。
* LIBERO Stage1 默认 contract：

  * action_dim = 7
  * state_dim = 8
  * views = 2
  * short offsets = [16, 8]
  * replan_stride = 16
  * horizon = 32
* LIBERO Stage1 token-cache dataset 适配。
* LIBERO-specific batch sanity check。
* 调用 Stage1 common loop。

---

## 5.7 `training/stage1/rmbench/`

负责 RMBench Stage1 训练：

* RMBench Stage1 config schema。
* RMBench Stage1 默认 contract：

  * action_dim 按 RMBench 设定。
  * state_dim 按 RMBench 设定。
  * views 按 RMBench spec。
  * short offsets 按 RMBench config。
  * horizon/replan_stride 按 RMBench config。
* RMBench token-cache dataset 适配。
* RMBench-specific batch sanity check。
* 调用 Stage1 common loop。

---

## 5.8 为什么 Stage1 要拆 LIBERO/RMBench？

Stage1 的训练 loop 可以共用，但 Stage1 的输入 contract 并不完全 benchmark-neutral：

```text
LIBERO:
  views = agentview + wrist
  state_dim = 8
  action_dim = 7
  gripper sign = raw LIBERO protocol
  short memory = [t-16, t-8]

RMBench:
  views/action/state/adapter 规则不同
  action_dim 可能不是 7
  norm_stats 和 official eval adapter 不同
```

所以正确设计是：

```text
stage1/common = 训练机制
stage1/libero = LIBERO Stage1 数据与 contract
stage1/rmbench = RMBench Stage1 数据与 contract
```

而不是一个 `train_stage1.py` 里面根据参数到处 `if benchmark == ...`。

---

## 6. `src/himem_bridge_vla/runtime/`

职责：在线推理 runtime。runtime 不直接处理 benchmark obs/action，但可以处理已经转换好的 `PolicyRequest`。

```text
src/himem_bridge_vla/runtime/
├── __init__.py
├── contract.py
├── feature_extractor.py
├── memory_builder.py
├── action_history.py
├── inference_engine.py
├── json_codec.py
├── checkpoint_loader.py
└── websocket_server.py
```

### `contract.py`

定义 runtime 输入输出：

* `PolicyRequest`
* `PolicyResponse`
* `RuntimeFeatures`
* `RuntimeState`

### `feature_extractor.py`

负责：

```text
images_by_view + prompt
-> current_visual_tokens
-> vlm_hidden_states
-> planner_vl_summary
```

这里统一 planner summary 定义。

### `memory_builder.py`

负责：

```text
short_memory_images_by_offset
-> short_memory_tokens
-> short_memory_mask
-> short_memory_time_ids
```

必须支持原设计 `[16, 8]`。

### `action_history.py`

负责：

* 校验 `executed_actions`。
* normalize `executed_actions`。
* 构建 `executed_action_mask`。
* 支持真实环境执行 action 回传。

### `inference_engine.py`

负责：

```text
PolicyRequest
-> RuntimeFeatures
-> HiMemBridgeVLA.predict_action()
-> PolicyResponse
```

### `checkpoint_loader.py`

负责：

* 加载 checkpoint。
* 加载 config。
* 加载 norm stats。
* 恢复模型。
* 设置 inference timesteps。

### `json_codec.py`

负责：

* JSON 到 `PolicyRequest`。
* `PolicyResponse` 到 JSON。
* legacy JSON 兼容。

### `websocket_server.py`

负责：

* websocket serve。
* 接收 JSON。
* 调用 `json_codec`。
* 调用 `inference_engine`。
* 返回 JSON。

不允许写 benchmark-specific obs/action 逻辑。

---

## 7. `src/himem_bridge_vla/benchmarks/`

职责：benchmark-specific 协议和 eval runner。

```text
src/himem_bridge_vla/benchmarks/
├── __init__.py
├── base.py
│
├── libero/
│   ├── __init__.py
│   ├── spec.py
│   ├── observation.py
│   ├── action.py
│   ├── history.py
│   ├── request_builder.py
│   ├── env_factory.py
│   ├── runner.py
│   └── diagnostics.py
│
└── rmbench/
    ├── __init__.py
    ├── spec.py
    ├── observation.py
    ├── action.py
    ├── request_builder.py
    ├── env_factory.py
    ├── runner.py
    ├── policy_adapter.py
    └── diagnostics.py
```

---

## 7.1 `benchmarks/base.py`

定义 benchmark adapter 接口：

```python
class BenchmarkAdapter(Protocol):
    spec: BenchmarkSpec

    def build_request(self, obs, prompt, history, *, reset_memory: bool) -> PolicyRequest:
        ...

    def decode_action(self, action_values):
        ...
```

---

## 7.2 `benchmarks/libero/`

### `spec.py`

负责 LIBERO 静态协议：

```text
name = "libero"
view_names = ("agentview_rgb", "eye_in_hand_rgb")
state_dim = 8
action_dim = 7
short_memory_offsets = (16, 8)
replan_stride = 16
```

### `observation.py`

负责：

* 从 LIBERO obs 取两个 view。
* 图像方向处理。
* quaternion 到 axis-angle。
* 拼 LIBERO state。

### `action.py`

负责：

* 模型 action chunk 到 LIBERO env action。
* gripper sign：`>=0 -> +1`，`<0 -> -1`。
* action chunk parse。

### `history.py`

负责：

* 保存每一步 obs。
* 保存每一步真实 env action。
* 取 `t-16` 图像。
* 取 `t-8` 图像。
* 取上一段 `executed_actions`。

### `request_builder.py`

负责：

```text
obs + prompt + history
-> PolicyRequest
```

必须保证：

```text
current images exactly 2 views
short memory offsets exactly [16, 8]
state_dim = 8
action_dim = 7
```

### `env_factory.py`

负责：

* 创建 LIBERO env。
* 加载 task suite。
* 设置 init state。
* seed。

### `runner.py`

负责：

* LIBERO episode loop。
* 调用 request_builder。
* 调用 runtime client。
* 执行 env.step。
* 记录 success/fail。
* 保存视频。
* 写 result。

---

## 7.3 `benchmarks/rmbench/`

### `spec.py`

负责 RMBench 静态协议：

* view names。
* state_dim。
* action_dim。
* task list。
* short memory offsets。
* replan_stride。

### `observation.py`

负责 RMBench obs 到 images/state。

### `action.py`

负责模型 action 到 RMBench action 协议。

### `request_builder.py`

负责 RMBench obs + history 到 `PolicyRequest`。

### `policy_adapter.py`

负责官方 RMBench policy adapter：

```text
HiMemBridgeVLA -> official RMBench eval_policy.py-compatible policy
```

### `runner.py`

负责 RMBench eval loop 或 official eval wrapper。

---

## 8. `src/himem_bridge_vla/evaluation/`

职责：评测结果统计和报告。注意：这里不处理 benchmark obs/action 协议。

```text
src/himem_bridge_vla/evaluation/
├── __init__.py
├── result_types.py
├── metrics.py
├── manifests.py
├── summaries.py
├── reports.py
└── gates.py
```

负责：

* `EpisodeResult`
* success rate
* average decision steps
* run manifest
* summary JSON
* metric gate
* report index

不负责：

* LIBERO obs to request。
* RMBench action decode。
* websocket request schema。
* benchmark view/mask protocol。

---

## 9. `src/himem_bridge_vla/diagnostics/`

职责：排查和 sanity check 逻辑。

```text
src/himem_bridge_vla/diagnostics/
├── __init__.py
├── visual_alignment.py
├── token_cache_vs_runtime.py
├── runtime_contract.py
├── gripper_protocol.py
├── planner_summary.py
└── checkpoint_probe.py
```

负责：

* LIBERO 图像方向 probe。
* cache vs runtime visual tokens 对比。
* planner_vl_summary 对比。
* short memory mask/shape 检查。
* gripper sign micro-test。
* checkpoint/config/norm_stats 检查。

---

## 10. `src/himem_bridge_vla/configs/`

职责：配置加载、schema、校验、merge。注意：YAML 文件放 repo root 的 `configs/`，这里放 Python 配置逻辑。

```text
src/himem_bridge_vla/configs/
├── __init__.py
├── loader.py
├── schema.py
├── resolver.py
├── model.py
├── training.py
├── eval.py
├── runtime.py
└── data.py
```

---

## 11. `src/himem_bridge_vla/utils/`

职责：低层通用工具。

```text
src/himem_bridge_vla/utils/
├── __init__.py
├── image.py
├── normalization.py
├── logging.py
├── path.py
├── reproducibility.py
└── serialization.py
```

只放无业务或弱业务工具，不放 benchmark 协议。

---

## 12. `scripts/`

职责：命令行入口。所有脚本只做：

```text
parse args
load config
call package main()
```

不写核心逻辑。

```text
scripts/
├── train/
│   ├── stage1/
│   │   ├── libero.py
│   │   └── rmbench.py
│   └── progress_warmup/
│       ├── libero.py
│       └── rmbench.py
│
├── cache/
│   ├── libero/
│   │   ├── build_replay_index.py
│   │   ├── build_stage1_token_cache.py
│   │   └── build_progress_warmup_cache.py
│   ├── rmbench/
│   │   ├── build_replay_index.py
│   │   ├── build_stage1_token_cache.py
│   │   ├── build_progress_warmup_cache.py
│   │   └── build_norm_stats.py
│   └── inspect_benchmarks.py
│
├── serve/
│   ├── policy_server.py
│   └── start_policy_server.sh
│
├── eval/
│   ├── libero/
│   │   ├── run_eval.py
│   │   ├── run_smoke.py
│   │   └── plan_run.py
│   └── rmbench/
│       ├── run_eval.py
│       ├── run_smoke.py
│       ├── plan_eval.py
│       └── install_policy_adapter.py
│
├── report/
│   ├── summarize_results.py
│   ├── write_manifest.py
│   └── build_report.py
│
├── quality/
│   ├── check_repo.sh
│   ├── preflight.py
│   ├── validate_configs.py
│   ├── validate_dataset.py
│   └── smoke_runtime.py
│
├── diagnose/
│   ├── probe_visual_alignment.py
│   ├── probe_runtime_contract.py
│   ├── probe_token_cache_vs_runtime.py
│   ├── probe_gripper_protocol.py
│   └── probe_planner_summary.py
│
├── setup/
│   ├── setup_libero_env.sh
│   ├── download_libero_checkpoint.sh
│   └── download_rmbench_tasks.py
│
├── maintenance/
│   └── export_unpushed_commits.sh
│
└── legacy/
    ├── train.py
    ├── himem_server.py
    └── old_eval/
```

---

## 13. `configs/`

职责：YAML/env 配置文件。

```text
configs/
├── model/
│   ├── direct_progress_w4.yaml
│   └── direct_progress_w4_debug.yaml
│
├── training/
│   ├── stage1/
│   │   ├── libero/
│   │   │   ├── libero_10_direct_progress_w4.yaml
│   │   │   └── smoke.yaml
│   │   └── rmbench/
│   │       ├── rmbench_direct_progress_w4.yaml
│   │       └── smoke.yaml
│   │
│   └── progress_warmup/
│       ├── libero/
│       │   └── libero_h32_r16_w4.yaml
│       └── rmbench/
│           └── rmbench_h32_r16_w4.yaml
│
├── data/
│   ├── libero/
│   │   ├── replay_index.yaml
│   │   ├── stage1_token_cache.yaml
│   │   └── progress_warmup_cache.yaml
│   └── rmbench/
│       ├── replay_index.yaml
│       ├── stage1_token_cache.yaml
│       ├── progress_warmup_cache.yaml
│       └── norm_stats.yaml
│
├── runtime/
│   ├── server.yaml
│   └── inference.yaml
│
├── eval/
│   ├── libero/
│   │   ├── smoke.yaml
│   │   └── full_10tasks.yaml
│   └── rmbench/
│       ├── smoke.yaml
│       └── full.yaml
```

---

## 14. `tests/`

职责：和源码结构一一对应的测试。

```text
tests/
├── core/
├── model/
├── data/
│   ├── readers/
│   ├── replay/
│   ├── token_cache/
│   └── progress_cache/
│
├── training/
│   ├── stage1/
│   │   ├── common/
│   │   ├── libero/
│   │   └── rmbench/
│   └── progress_warmup/
│       ├── common/
│       ├── libero/
│       └── rmbench/
│
├── runtime/
├── benchmarks/
│   ├── libero/
│   └── rmbench/
├── evaluation/
└── diagnostics/
```

测试必须跟随模块边界：

* LIBERO action 测试只在 `tests/benchmarks/libero/`。
* RMBench adapter 测试只在 `tests/benchmarks/rmbench/`。
* Stage1 common loop 测试在 `tests/training/stage1/common/`。
* LIBERO Stage1 contract 测试在 `tests/training/stage1/libero/`。
* Runtime contract 测试在 `tests/runtime/`。

---

## 15. `docs/`

职责：设计、协议、recipe、迁移说明。

```text
docs/
├── architecture/
│   ├── active_model_path.md
│   ├── runtime_contract.md
│   ├── direct_bridge_attention.md
│   └── progress_state_planner.md
│
├── benchmark_contracts/
│   ├── libero_contract.md
│   └── rmbench_contract.md
│
├── training/
│   ├── stage1/
│   │   ├── common_stage1_contract.md
│   │   ├── libero_stage1_recipe.md
│   │   └── rmbench_stage1_recipe.md
│   ├── progress_warmup/
│   │   ├── common_progress_warmup_contract.md
│   │   ├── libero_progress_warmup_recipe.md
│   │   └── rmbench_progress_warmup_recipe.md
│   └── token_cache_recipe.md
│
├── evaluation/
│   ├── libero_eval.md
│   ├── rmbench_eval.md
│   └── eval_contract_checklist.md
│
└── engineering/
    ├── project_layout.md
    ├── migration_plan.md
    ├── reproducibility.md
    └── coding_rules.md
```

---

## 16. `evaluations/`

职责：legacy only。

```text
evaluations/
└── legacy/
    ├── libero/
    └── rmbench/
```

不允许新增核心逻辑。

---

## 17. 依赖方向

必须保持单向依赖：

```text
core
  ↑
model, data, runtime, benchmarks, evaluation
  ↑
training, diagnostics
  ↑
scripts
```

更具体：

```text
model 不 import benchmarks
model 不 import runtime
training 可以 import data/model，但不 import eval runner
runtime 可以 import model/core/data normalization，但不 import benchmark env
benchmarks 可以 import core/runtime contract，但不 import training
evaluation 不 import benchmark obs/action 协议
scripts 可以 import 所有 package main，但不写核心逻辑
legacy 不被任何新代码 import
```

---

## 18. 重构迁移顺序

### Phase 0：写文档，不动代码

新增：

```text
docs/engineering/project_layout.md
docs/architecture/runtime_contract.md
docs/benchmark_contracts/libero_contract.md
docs/benchmark_contracts/rmbench_contract.md
```

### Phase 1：引入 `src/` layout

* 新建 `src/himem_bridge_vla/`。
* 迁移 package。
* 修 import。
* 保留旧脚本入口。
* 跑现有测试。

### Phase 2：拆 `scripts/`

只移动脚本到子目录，不改行为。

### Phase 3：拆 benchmark adapters

新增：

```text
src/himem_bridge_vla/benchmarks/libero/
src/himem_bridge_vla/benchmarks/rmbench/
```

迁移：

* obs parsing
* state building
* action decode
* gripper protocol
* RMBench policy adapter

### Phase 4：拆 training Stage1/progress warmup

新增：

```text
training/stage1/common
training/stage1/libero
training/stage1/rmbench
training/progress_warmup/common
training/progress_warmup/libero
training/progress_warmup/rmbench
```

把训练机制和 benchmark-specific dataset/config 分开。

### Phase 5：拆 data cache

新增：

```text
data/token_cache/common.py
data/token_cache/libero.py
data/token_cache/rmbench.py
data/progress_cache/common.py
data/progress_cache/libero.py
data/progress_cache/rmbench.py
```

### Phase 6：重建 runtime contract

新增：

```text
runtime/contract.py
runtime/feature_extractor.py
runtime/memory_builder.py
runtime/inference_engine.py
runtime/websocket_server.py
```

旧 `himem_server.py` 迁入 legacy。

### Phase 7：补完整原设计 contract

* Stage1 cache 补 `planner_vl_summary`。
* Runtime 计算同定义 `planner_vl_summary`。
* LIBERO/RMBench runtime short memory 都按各自 spec 提供 offsets。
* executed_actions 使用真实执行动作。
* cache-vs-runtime probe 入库。

### Phase 8：清理 legacy / 无效入口

* `evaluations/legacy/` 只保留必要旧脚本。
* `scripts/legacy/` 默认不保留；确有外部依赖时才临时放兼容入口，并注明迁移期限。
* 无调用方、无测试覆盖、与当前单卡 Stage1/LIBERO 训练路径冲突的脚本、配置和测试应删除或迁到明确的 fixture/legacy 角落。
* 新代码不 import legacy。

---

## 19. 工程规则

1. `scripts/` 只允许 CLI 入口，不允许核心逻辑。
2. LIBERO 逻辑只能在 `benchmarks/libero/` 或对应 `data/training/*/libero/`。
3. RMBench 逻辑只能在 `benchmarks/rmbench/` 或对应 `data/training/*/rmbench/`。
4. Runtime 不允许写 benchmark-specific obs/action 逻辑。
5. Model 不允许 import benchmarks。
6. Training common 不允许写 LIBERO/RMBench 特例。
7. Stage1 common 只负责训练机制；Stage1 LIBERO/RMBench 子模块负责各自数据和 contract。
8. Progress warmup common 只负责训练机制；LIBERO/RMBench 子模块负责各自数据和 contract。
9. Evaluation 只负责 metrics/report，不负责 obs/action 协议。
10. Diagnostics 可以读 runtime/data/benchmark，但不能被 model/training/runtime 依赖。
11. Legacy 不允许被新代码 import。
12. Refactor commit 和 behavior-fix commit 必须分开。
13. 每迁移一个模块，必须有对应测试。
14. 不允许用全局 `MAX_VIEWS` 表示所有 benchmark 的 view 数。
15. 不允许让 `fused_tokens` 同时表示 raw visual tokens 和 LM hidden sequence；必须显式命名。

---

## 20. 最终判断

彻底重构后的项目应该让每个问题都有固定归属：

```text
LIBERO view/state/action/history -> benchmarks/libero/
RMBench view/state/action/adapter -> benchmarks/rmbench/
Stage1 training mechanism -> training/stage1/common/
LIBERO Stage1 contract -> training/stage1/libero/
RMBench Stage1 contract -> training/stage1/rmbench/
Stage1 token cache common -> data/token_cache/common.py
LIBERO token cache -> data/token_cache/libero.py
RMBench token cache -> data/token_cache/rmbench.py
Runtime 推理 -> runtime/
模型结构 -> model/
评测统计 -> evaluation/
排查工具 -> diagnostics/
脚本入口 -> scripts/
```

这样之后，项目不会再因为一个 LIBERO eval patch 影响 RMBench，也不会因为一个 server mask 改动污染 Stage1 训练 contract。`planner_vl_summary`、`short memory [t-16,t-8]`、`executed_actions` 这些原设计关键输入，也能在 data、training、runtime、benchmark 四个层面严格闭合。
