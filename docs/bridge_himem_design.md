# HiMem-Bridge-VLA 设计与配置说明

这份代码的目标不是把 HiMem-Bridge-VLA 改成一个全新的 VLA，而是在原有
InternVL3 + FlowMatching action head 之间加一条可控的
BridgeAttention + HiMem 实验路径。现在所有影响实验结论的参数都应写在
`configs/bridge_himem/**/*.yaml`，代码只负责读取和执行配置。

## 配置入口

训练时传：

```bash
python scripts/train.py \
  --dataset_config_path configs/datasets/<benchmark>.yaml \
  --bridge_himem_config configs/bridge_himem/experiments/crosskv_clean.yaml
```

YAML 分层：

- `configs/bridge_himem/base.yaml`：共享默认值，包含维度、层数、memory writer、segment accumulator 等公共参数。
- `configs/bridge_himem/experiments/baseline.yaml`：baseline fused-only。
- `configs/bridge_himem/experiments/crosskv_clean.yaml`：方案 A，只覆盖 A 需要变化的字段。
- `configs/bridge_himem/experiments/mixed_latent_clean.yaml`：方案 B，只覆盖 B 需要变化的字段。
- `configs/bridge_himem/experiments/mixed_latent_skill.yaml`：方案 B + learnable skill tokens。

实验 YAML 使用 `extends` 继承 base，例如：

```yaml
extends: ../base.yaml

experiment_name: crosskv_clean

bridge:
  enabled: true
  variant: crosskv

context:
  mode: bridge_clean

memory:
  enabled: true
  placement: crosskv
```

旧的平铺字段如 `use_bridge`、`bridge_num_layers` 仍然兼容，但新实验不要手动散写这些字段。
训练入口会先把 YAML 解析成 typed config，再转换成旧字段供现有模块使用。每次训练会在
`save_dir` 下写出 `resolved_config.json`、`environment.json` 和 `reproducibility.json`，
复现时应优先看这些文件。

## fused token 是什么

这里的 `fused_tokens` 不是 VLA-Adapter 论文里的 bridge latent，也不是新增的 memory token。
它就是原 HiMem-Bridge-VLA baseline 已经在用的 InternVL3 最后一层 hidden states：

```text
RGB views + prompt
  -> InternVL3 vision features
  -> text/image token sequence
  -> InternVL3 language_model hidden states
  -> final hidden sequence = fused_tokens
  -> FlowMatchingActionHead
```

所以 fused token 的角色应该作为消融变量，而不是默认强行拼进去：

- `context.mode: fused_only`：只走 baseline fused tokens。
- `context.mode: bridge_clean`：action head 只看 bridge tokens，解释性最干净。
- `context.mode: bridge_residual`：`concat(fused_tokens, bridge_tokens)`，更像旧实现。
- `context.mode: bridge_gated_residual`：`concat(tanh(gate) * fused_tokens, bridge_tokens)`。

现在 A/B clean 配置都用 `bridge_clean`，避免把“memory 放哪儿”的比较和
“是否保留 fused residual”的比较混在一起。

## baseline、Evo 初始化和目标模型

`baseline_fused_only` 不是当前项目要主推的目标模型。它只是对照组，用来回答：

```text
不加 BridgeAttention、不加 HiMem、不加 skill token 时，原始 VLM fused tokens + FlowMatching
action head 能做到多少。
```

真正需要比较的两条主线是：

- `crosskv_clean.yaml`：memory 进入 BridgeAttention 的 condition/KV 路径。
- `mixed_latent_clean.yaml` / `mixed_latent_skill.yaml`：memory 作为 action-head context
  token 直接参与动作生成。

如果已有 Evo VLA checkpoint，并且它的 VLM、FlowMatching action head、`horizon`、
`per_action_dim`、`state_dim` 与当前配置兼容，那么它应作为 shared initialization，
优先替代从头 warm-up：

```text
Evo checkpoint
  -> baseline_fused_only finetune, only as control
  -> crosskv_clean
  -> mixed_latent_clean / mixed_latent_skill
```

这样可以保证 baseline、CrossKV 和 MixedLatent 从同一个初始化出发，实验结论更干净。
如果 Evo checkpoint 不包含 Bridge/HiMem 模块，不能直接用严格 DeepSpeed resume 加载到
`crosskv_clean` 或 `mixed_latent_clean`；需要部分加载 VLM/action-head 权重，新初始化
Bridge/HiMem/boundary/progress/skill 模块。这个 partial pretrain loader 属于下一步工程项，
不要和当前配置消融混在一起。

## 取哪些 VLM feature 层

默认取：

```yaml
vlm:
  raw_layers: [3, 7, 11, 14]
```

当前 `InternVL3Embedder` 把 language model 截到 14 层，同时
`outputs.hidden_states` 包含 embedding output，所以索引范围是 `0..14`。
因此 `[3, 7, 11, 14]` 对应浅层、中层、深层和最终层。若以后不再截层，
只需要改 YAML，不应改 bridge 代码。

BridgeAdapter 有 4 个 BridgeAttention block 时，对应关系是：

```text
block 0 <- hidden_states[3]
block 1 <- hidden_states[7]
block 2 <- hidden_states[11]
block 3 <- hidden_states[14]
```

如果 block 数超过 raw layer 数，后面的 block 会复用最后一个 raw layer。
这条规则在 `BridgeAdapter.forward()` 里实现，参数由 YAML 决定。

## 动作生成过程怎么对应

整体路径是：

```text
selected VLM hidden states + learned action query tokens + state + optional memory
  -> BridgeAdapter
  -> bridge_tokens
  -> context.mode 选择 action head 输入
  -> FlowMatchingActionHead
  -> action chunk
```

当前 action query 不是插进 InternVL3 tokenizer 的特殊 token，而是
BridgeAdapter 内部的 learnable tokens：

```yaml
action_query:
  source: learned_bridge
  num_tokens: 64
```

也就是说，现阶段它们的作用是让 bridge latent 有一组稳定的“动作查询槽位”。
真正把 action query token 注入 VLM token sequence 属于下一阶段，不要和当前实验混在一起。

## BridgeAttention 机制

每个 BridgeAttention block 的输入是：

- `action_tokens`：要生成的 bridge/action latent，形状 `[B, A, D]`。
- `raw_features`：某一层 VLM hidden states，形状 `[B, R, D_raw]`。
- `action_query_features`：learnable action query tokens，形状 `[B, Q, D]`。
- `proprio_embedding`：机器人 state 投影后的 token。
- `memory_context`：方案 A 时读出的 memory tokens。

计算结构是：

```text
self_out  = SelfAttention(action_tokens)
raw_out   = CrossAttention(action_tokens, raw_features)
query_out = CrossAttention(action_tokens, [action_query, proprio, optional_memory])
output    = FFN(action_tokens + self_out + tanh(raw_gate) * raw_out + query_out)
```

`raw_gate_init: 0.0` 会让 raw VLM 注入从 0 开始，这是 VLA-Adapter 里最值得保留的稳定性设计。

## HiMem 分层结构

现在 HiMem 不再是“把 bridge tokens 平均一下直接写进 bank”。运行时分三层：

```text
Level 0 frame tokens:
  BridgeAdapter 输出 bridge_tokens

Level 1 segment tokens:
  HiMemTokenWriter 用 learnable write queries 从 bridge_tokens 中蒸馏出固定数量 memory tokens
  HierarchicalEpisodeMemory 对同一 episode 的 pending segment 做 EMA 更新

Level 2 episode bank:
  EpisodeMemoryBank 按 episode_id 保存 segment tokens
  read 时用当前 fused/bridge query 做 cosine top-k 检索
```

一次推理的顺序是：

```text
read memory_context by episode_id
  -> generate bridge/action context
  -> action head 生成动作
  -> boundary_head 输出 boundary probability
  -> threshold 通过时写入 episode memory bank
```

相关 YAML：

```yaml
memory:
  enabled: true
  bank_max_tokens: 64
  read_top_k: 8
  write_threshold: 0.5
  writer:
    num_tokens: 4
    num_heads: 8
  segment:
    accumulator: ema
    ema_decay: 0.9
    write_policy: boundary
```

含义：

- `writer.num_tokens: 4`：每个 segment 最多写 4 个 memory tokens。
- `segment.accumulator: ema`：同一个未结束片段内，memory token 做指数滑动更新。
- `write_policy: boundary`：只有 boundary probability 超过阈值才落盘到 episode bank。
- `write_policy: always`：调试用，每步都写。

## 方案 A 和方案 B 的控制变量

两个方案共享完全相同的 memory writer、segment accumulator、episode bank 和读写阈值。
唯一核心差别是 memory token 参与动作生成的位置。

### 方案 A：CrossKV Memory

配置：

```yaml
bridge:
  variant: crosskv
memory:
  placement: crosskv
context:
  mode: bridge_clean
```

路径：

```text
memory_context
  -> BridgeAttention 的 query/proprio/memory cross-attn branch
  -> bridge_tokens
  -> action head
```

解释：memory 是 bridge 生成过程的条件。action head 不直接看 memory token，只看已经融合过 memory 的 bridge tokens。

### 方案 B：MixedLatent Memory

配置：

```yaml
bridge:
  variant: mixed_latent
memory:
  placement: mixed_latent
context:
  mode: bridge_clean
```

路径：

```text
BridgeAdapter 不读 memory
bridge_tokens + memory_context + optional skill_tokens
  -> action head context tokens
```

解释：memory token 不先进 bridge，而是和 bridge tokens 一起作为 action head 的上下文。
这个方案不是方案 A 的升级版，而是在比较“memory 应该先影响 bridge latent”还是
“memory 应该直接参与 action token 生成”。

## skill token 当前定义

`configs/bridge_himem/experiments/mixed_latent_skill.yaml` 中的 skill token 是 learnable tokens：

```yaml
skill:
  enabled: true
  num_tokens: 4
```

当前实现里它们只在 `mixed_latent` 中追加到 action head context。它们不是从数据里自动聚类出来的离散技能，
也没有单独的 skill id 监督。这个消融只回答一个很窄的问题：在 mixed-latent 路径里，额外的可学习高层槽位是否有帮助。

## boundary / progress 辅助监督

BridgeAdapter 现在有两个辅助 head：

- `boundary_head`：预测当前片段是否应该写入 HiMem。
- `progress_head`：预测当前 trajectory / subtask 进度。

训练权重不写死在代码里，而是在 `configs/training/*.yaml` 中配置：

```yaml
boundary_loss_weight: 1.0
progress_loss_weight: 0.2
```

stage1 warm-up 默认设为 `0.0`，避免在 action-head 对齐前引入额外变量；stage2 默认开启，
让 `memory.segment.write_policy: boundary` 不再依赖完全未监督的 boundary head。

## 现在还没有做的事

- action query token 注入 InternVL3 token sequence。
- `skill_id` 的显式监督目标，例如 skill classifier 或离散 skill routing。
- Evo-only checkpoint 到 Bridge/HiMem 模型的 partial pretrain loader。
- trajectory batch 内的训练时 memory rollout。
- optical-flow tokenizer 或 HiMem-WAM 的完整低层/高层动作发现。
- 持久化跨进程 memory bank。

这些都不应该偷偷混进 A/B clean 实验。先把 CrossKV 和 MixedLatent 的位置变量比较干净，再决定下一步加什么。
