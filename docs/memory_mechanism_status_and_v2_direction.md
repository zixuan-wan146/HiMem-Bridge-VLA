# 记忆机制现状与 V2 方向

状态：工作设计文档。本文记录当前代码里的 memory 机制、这次讨论中提出的疑惑，以及基于近期 VLA/mLLM/WAM 工作整理出的下一版重建方向。

## 当前目标

目标不是把 memory 做成 planner 的内部状态，而是让 planner expert 和 memory expert 解耦，并把二者的输出交给 Bridge-Attention 自己融合。

推荐的职责划分是：

```text
planner expert:
  产生高层计划、subtask 或 remaining-plan tokens。

memory expert:
  维护短期执行记忆和长期 subtask 级别记忆。

transition trigger:
  根据最近 action/state 历史判断阶段性任务是否结束。
  触发长期 memory write 和 hard replan。

bridge attention:
  学习融合当前观测、planner tokens、短期 memory tokens、长期 memory tokens、
  proprioception 和 action-query tokens，并服务于 action head。
```

因此，planner 和 memory 可以共享 transition trigger 发出的事件，但不应该共享同一个内部表示，也不应该被强行合并成一个模块。

## 当前 Memory 机制

目前 memory 相关代码主要在：

```text
himem_bridge_vla/model/himem/memory.py
himem_bridge_vla/model/himem/controller.py
himem_bridge_vla/model/himem/boundary.py
himem_bridge_vla/model/himem_bridge_vla.py
himem_bridge_vla/model/bridge/adapter.py
himem_bridge_vla/transition_trigger_manager.py
scripts/himem_server.py
configs/bridge_himem/base.yaml
configs/bridge_himem/experiments/crosskv_clean.yaml
configs/bridge_himem/experiments/mixed_latent_clean.yaml
configs/bridge_himem/experiments/coarse_planner_crosskv.yaml
```

当前实现更接近一个轻量在线原型，不是已经验证充分的长期记忆系统。

### Bank 和 Writer

`EpisodeMemoryBank` 目前是进程内字典：

```text
episode_id -> token sequence
```

它支持用当前 query token 的池化表示做 cosine top-k 读取，并按 `max_tokens` 截断存储长度。它目前不保存结构化 entry，也没有 segment 起止步数、trigger score、写入来源、成功/失败状态、检索 key、metadata 或持久化存储。

`HiMemTokenWriter` 使用 learned write queries cross-attend 到 bridge tokens 上，输出固定数量的 memory tokens。默认写 4 个 token。这个 bottleneck 是有价值的，但当前 writer 没有直接监督，尚未被明确训练成“应该存什么”。

### Segment Controller

`HierarchicalEpisodeMemory` 会维护一个 pending segment，只有 gate 通过时才写入 bank。当前默认配置大致是：

```yaml
memory:
  bank_max_tokens: 64
  read_top_k: 8
  write_threshold: 0.5
  writer:
    num_tokens: 4
  segment:
    accumulator: ema
    ema_decay: 0.9
    write_policy: boundary
```

当前 segment accumulator 是对候选 memory tokens 做 EMA。这个实现简单稳定，但会强烈压缩时间结构，也没有显式表达 subtask start/end、action delta、关键帧或状态变化。

### 推理流程

`HiMemBridgeVLA.run_inference()` 推理时大致执行：

```text
1. 用 VLM 编码当前 observation / instruction。
2. 根据 session_id 和 episode_id 构造 memory_episode_id。
3. 用当前 fused/bridge tokens 查询 memory。
4. 通过 BridgeAdapter 和 action head 预测 action。
5. 根据写入 gate，可能把最近 bridge output 写入 memory。
```

如果 server 提供了 `memory_write_gate`，这个外部 gate 会覆盖内部 boundary gate。如果没有外部 gate，则回退到 bridge 侧 `boundary_logits` 的 sigmoid 作为写入 gate。

### Memory 注入位置

当前有两个 memory placement：

```text
crosskv:
  memory_context 进入 BridgeAdapter / BridgeAttention 的 condition tokens。
  action head 主要看 bridge tokens。

mixed_latent:
  BridgeAdapter 不直接使用 memory。
  action context 在 action head 前拼接 bridge tokens 和 memory_context。
```

如果目标是让 Bridge-Attention 自己学会融合 plan / memory / current observation，`crosskv` 更适合作为 v2 默认方向。

### Transition Trigger 集成

当前 transition trigger 的语义和目标方向是匹配的：

```text
soft_plan:
  只触发重新规划，不写 memory。

memory_write:
  提交已完成 segment 到 memory，并强制 hard replan。
```

关键不变量是：

```text
memory_write => hard_plan
soft_plan    does not imply memory_write
```

server 通过 `transition_frame` 更新在线 trigger session。如果存在 `transition_frame` 且 trigger 判断需要 `memory_write`，模型会收到外部 memory write gate。如果请求没有 `transition_frame`，则保留旧的 bridge-boundary memory 行为。

## 当前主要问题

这些问题主要是机制和训练问题，不只是代码细节。

1. **memory token 语义不清楚**

   目前 memory token 只是 bridge tokens 的 learned compression。它没有被明确成 subtask summary、world-state summary、action-state trace、关键帧状态、failure record 或 planner state。

2. **writer 缺少直接监督**

   当前 writer 没有被训练去重建未来相关状态、预测完成情况、提升检索质量或保留动作关键细节。如果训练时没有 memory rollout，下游 action loss 也很难给 writer 足够信号。

3. **训练过程没有强迫模型使用 memory**

   如果训练样本只包含当前帧/当前观测，action head 很容易完全忽略 memory。要让推理时的 memory 有效，训练时必须构造“过去已经写入 memory”的 replay 状态。

4. **长期 memory entry 没有结构化**

   当前缺少 `MemoryEntry`，无法记录 `episode_id`、`segment_id`、`start_step`、`end_step`、`trigger_score`、`write_source`、检索 key、mask 和 metadata。

5. **内部 gate 和外部 trigger 混在一起**

   当前 bridge boundary head 和 transition trigger 都可能承担写入 gate 的职责。v2 里应该让 transition trigger 成为长期写入的主事件源；boundary head 可以保留为辅助监督、诊断或消融项。

6. **还没有真正的短期记忆机制**

   当前 pending segment accumulator 更像“等待写入长期 memory 的缓存”，不是每一步都被 Bridge-Attention 读取的 working memory。

## 当前疑惑整理

这次讨论里的关键疑惑可以整理成五个问题。

1. **memory token 如何获取？**

   长期 memory 和短期 memory 都需要回答这个问题。当前 writer 压缩 bridge tokens，但这还没有定义“应该存什么信息”。

2. **长短期之间是否要交互？**

   我们希望同时有 subtask 级别长期记忆和较短窗口的 working memory，但二者的信息流方向还需要确定。

3. **长短期 memory token 如何进入 Bridge-Attention？**

   可选方案包括提前融合成一个 memory 表示、作为不同 token group 同时输入，或分别建立不同 attention path。

4. **读写机制具体怎么设置？**

   包括写入 gate、检索 key、top-k、容量控制、替换/合并策略，以及 planner 是否读取 memory。

5. **memory write 是否需要双重门控？**

   transition trigger 已经提供了阶段完成信号。再加一个 hard gate 可能降低误写，但也会增加漏写，并使问题定位更困难。

## 近期工作启发

下面这些论文主要作为机制启发，不代表要照搬架构。

### MemoryVLA

MemoryVLA 使用 working memory + long-term memory bank。当前观测被编码成 working memory，再从长期 bank 里检索相关历史，最后进行自适应融合后生成动作。

对我们的启发是：

```text
短期 working memory 和长期 memory bank 应分开，
但当前 working state 可以参与长期 memory 的检索。
```

Reference: https://arxiv.org/abs/2508.19236

### HELM

HELM 认为只增加 context length 不能解决长程 VLA 问题，关键还包括 episodic memory、verification、rollback 和 replanning。

对我们的启发是：

```text
memory write 应该和执行循环里的事件绑定，
例如 subgoal completion、failure、checkpoint 或 recovery，
而不是只依赖被动历史长度。
```

这支持保留 transition trigger，并让它成为 memory write / hard replan 的主要事件源。

Reference: https://arxiv.org/abs/2604.18791

### LoHo-Manip

LoHo-Manip 使用解耦的 manager/executor 架构。manager 以 receding-horizon 方式预测 progress-aware remaining plan，executor 专注局部 VLA 控制。

对我们的启发是：

```text
planner 可以和低层执行解耦；
重复、闭环的 replan 通常比一次性长计划更稳。
```

这支持 planner expert 和 memory expert 解耦。

Reference: https://arxiv.org/abs/2604.21924

### VQ-Memory

VQ-Memory 把过去 proprioceptive states 编码成紧凑离散 temporal tokens，用于非马尔可夫长程任务。

对我们的启发是：

```text
短期 memory 不应该只看视觉或语言；
proprioception、action history 和 action delta 也可以提供重要 phase cue。
```

Reference: https://arxiv.org/abs/2603.09513

### VLA-Pro

VLA-Pro 把 task-relevant procedural memories 存储为可检索、可融合的参数化专家。

对我们的启发是：

```text
memory 可以是独立 expert，按当前 multimodal context 检索后再融合，
不需要被 planner 吸收。
```

Reference: https://arxiv.org/abs/2605.29562

### WAM 方向

World Pilot、OA-WAM、AHA-WAM 等 WAM 风格工作说明，动作生成不只需要静态语义 token，也能从 dynamics、future-state prior、object slot state 或 action prior 中获益。

对我们的启发是：

```text
memory token 应该保留执行相关的状态/动作动态，
不应只是“发生了什么”的语言摘要。
```

References:

```text
World Pilot: https://arxiv.org/abs/2606.12403
OA-WAM:     https://arxiv.org/abs/2605.06481
AHA-WAM:    https://arxiv.org/abs/2606.09811
```

## 推荐的 V2 架构

