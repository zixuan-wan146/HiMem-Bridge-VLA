# Dual-FIFO Visual Token Memory 中文设计

状态：当前 memory 方案设计。本文只定义 memory 侧的数据结构、推理读写规则、padding/mask 规则和压缩模块。

## 1. 设计目标

memory 只做一件事：把历史视觉证据以小 token budget 的形式保留下来，并在后续推理时提供给主模型。

形式上，memory 在 low-level step $t$ 提供：

$$
(C_t^M,\ m_t^M)=\operatorname{MemoryRead}(\mathcal B_t^R,\mathcal B_t^G,t)
$$

其中：

- $\mathcal B_t^R$ 是 recent memory。
- $\mathcal B_t^G$ 是 stage memory。
- $C_t^M$ 是压缩后的 memory tokens。
- $m_t^M$ 是 memory token mask，标记哪些 token 真实有效。

状态 $s_t$ 仍然走原始 state path。memory entry 只承载视觉证据。

## 2. Dual-FIFO Memory Bank

每个 episode 维护两个 FIFO：

$$
\mathcal B_t^R = \operatorname{FIFO}_{N_R}(\text{recent visual entries})
$$

$$
\mathcal B_t^G = \operatorname{FIFO}_{N_G}(\text{stage memory entries})
$$

默认容量：

```text
N_R = 2
N_G = 4
```

含义：

- $\mathcal B_t^R$ 保存确定性 recent visual context，用来表达局部运动上下文。
- $\mathcal B_t^G$ 保存 stage visual summary，用来表达任务阶段完成或阶段切换时的视觉证据。

两个 FIFO 输出时都按时间从旧到新排列。真实 entry 不足时，在每个 bank 内使用 trailing padding 补齐：

$$
\mathcal B_t^R = [e_1^R,\ldots,e_k^R,\operatorname{PAD},\ldots,\operatorname{PAD}]
$$

$$
\mathcal B_t^G = [e_1^G,\ldots,e_j^G,\operatorname{PAD},\ldots,\operatorname{PAD}]
$$

其中 $k \le N_R$，$j \le N_G$。padding entry 的 mask 必须为 0。

## 3. Memory Entry 定义

正式 entry 定义为：

$$
e_i = \left(\{V_i^v\}_{v \in \mathcal V_i},\ \tau_i,\ \eta_i\right)
$$

其中：

- $V_i^v \in \mathbb R^{N_v \times D}$：第 $i$ 个历史时刻、第 $v$ 个视角的 visual tokens。这里默认 $D$ 已经等于模型 hidden dim。
- $\mathcal V_i$：该 entry 中可用的视角集合，例如 LIBERO 是 $\{\text{base}, \text{wrist}\}$，双臂任务可以是 $\{\text{scene}, \text{left}, \text{right}\}$。
- $\tau_i \in \mathbb N$：该 entry 对应的 low-level executed step。
- $\eta_i \in \{R,G\}$：该 entry 的 memory 类型，$R$ 表示 recent memory，$G$ 表示 stage memory。

实现中如果需要 `episode_id`、`source`、`score` 这类信息，可以作为日志或 replay builder 的 sidecar 字段保存，但它们不是 memory entry 的语义内容。

### 3.1 $\tau_i$ 怎么表示

$\tau_i$ 用整数表示 low-level step index。它不是推理次数，也不是 chunk index。

例子：当前已经执行到第 $t=96$ 个 low-level step，某个 entry 来自第 64 个 low-level step，则：

$$
\tau_i = 64
$$

它距离当前时刻的 age 是：

$$
\Delta_i = t-\tau_i = 96-64 = 32
$$

实现里使用的是 $\Delta_i$，不是绝对 $\tau_i$。$\Delta_i$ 会被截断到 `max_age_steps` 以内，然后查 age embedding。

### 3.2 $\eta_i$ 怎么表示

$\eta_i$ 是离散类型标记：

$$
\eta_i = R
$$

表示该 entry 来自 recent FIFO。

$$
\eta_i = G
$$

表示该 entry 来自 stage FIFO。

实现时使用整数编码：

```text
R -> 0
G -> 1
```

padding slot 不需要正式的 memory type。实现中可以用 `-1` 或内部常量表示 padding，但 padding token 必须被 mask 掉，不能作为真实类型进入 attention。

## 4. Padding 与 Memory Mask

memory bank 对外输出定长 entries。以默认配置为例：

```text
N_R = 2
N_G = 4
n_m = 1
```

最多输出：

$$
T_M = (N_R + N_G)n_m = 6
$$

假设当前只有一个 recent entry，没有 stage entry，则 entry slot 是：

```text
slot 0: real recent entry
slot 1: padding recent entry
slot 2: padding stage entry
slot 3: padding stage entry
slot 4: padding stage entry
slot 5: padding stage entry
```

对应 token mask 是：

$$
m_t^M = [1,0,0,0,0,0]
$$

如果每个 entry 压缩成 $n_m=2$ 个 tokens，那么一个真实 entry 对应两个有效 token：

$$
m_t^M = [1,1,0,0,0,0,0,0,0,0,0,0]
$$

padding 的处理原则：

1. padding entry 可以用全零 tensor 表示。
2. padding entry 可以不进入 compressor；如果进入，输出 token 也必须置零。
3. 下游 attention 或融合模块必须使用 $m_t^M$ 屏蔽 padding token。
4. mask 为 0 的 token 不参与 attention，不参与 loss，也不参与统计。

不要把 padding slot 设计成可学习 null token。否则模型可能利用“缺失 memory 的位置”作为捷径，导致 episode 开头或视角缺失场景下行为不稳定。padding 的语义应该是“不存在”，不是“一种特殊记忆”。

## 5. 时间语义

要区分三个概念：

- low-level step：机器人或仿真环境实际执行动作的最小步。
- inference step：模型调用一次 forward，输出一个 action chunk。
- action horizon：一次 inference 输出的未来动作长度，当前为 $H=32$。

Memory 里的 $t$ 和 $\tau_i$ 都使用 low-level step。

当前 action horizon 是：

$$
H = 32
$$

recent memory 默认取：

$$
\mathcal S_t = \{I_{t-32}, I_{t-16}\}
$$

这表示：在当前 low-level step $t$ 做推理时，memory 读取 32 步前和 16 步前的视觉证据。

当前帧 $I_t$ 不放进 memory，因为它已经走当前观测路径。Bridge 或后续主模型实际看到的是：

$$
I_t + \{I_{t-32}, I_{t-16}\}
$$

如果系统没有每个 low-level step 的图像，只在每次 inference 时有图像，那么严格的 $I_{t-16}$ 可能不存在。这时使用 fallback：

$$
\mathcal S_t = \{\text{current inference 前最近两个历史观测}\}
$$

这个 fallback 的语义不同，需要在实验记录中明确标注。

## 6. 推理频率与执行频率

ActionHead 输出 32 steps，并不意味着必须等 32 steps 全部执行完才下一次推理。

更通用的执行方式是 receding-horizon control。设：

- $H$：模型一次输出的 action horizon，当前 $H=32$。
- $R$：replan stride，也就是执行多少个 low-level actions 后再次推理。
- $f_c$：底层控制频率，例如 20 Hz。
- $f_p$：策略推理频率，理想情况下约为 $f_c/R$。

如果 $R=32$，就是完整 open-loop chunk：

```text
infer once -> execute 32 actions -> infer again
```

这种方式调用次数少，但闭环修正慢。如果推理和相机采集不能被隐藏，就可能表现为一卡一卡。

更常见的是 $R < H$，例如：

$$
H=32,\quad R=8
$$

流程是：

```text
t = 0:  infer, get actions a_0 ... a_31
t = 0..7: execute first 8 actions
t = 8:  observe again, infer next chunk a_8 ... a_39
t = 8..15: execute first 8 actions from new chunk
```

这样模型每次都预测 32 步，但只执行前 $R$ 步，然后重新观察、重新推理。它更接近 demo 里看到的连续闭环行为。

推理频率不只由 GPU 算力决定，还由这些因素共同决定：

```text
model forward latency
camera capture latency
image preprocessing latency
robot / simulator step latency
network and IPC latency
chosen replan stride R
action horizon H
safety constraints and controller rate
whether inference is synchronous or asynchronous
```

如果使用异步推理，可以在 action buffer 还没执行完时提前发起下一次 inference。这样即使单次推理较慢，也不一定会卡住控制循环。

## 7. 视觉特征接口

memory 输入直接使用视觉塔处理后的 visual tokens。当前要求这些 tokens 已经对齐到模型 hidden dim：

$$
V_i^v \in \mathbb R^{N_v \times D}
$$

其中 $D$ 等于 policy / Bridge hidden dim，例如：

```text
D = 896
```

memory 侧只补 view embedding：

$$
K_i^v = V_i^v + E_v
$$

保留的 compression 参数是：

```text
memory_queries: [n_m, D]
cross_attention: standard Q/K/V/O projections inside MultiheadAttention
view_embedding: [num_views, D]
age_embedding: [max_age_steps + 1, D]
type_embedding: [2, D]
layer_norm
```

`view_embedding` 必须保留。否则 base/wrist 或 scene/left/right 拼接后，模型无法区分 token 来自哪个摄像机。

## 8. Recent Memory

recent memory 的作用是保留最近一小段时间的视觉上下文。它按确定性 offset 取历史观测：

$$
\mathcal B_t^R = \{I_{t-\delta_1}, I_{t-\delta_2}, \ldots\}
$$

其中：

- $\mathcal B_t^R$：当前时间步的 recent memory buffer。
- $I_{t-\delta}$：距离当前时间步 $\delta$ 步之前的观测。
- $\delta$：recent memory offset。

实际写入 memory entry 时，存的是历史观测经过视觉塔后的 tokens：

$$
e_{t-\delta}^{R}
=
\left(\{V_{t-\delta}^{v}\}_{v \in \mathcal V},\ t-\delta,\ R\right)
$$

$$
V_{t-\delta}^{v}
=
\operatorname{VisionTower}(I_{t-\delta}^{v})
$$

推荐第一版：

```text
N_R = 2
offsets = [32, 16]
```

也就是：

$$
\mathcal B_t^R = \{e_{t-32}^{R}, e_{t-16}^{R}\}
$$

## 9. Stage Memory

stage memory 保存 stage visual summary。这里的“摘要”不是语言摘要，而是一个任务阶段完成或阶段切换时刻对应的 visual-token entry：

$$
\mathcal B_t^G = \operatorname{FIFO}_{N_G}(\mathcal B_{t-1}^G \oplus e_{\tau^*})
$$

默认：

```text
N_G = 4
```

写入由 stage visual summary 分数决定：

$$
\text{write if } q_t > \theta \text{ and } t-\tau_{\text{last}} > \Delta
$$

其中：

- $q_t$：当前时间步的 stage visual summary 分数。
- $\theta$：写入阈值。
- $\tau_{\text{last}}$：上一次写入 stage memory 的时间步。
- $\Delta$：最小写入间隔，避免连续写入相似帧。
- $e_t^G$：当前帧被写入 stage memory 时形成的 stage memory entry。

当满足写入条件时：

$$
\mathcal B_t^G
=
\operatorname{FIFO}_{N_G}(\mathcal B_{t-1}^G \oplus e_t^G)
$$

其中：

$$
e_t^G
=
\left(\{V_t^v\}_{v \in \mathcal V},\ t,\ G\right)
$$

推荐第一版：

```text
N_G = 4
theta = 0.6
Delta = 32
```

## 10. View-Aware Learned Query Compression

每个 entry 的原始 visual tokens 太多，不能直接全部传给下游模块。对一个 entry，先把可用视角的 tokens 加上 view embedding 后拼接：

$$
K_i = [V_i^v + E_v]_{v \in \mathcal V_i}
$$

其中：

- $V_i^v \in \mathbb R^{N_v \times D}$。
- $E_v \in \mathbb R^D$。
- $K_i \in \mathbb R^{N_i \times D}$。

然后使用可学习 query 压缩：

$$
Q_m \in \mathbb R^{n_m \times D}
$$

$$
Z_i = \operatorname{CrossAttn}(Q_m,\ K_i,\ K_i)
$$

再加入 age 和 type：

$$
M_i^t = \operatorname{LN}(Z_i + E_{\text{age}}(\operatorname{clip}(t-\tau_i)) + E_{\eta_i})
$$

其中：

- $E_{\text{age}}$ 表示这段记忆距离当前有多远。
- $E_{\eta_i}$ 表示这是 recent memory 还是 stage memory。
- padding entry 的输出直接置零，并通过 mask 屏蔽。

推荐 compressor 参数：

```text
D = 896
num_heads = 8
dropout = 0.0
max_age_steps = 512
```

最终 memory context：

$$
C_t^M = [M_1^{R,t},\ldots,M_{N_R}^{R,t},M_1^{G,t},\ldots,M_{N_G}^{G,t}]
$$

$$
C_t^M \in \mathbb R^{T_M \times D}
$$

其中：

$$
T_M = (N_R+N_G)n_m
$$

## 11. Token Budget

`n_m` 表示每个 memory entry 压缩成几个 memory tokens。当前默认规则直接按相机/机器人形态决定：

```text
LIBERO / 双视角单臂 benchmark: n_m = 1
双臂 / 三摄像机 benchmark / 三摄像机机器人: n_m = 2
```

LIBERO 默认一个 entry 压缩成一个 token：

```text
views = [base, wrist]
N_R = 2
N_G = 4
n_m = 1
T_M = 6
```

理由：LIBERO 主要是 base + wrist 两个视角，任务尺度较短。每个 entry 用一个 token 可以先用最小成本验证 memory 是否有效。

双臂 / ALOHA / 三摄像机场景默认一个 entry 压缩成两个 tokens：

```text
views = [scene, left, right]
N_R = 2
N_G = 4
n_m = 2
T_M = 12
```

理由：双臂或三摄像机场景中，不同视角往往承载不同操作臂、工作区和局部接触信息。一个 token 可能过度压缩左右手或多视角细节，因此默认使用两个 tokens，但仍保持总 token budget 很小。

推荐消融：

```text
n_m in {1, 2, 4}
N_R in {1, 2, 3}
N_G in {0, 2, 4, 8}
```

默认不使用 `n_m=4`。只有当 `n_m=1/2` 明确出现视觉证据压缩不足时，再把 `n_m=4` 作为高成本消融。

## 12. Runtime 状态

每个 episode runtime 维护：

```text
recent visual-token history indexed by low-level step
stage memory FIFO
last_stage_write_tau
action buffer for current predicted chunk
```

第一阶段先实现：

```text
current low-level step t
-> read recent entries: t-32, t-16 if available
-> compute stage-summary score q_t
-> write current entry into stage FIFO if write condition is satisfied
-> read stage FIFO entries
-> padding + entry mask
-> view/type/age embedding
-> learned query compression
-> output C_t^M and m_t^M
```

如果没有 low-level intermediate observations，就不能声称使用了严格的 $I_{t-32}, I_{t-16}$。这时应记录为 inference-frame fallback。

## 13. Config 草案

```yaml
memory:
  enabled: true
  kind: dual_fifo_visual
  hidden_dim: 896
  views: [base, wrist]
  recent:
    capacity: 2
    offsets: [32, 16]
  stage:
    capacity: 4
    score_threshold: 0.6
    min_write_interval: 32
  compression:
    entry_tokens: 1
    num_heads: 8
    dropout: 0.0
    max_age_steps: 512
```

ALOHA / 三视角任务覆盖：

```yaml
memory:
  views: [scene, left, right]
  compression:
    entry_tokens: 2
```

## 14. 实现顺序

1. 增加 `VisualMemoryEntry` 数据结构。
2. 增加 deterministic recent memory 读取。
3. 增加 stage FIFO 在线写入和读取。
4. 增加 padding + memory mask。
5. 增加 `VisualMemoryCompressor`，先只保证输出 $C_t^M$ 和 $m_t^M$。
6. 增加单测覆盖 FIFO、mask、shape、padding 不泄漏。

## 15. 主要风险

- 每个样本复制历史 visual tokens 会导致 cache 膨胀，应使用共享 cache + timestep indices。
- padding 没有 mask 会污染 memory context。
- 没有中间 low-level frames 时，recent memory 语义会改变。
- 如果训练时没有 memory replay，推理时加入 memory 很可能无效。
