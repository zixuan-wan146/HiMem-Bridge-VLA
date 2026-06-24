# Dual-FIFO Visual Memory Q&A

本文回答 `docs/dual_fifo_visual_memory_design_zh.md` 中容易混淆的问题。符号问题用 LaTeX 表达，参数和配置使用代码块。

## Q1: 为什么删掉 $\text{metadata}_i$？

正式 memory entry 定义为：

$$
e_i = \left(\{V_i^v\}_{v \in \mathcal V_i},\ \tau_i,\ \eta_i\right)
$$

不再写成：

$$
e_i = \left(\{V_i^v\}_{v \in \mathcal V_i},\ \tau_i,\ \eta_i,\ \text{metadata}_i\right)
$$

原因是 $\text{metadata}_i$ 容易把实现细节和模型语义混在一起。

模型真正需要的 entry 语义只有三个：

- 视觉证据：$\{V_i^v\}_{v \in \mathcal V_i}$。
- 时间位置：$\tau_i$。
- 记忆类型：$\eta_i$。

`episode_id`、`source`、`kfs_score` 这些信息仍然可能在实现中存在，但它们属于日志字段或 replay builder 字段，不属于 entry 的数学定义。

## Q2: $\tau_i$ 到底怎么表示？

$\tau_i$ 是 low-level executed step 的整数编号。

它不是 inference call 编号，也不是 action chunk 编号。

例如：

- 控制频率是 20 Hz。
- 当前已经执行到第 $t=96$ 个 low-level step。
- 某个 memory entry 来自第 64 个 low-level step 的图像。

那么：

$$
\tau_i = 64
$$

这个 entry 距离当前的 age 是：

$$
\Delta_i = t-\tau_i = 96-64 = 32
$$

实现中真正进入 embedding 的通常是 $\Delta_i$。例如：

```text
age = current_step - entry_tau
age = min(age, max_age_steps)
age_embedding = embedding(age)
```

然后把 age embedding 加到压缩后的 memory token 上：

$$
M_i^t = \operatorname{LN}(Z_i + E_{\text{age}}(\Delta_i) + E_{\eta_i})
$$

直观理解：$\tau_i$ 让模型知道“这段视觉证据离现在有多远”。

## Q3: type embedding 是怎么回事？

$\eta_i$ 是 memory 类型：

$$
\eta_i \in \{S,L\}
$$

其中：

- $S$ 表示 short-term memory。
- $L$ 表示 long-term memory。

实现里编码成：

```text
S -> 0
L -> 1
```

然后查表得到 type embedding：

$$
E_{\eta_i} = \operatorname{Embedding}(\eta_i)
$$

例如：

$$
e_i = \left(\{V_i^{\text{base}}, V_i^{\text{wrist}}\},\ 64,\ S\right)
$$

表示这个 entry 来自第 64 个 low-level step，是一个短期 memory entry。

再比如：

$$
e_j = \left(\{V_j^{\text{base}}, V_j^{\text{wrist}}\},\ 128,\ L\right)
$$

表示这个 entry 来自第 128 个 low-level step，是一个长期 keyframe entry。

type embedding 的作用是告诉 compressor 和后续模块：这段视觉证据是局部短期上下文，还是长期关键事件。两者的时间尺度不同，不应该只靠 age embedding 区分。

padding slot 不应该有可学习 type embedding。padding 的语义是“不存在”，不是第三种 memory。

## Q4: padding entry 和 memory mask 怎么处理？

先明确 `n_m` 的默认决策：

```text
LIBERO / 双视角单臂 benchmark: 每个 entry 压缩成 1 个 token, n_m=1
双臂 / 三摄像机 benchmark / 三摄像机机器人: 每个 entry 压缩成 2 个 tokens, n_m=2
```

理由是：LIBERO 的 base + wrist 两个视角通常可以先用一个 token 表示一个历史视觉证据，成本最低、变量最少。双臂或三摄像机场景中，scene / left / right 或左右手局部信息同时重要，一个 token 容易过度压缩，所以默认给每个 entry 两个 tokens。

默认配置：

```text
N_S = 2
N_L = 4
n_m = 1
```

总 memory token 数是：

$$
T_M = (N_S+N_L)n_m = 6
$$

假设当前只有 1 个真实 short entry，long memory 为空，那么 mask 是：

$$
m_t^M = [1,0,0,0,0,0]
$$

如果 $n_m=2$，一个真实 entry 对应两个有效 mask：

$$
m_t^M = [1,1,0,0,0,0,0,0,0,0,0,0]
$$

实现原则：

1. padding entry 的 raw tensor 可以是全零。
2. padding entry 可以不进入 compressor；如果进入，输出 token 必须置零。
3. 下游 attention 或融合模块必须看到 $m_t^M$。
4. mask 为 0 的 token 不参与 attention，不参与 loss，也不参与统计。

不要让 padding slot 成为可学习 null token。如果 padding token 可学习，模型可能学到：

$$
\text{slot 2 is padding} \Rightarrow \text{episode is early}
$$

这会把“没有 memory”变成一种额外状态信号。当前方案希望 padding 的语义就是“不存在”。

## Q5: 时间语义怎么理解？

要区分三个时间概念：

### low-level step

机器人控制器或仿真环境实际执行一个动作的最小单位。例如 20 Hz 控制频率下，每 0.05 秒执行一步。

### inference step

模型被调用一次，输入当前观测，输出一个 action chunk。

### action horizon

模型一次输出多少个未来动作。当前：

$$
H = 32
$$

所以一次 inference 会输出：

$$
[a_t,a_{t+1},\ldots,a_{t+31}]
$$

memory 里的 $t$ 和 $\tau_i$ 使用 low-level step，是为了让 memory 的时间距离和真实执行过程对齐。

短期 memory：

$$
\mathcal S_t = \{I_{t-32},I_{t-16}\}
$$

意思是：当前在 low-level step $t$ 推理时，额外给模型看 32 步前和 16 步前的历史图像。

这样当前观测和 memory 形成：

$$
I_{t-32} \rightarrow I_{t-16} \rightarrow I_t
$$

它的用途是提供局部运动上下文。例如：

- 物体是否已经被推近目标。
- wrist view 中手爪是否正在接近物体。
- 场景变化是持续变化还是突然变化。
- 当前图像单帧看不出的 phase 信息。

这本质上就是 rolling / receding-horizon 语境下的历史证据读取。

## Q6: 如果没有 $I_{t-16}$ 怎么办？

这取决于 runtime 是否保存 low-level 中间观测。

如果每个 low-level step 都有图像，严格使用：

$$
\mathcal S_t = \{I_{t-32},I_{t-16}\}
$$

如果只有每次 inference 时有图像，而中间执行过程没有图像，那么 $I_{t-16}$ 可能不存在。

此时 fallback 为：

$$
\mathcal S_t = \{\text{当前推理前最近两个历史观测}\}
$$

这不是同一个实验条件。文档要求记录这个 fallback，是为了后续比较时不会把两种 memory 语义混在一起。

## Q7: 输出 32 steps 后，是不是必须等 32 steps 执行完再推理？

不是。

模型输出 32 steps 只是说明 action horizon 是：

$$
H = 32
$$

实际执行时还需要定义 replan stride：

$$
R = \text{每次推理后实际执行多少步再重新推理}
$$

如果：

$$
R = H = 32
$$

就是完整 open-loop chunk：

```text
infer once
execute 32 low-level actions
infer again
```

这种方式调用模型少，但闭环慢。如果模型推理和相机采集不能被隐藏，就可能看起来一卡一卡。

更常见的是 receding-horizon：

$$
R < H
$$

例如：

$$
H = 32,\quad R = 8
$$

流程是：

```text
t = 0:  infer, get actions a_0 ... a_31
t = 0..7: execute first 8 actions
t = 8:  observe again, infer next chunk a_8 ... a_39
t = 8..15: execute first 8 actions from new chunk
```

也就是说，模型总是预测未来 32 步，但只执行前 $R$ 步，然后重新观察和推理。

这就是为什么 demo 看起来通常是连续闭环的，而不是 32 步一卡。

## Q8: 推理频率只和 GPU 算力有关吗？

不是。GPU 算力只决定 policy forward latency 的一部分。

实际推理频率还受这些因素影响：

```text
模型 forward latency
图像采集 latency
图像预处理 latency
机器人或仿真 step latency
网络 / IPC latency
控制器频率
replan stride R
action horizon H
是否异步推理
安全约束和动作平滑策略
```

如果控制频率是 $f_c$，replan stride 是 $R$，同步推理理想情况下策略频率大约是：

$$
f_p \approx \frac{f_c}{R}
$$

但这只是理想值。真实系统必须满足：

$$
\text{inference latency} < R \times \text{control period}
$$

否则执行完前 $R$ 步时，下一个 chunk 还没算完，就会卡住。

## Q9: 异步推理是怎么避免卡顿的？

系统可以维护一个 action buffer。

模型在时刻 $t$ 输出：

$$
[a_t,a_{t+1},\ldots,a_{t+31}]
$$

控制器开始执行 buffer 里的动作。同时，系统可以在 buffer 还没空之前提前发起下一次 inference。

如果当前 buffer 还剩 $K$ 步，单次推理延迟折算成 $L$ 个 low-level steps，则应该保证：

$$
K > L + \text{safety margin}
$$

这样新 chunk 在旧 buffer 用完前已经准备好，控制就不会卡。

## Q10: 这和 memory 的 $t-32,t-16$ 有什么关系？

memory 的 $t$ 是实际执行时间线上的 low-level step。

如果使用 receding-horizon，假设：

$$
H = 32,\quad R = 8
$$

那么 inference 发生在：

$$
t = 0,8,16,24,32,\ldots
$$

在 $t=32$ 推理时，短期 memory 可以取：

$$
\mathcal S_{32} = \{I_0,I_{16}\}
$$

在 $t=40$ 推理时：

$$
\mathcal S_{40} = \{I_8,I_{24}\}
$$

前提是系统保存了这些中间观测。如果只保存 inference 时刻图像，那么 $I_{24}$ 存在，因为 $24$ 是 inference step；但如果 $R=32$，则 $I_{16}$ 就不存在。

所以 memory 设计必须和 runtime 的观测频率、replan stride 一起定义。

## Q11: 初始阶段 memory 不足怎么办？

episode 开头没有足够历史帧。此时 padding + mask 是正常行为。

例如 $t=0$：

$$
\mathcal S_0 = \varnothing
$$

mask 全 0：

$$
m_t^M = [0,0,0,0,0,0]
$$

到 $t=16$，如果有 $I_0$，但没有 $I_{-16}$：

$$
\mathcal S_{16} = \{I_0\}
$$

当前实现建议在每个 bank 内 compact real entries，然后 trailing padding，所以 mask 可以是：

$$
m_t^M = [1,0,0,0,0,0]
$$

entry 的真实时间距离由 age embedding 表达，不靠固定 slot 表达。

## Q12: 压缩里到底需不需要 $W_v$？

第一阶段不需要。

当前默认 memory 接收已经对齐到模型 hidden dim 的 visual tokens：

$$
V_i^v \in \mathbb R^{N_v \times D}
$$

其中 $D$ 已经等于模型 hidden dim，例如 896。

这时 compression 只需要：

```text
view_embedding
memory_queries
cross_attention
age_embedding
type_embedding
layer_norm
```

不需要再写：

$$
W_v V_i^v
$$

原因是当前 observation path 已经负责视觉塔输出到模型 hidden dim 的对齐。memory 如果再做一套 view-specific projection，参数会变多，而且容易和当前观测路径重复。

只有当我们保存的是视觉塔原始输出，并且：

$$
D_v \ne D
$$

才需要统一输入投影：

```text
memory_input_proj = Linear(D_v, D)
```

这个投影应该是 memory 级别的统一 projection，不建议一开始做每个摄像机一个 `W_base/W_wrist/W_left/W_right`。

## Q13: KFS 现在要不要写？

先不写。

当前优先级是把 memory 推理路径写稳：

```text
current low-level step t
-> short entries: t-32, t-16
-> long FIFO entries already written from outside
-> padding + mask
-> compression
-> C_t^M + m_t^M
```

KFS 会引入额外问题：

```text
keyframe label 是否可靠
selector score 是否稳定
写入时机是否和 action chunk 对齐
selector 错误是否污染 long memory
```

如果现在直接写 KFS，会把 memory 本体问题和 selector 问题混在一起。更稳的是先用 external/oracle long writes 验证 long FIFO，然后再训练 KFS。

## Q14: 训练是不是直接端到端？

memory compressor 最终确实靠 action loss 学习，因为没有“正确压缩 token”的单独标签。但不建议一开始全量端到端训练整个模型。

推荐：

```text
freeze visual tower
reuse current observation path aligned visual tokens
train memory compressor
train only necessary small adapter / connection layers
supervise by action prediction / flow matching loss
```

训练样本在时刻 $t$ 只能用已经可见的信息：

$$
I_t,\ I_{t-16},\ I_{t-32},\ \{e_\tau^L \mid \tau < t\}
$$

不能偷看未来：

$$
I_{t+8},\ I_{t+16}
$$

也不能使用未来才知道的 keyframe 写入。

所以当前先写推理路径是合理的。训练时直接复用同一套 memory 构造逻辑，避免 train/inference 不一致。

## Q15: 现在应该先实现哪一块？

如果暂时不管 Bridge-Attention 接入，memory 侧优先实现：

1. `VisualMemoryEntry`。
2. deterministic short FIFO。
3. external/oracle long FIFO。
4. padding + memory mask。
5. `VisualMemoryCompressor`，输出 $C_t^M$ 和 $m_t^M$。
6. 单测覆盖 FIFO 顺序、mask、shape、padding 不泄漏。

完成这些后，再考虑训练入口、Bridge 接入和 KFS。
