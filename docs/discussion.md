# Memory 方案讨论记录

本文记录 Dual-FIFO Visual Token Memory 相关问答和阶段性判断。这里是讨论记录，不是最终定稿；最终确定的设计会同步到 `docs/dual_fifo_visual_memory_design_zh.md`。

## 1. Visual Tokens 是否可以直接作为 Memory 输入

问题：

我们现有 memory 需要的 visual tokens，是否可以直接使用视觉塔已经处理好的 tokens？是否还需要额外的投影矩阵？

当前判断：

可以直接使用视觉塔输出后的 visual tokens 作为 memory 输入基础。

当前假设是：

$$
K_t^v \in \mathbb{R}^{N_v \times D}
$$

其中：

- $K_t^v$：时间步 $t$、视角 $v$ 的 visual tokens。
- $N_v$：该视角下的 token 数。
- $D$：policy / Bridge hidden dimension。

memory 侧不额外引入 view-specific projection：

$$
W_v K_t^v
$$

当前只补 view embedding：

$$
\tilde K_t^v = K_t^v + E_v
$$

其中：

- $E_v \in \mathbb{R}^{D}$：视角 $v$ 的 view embedding。
- $\tilde K_t^v$：加入 view identity 后的 visual tokens。

理由：

- 视觉塔输出后的 tokens 已经对齐到模型 hidden dimension。
- memory compressor 只需要知道 token 来自哪个 view。
- 多视角 tokens 拼接前加 $E_v$，可以避免不同相机视角完全混在一起。

注意：

- `image_stats` encoder 只是 IO smoke / test 用，不代表真实视觉塔特征。
- 如果以后换视觉塔，输出维度不是 $D$，再讨论是否加入统一 projection。

## 2. Recent Memory 设计

问题：

recent memory 怎么取？是否需要阶段性摘要分数？

当前判断：

recent memory 不使用阶段性摘要分数，只做确定性历史采样。

recent memory 形式：

$$
\mathcal{B}_t^R = \{I_{t-\delta_1}, I_{t-\delta_2}, \ldots\}
$$

第一版设置：

$$
N_R = 2
$$

$$
\text{offsets} = [32,16]
$$

也就是：

$$
\mathcal{B}_t^R = \{I_{t-32}, I_{t-16}\}
$$

这里 $I_{t-\delta}$ 是历史观测。实际进入 memory entry 的不是原始图像，而是历史观测经过视觉塔后的 visual tokens：

$$
K_{t-\delta}^{v}
=
\operatorname{VisionTower}(I_{t-\delta}^{v})
$$

对应 recent memory entry：

$$
e_{t-\delta}^{R}
=
\left(\{K_{t-\delta}^{v}\}_{v \in \mathcal{V}},\ t-\delta,\ R\right)
$$

因此第一版 recent memory 可以写成：

$$
\mathcal{B}_t^R
=
\{e_{t-32}^{R}, e_{t-16}^{R}\}
$$

理由：

- $t-16$ 提供较近的局部状态变化。
- $t-32$ 提供更早一点的运动参照。
- 该设置和 action horizon $H=32$ 比较匹配。
- recent memory 的职责是局部视觉上下文，不负责关键帧选择。

## 3. Stage Memory 设计

问题：

stage memory 保存什么？怎么写入？容量多少？

当前判断：

stage memory 保存阶段性视觉摘要对应的 visual tokens，而不是保存所有历史帧。

stage memory 形式：

$$
\mathcal{B}_t^G
=
\operatorname{FIFO}_{N_G}
(\mathcal{B}_{t-1}^G \oplus e_{\tau^*})
$$

其中：

- $\mathcal{B}_t^G$：当前时间步的 stage memory buffer。
- $N_G$：stage memory 容量。
- $e_{\tau^*}$：被选中的关键帧 entry。
- $\oplus$：追加写入 FIFO。

第一版容量：

$$
N_G = 4
$$

写入规则：

$$
\text{write if } q_t > \theta \text{ and } t-\tau_{\text{last}} > \Delta
$$

其中：

- $q_t$：当前时间步的 stage-summary score。
- $\theta$：写入阈值。
- $\tau_{\text{last}}$：上一次写入 stage memory 的 low-level step。
- $\Delta$：最小写入间隔。

推荐初始设置：

$$
\theta = 0.6
$$

$$
\Delta = 32
$$

如果当前帧被写入 stage memory，则：

$$
e_t^G
=
\left(\{K_t^v\}_{v \in \mathcal{V}},\ t,\ G\right)
$$

写入后：

$$
\mathcal{B}_t^G
=
\operatorname{FIFO}_{N_G}
(\mathcal{B}_{t-1}^G \oplus e_t^G)
$$

## 4. 文档写法原则

问题：

设计文档里是否需要写很多当前不用的机制？

当前判断：

不需要。

设计文档只写当前采用的机制和必要解释。没有采用的模块，不专门展开写，避免浪费篇幅，也避免后续实现时被误读。

已经同步到 design 文档的调整：

- 删除单独的“非目标”清单。
- 删除未定或不用机制的展开说明。
- 保留当前采用的 visual token input、recent memory、stage memory、compression、padding/mask、runtime 和 config。

## 5. KFS 原始三帧设计的问题

问题：

Keyframe Selector 是否应该使用原始三帧 MLP 方案？

原始 KFS 形式：

$$
p_t = \sigma(\operatorname{MLP}_{\text{kfs}}(x_t))
$$

需要明确：

$$
x_t \neq p_t
$$

其中：

- $x_t$：输入给 KFS 的特征向量。
- $p_t$：MLP 输出经过 sigmoid 后得到的关键帧概率。

原始三帧特征大致为：

$$
d_1 = g_t - g_{t-1}
$$

$$
d_2 = g_{t-1} - g_{t-2}
$$

$$
x_t =
[
g_{t-2},
g_{t-1},
g_t,
d_1,
d_2,
|d_1|,
|d_2|,
\cos(g_t,g_{t-1}),
\cos(g_{t-1},g_{t-2})
]
$$

当前判断：

第一版不建议直接采用三帧 MLP KFS。

原因：

- 特征维度较大。
- 训练和调参成本较高。
- 和 recent memory、stage memory、rolling inference 放在一起时，系统复杂度会升得很快。
- 容易把 selector 质量问题和 memory 机制本身的问题混在一起。

## 6. keyframe 定义：stage visual summary

本轮重新讨论后，先明确一个问题：

$$
\text{keyframe 到底是什么？}
$$

当前结论：

$$
\text{keyframe} =
\text{stage visual summary}
$$

这里的“摘要”不是语言摘要，也不是把历史压成文字，而是一个任务阶段完成或阶段切换时刻对应的 visual-token entry。

更具体地说：

$$
e_t \text{ is keyframe}
\iff
\text{frame } t \text{ visually supports that a task phase has completed or changed.}
$$

stage memory保存的是：

$$
e_t^G
=
\left(\{K_t^v\}_{v \in \mathcal V},\ t,\ L\right)
$$

也就是stage visual summary对应的 visual tokens。

### 6.1 为什么不采用“短暂视觉证据”作为第一版定义

另一种有说服力的定义是：短暂出现、之后需要记住的视觉证据。

这种定义关注的是：

$$
\text{当前出现的信息，未来决策还会用到，但未来观测里可能看不到。}
$$

典型例子：

- 曾经看到某个目标物体的位置。
- 曾经观察到 block 的排列顺序。
- 曾经看到某个按钮/标记/对象状态。
- 之后相机视角变化、遮挡或机器人移动后，这个证据不再明显。

这种定义从 memory 的理想目标来看很有说服力，因为它直接对应：

$$
\text{memory should preserve task-relevant evidence that may disappear later.}
$$

但它不适合作为第一版定义，原因是监督和采样管线成本太高：

- 需要知道哪些视觉证据对未来决策有用。
- 很难只靠普通 trajectory 自动生成可靠标签。
- 用视觉变化 heuristic 容易把噪声、遮挡、相机运动当成 keyframe。
- 采样管线和标注规则需要单独设计，成本明显高于当前阶段目标。

因此第一版不把 keyframe 定义成“短暂视觉证据”。

### 6.2 stage visual summary的含义

stage visual summary关注的是：

$$
\text{subgoal / phase / transition 已经完成。}
$$

典型例子：

- 物体已经被 grasp。
- 抽屉已经打开。
- block 已经被放到目标区域。
- 按钮已经按下。
- 当前任务阶段切换到下一阶段。

它的优点：

- 可以沿用之前 `transition_trigger` 的训练思路。
- 标签更容易从状态、脚本、事件或已有 transition 标注中得到。
- 和stage memory的 FIFO 写入更容易结合。

当前选择这个定义的原因：

- 成本可控。
- 可以复用 transition-trigger 类监督。
- 先保证 stage memory 写入的是稳定任务事件，而不是普通视觉变化。

对 selector 的影响：

KFS 不应该只是视觉变化检测器。它应该尽量预测：

$$
\text{stage completion / transition evidence}
$$

而不是单纯预测：

$$
\text{large visual change}
$$

因此，前面讨论的 global-local change heuristic 可以作为 debug baseline，但不能直接等价于最终 KFS 定义。

## 7. Recent Memory Offsets 重新讨论

原始默认：

$$
\text{offsets} = [32,16]
$$

重新讨论后，问题在于：

- $32$ 个 low-level steps 在很多情况下已经跨到上一次甚至更早的 inference chunk。
- recent memory 的职责是提供当前推理附近的局部视觉上下文。
- 如果 offset 太远，它更像 stage memory，而不是 recent memory。

如果 action horizon 是：

$$
H = 32
$$

并且 replan stride 可能是：

$$
R = 8
$$

那么：

$$
t-8
$$

大约对应上一次 replan observation；

$$
t-16
$$

大约对应上两次 replan observation；

$$
t-32
$$

大约已经是四次 replan 之前。

当前更倾向把 recent memory 第一版改为：

$$
\text{offsets} = [16,8]
$$

对应：

$$
\mathcal{B}_t^R
=
\{e_{t-16}^{S}, e_{t-8}^{S}\}
$$

这个设置不是太小，原因是：

- $t-8$ 更接近当前状态，适合作为局部闭环参考。
- $t-16$ 保留一个稍早的运动参照。
- 两者仍然覆盖最近一小段历史，不会像 $t-32$ 那样偏远。
- 如果控制频率约为 $20$ Hz，那么 $8$ steps 约为 $0.4$ 秒，$16$ steps 约为 $0.8$ 秒，对 manipulation 的局部状态变化是合理范围。

备选：

$$
\text{offsets} = [24,12]
$$

这个覆盖更长，但和常见的 replan stride $R=8$ 不完全对齐。除非后续实际控制频率或采样频率表明 $[16,8]$ 太短，否则第一版更建议用：

$$
[16,8]
$$

待确认点：

- 如果系统只在 inference step 保存图像，而不是每个 low-level step 都保存图像，则 offset 必须和可用采样点对齐。
- 如果 replan stride 不是 $8$，recent offsets 应该重新按 $R$ 调整。

## 8. KFS 简化方向：只比较一个历史采样帧

当前更倾向第一版只比较：

$$
t
$$

和：

$$
t-\delta
$$

也就是比较：

$$
K_t^v
\quad \text{vs} \quad
K_{t-\delta}^v
$$

建议先讨论：

$$
\delta = 16
$$

理由：

- recent memory 已经使用 $[32,16]$。
- $\delta=16$ 与当前recent memory窗口自然对齐。
- 如果后续明确 replan stride $R$，也可以讨论是否让 $\delta$ 跟 $R$ 走。

## 9. KFS：不能只用 Global MeanPool

一个直接想法是：

$$
g_t =
\operatorname{LN}
(\operatorname{MeanPool}_{v,n}(K_t^v))
$$

然后：

$$
s_{\text{global}}
=
1-\cos(g_t,g_{t-\delta})
$$

问题：

manipulation 的关键变化经常是局部变化，例如：

- 夹爪接触物体。
- 小物体移动。
- 抽屉缝隙变化。
- 门把手角度变化。

如果一帧有 $N=256$ 个 visual tokens，真正变化的 token 只有 $k=4$ 个，MeanPool 后局部变化大约被稀释为：

$$
\frac{k}{N}
=
\frac{4}{256}
=
0.015625
$$

也就是只剩约 $1.56\%$。因此：

$$
\text{MeanPool 可以保留，但不能作为唯一依据。}
$$

## 10. KFS：Global Branch

global branch 用来捕捉整体场景变化。

对每个 view 先做全局表示：

$$
g_t^v =
\operatorname{LN}
(\operatorname{MeanPool}_{n}(K_t^v))
$$

再比较当前帧和历史采样帧：

$$
s_{\text{global}}^v
=
\frac{1-\cos(g_t^v,g_{t-\delta}^v)}{2}
$$

多视角聚合先考虑：

$$
s_{\text{global}}
=
\max_v s_{\text{global}}^v
$$

理由：

关键事件可能只在某一个 view 中明显，例如 wrist camera 或 side camera。使用 $\max_v$ 比对所有 view 平均更敏感。

## 11. KFS：Local Branch

local branch 用来捕捉局部 token 变化。

对每个 view、每个 token 计算变化：

$$
a_{t,n}^v
=
\frac{1-\cos(K_{t,n}^v,K_{t-\delta,n}^v)}{2}
$$

其中：

- $K_{t,n}^v$：当前帧、第 $v$ 个 view、第 $n$ 个 token。
- $K_{t-\delta,n}^v$：历史采样帧对应位置的 token。
- $a_{t,n}^v$：该局部 token 的变化强度。

普通 Top-K 版本：

$$
s_{\text{local}}^v
=
\operatorname{TopKMean}_{n}(a_{t,n}^v)
$$

当前更倾向的 local contrast 版本：

$$
s_{\text{local}}^v
=
\operatorname{TopKMean}_{n}(a_{t,n}^v)
-
\operatorname{Mean}_{n}(a_{t,n}^v)
$$

多视角聚合：

$$
s_{\text{local}}
=
\max_v s_{\text{local}}^v
$$

Top-K 推荐：

$$
K = \max(4,\lfloor 0.03N_v \rfloor)
$$

理由：

- 如果整张图都在动，$\operatorname{TopKMean}$ 高，$\operatorname{Mean}$ 也高，差值不会特别大。
- 如果只有少数关键区域变化，$\operatorname{TopKMean}$ 高，$\operatorname{Mean}$ 不高，差值更明显。
- 这有助于降低相机抖动、光照整体变化、机械臂大范围遮挡造成的误触发。

## 12. KFS：Heuristic Score

当前倾向第一版不用 MLP，先用可解释的 heuristic score：

$$
q_t
=
\alpha s_{\text{global}}
+
\beta s_{\text{local}}
$$

其中：

$$
\alpha + \beta = 1
$$

推荐初始权重：

$$
\alpha = 0.4
$$

$$
\beta = 0.6
$$

原因：

manipulation 中的关键事件通常更偏局部变化，因此 local branch 权重略高。

写入stage memory：

$$
\text{write if } q_t > \theta \text{ and } t-\tau_{\text{last}} > \Delta
$$

注意：

这里建议使用 $q_t$ 或 `stage_summary_score_t`，不要叫 $p_t$。因为 heuristic score 没有经过监督校准，不是严格概率。

## 13. KFS：轻量 MLP 备选

如果后续需要学习式 KFS，可以构造更轻的两帧特征：

$$
x_t =
[
g_t,
g_t-g_{t-\delta},
|g_t-g_{t-\delta}|,
s_{\text{global}},
s_{\text{local}}
]
$$

然后：

$$
p_t =
\sigma(\operatorname{MLP}_{\text{kfs}}(x_t))
$$

这个版本相比原始三帧 KFS 更轻：

- 只比较一个历史采样帧。
- 保留全局变化。
- 加入局部 Top-K 变化。
- 不构造 $g_{t-2}$、$d_1$、$d_2$ 这类三帧特征。

## 14. KFS 当前待讨论点

KFS 尚未最终定稿。

需要继续讨论：

1. recent memory offsets 是否正式改为 $[16,8]$。
2. $\delta$ 用 $16$，还是跟 replan stride $R$ 走。
3. local score 用 $\operatorname{TopKMean}$，还是用 $\operatorname{TopKMean}-\operatorname{Mean}$。
4. threshold $\theta$ 固定 sweep，还是根据 episode / task 的 score distribution 选。
5. global/local 权重是否先用 $\alpha=0.4,\ \beta=0.6$。
6. stage memory 里当前写的 $\theta=0.6$ 是否适用于 heuristic score。这个需要看 score distribution 后再定。

## 15. 当前已同步到 Design 文档的内容

已经同步到 `docs/dual_fifo_visual_memory_design_zh.md`：

- visual tokens 直接作为 memory 输入。
- memory 侧只加 view embedding。
- recent memory：$N_R=2$，$\text{offsets}=[32,16]$。
- stage memory：$N_G=4$。
- keyframe 定义为stage visual summary。
- stage memory 在线写入规则。
- padding 和 mask 规则。
- token budget：LIBERO 用 $n_m=1$，三摄像机 / 双臂用 $n_m=2$。

尚未同步成最终 design 的内容：

- recent memory offsets 是否从 $[32,16]$ 改成 $[16,8]$。
- KFS global-local heuristic 细节。
- KFS score threshold。
- KFS 的 $\delta$。
- 是否使用 $\operatorname{TopKMean}-\operatorname{Mean}$。
