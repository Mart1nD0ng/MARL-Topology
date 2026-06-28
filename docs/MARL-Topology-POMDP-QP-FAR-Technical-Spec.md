# MARL-Topology 下一版技术方案：POMDP-QP-FAR

> 文档类型：技术设计文档  
> 目标：同时解决“时序组件无效”和“可行域发现困难”两个核心问题。  
> 推荐方案：**Stale / Partial CSI POMDP + QP-FAR（Quorum-Potential Feasible-Anchored Residual Learning）**  
> 中文名：**过时/部分 CSI 的法定人数势函数—可行锚定残差学习**

---

# 1. 为什么需要这一版方案

当前动态实验已经证明：

1. 在完整当前 CSI、上一拓扑可观测的设定下，memoryless actor 已经很强，recurrent 组件没有明显收益；
2. deployable local heuristic 显著强于 learned RL；
3. COMA、SCQ、chance、Pareto、PNA 等高级机制没有解决 feasibility-region learning；
4. 可行域发现是当前真正瓶颈；
5. 复杂机制大多是在“agent 已经进入可行域附近”之后才有意义。

因此下一版方案不应继续单纯堆叠 MARL 技巧，而应同时解决两个问题：

```text
A. 让时序任务真的需要历史，而不是每帧完整观测当前 CSI；
B. 让策略不再从全动作空间随机寻找可行拓扑，而是从强可部署 heuristic 附近做残差修复。
```

---

# 2. 总体方案

推荐整体架构：

\[
\boxed{
\text{Stale/Partial CSI POMDP}
+
\text{Local Hysteresis Anchor}
+
\text{Quorum-Deficit Potential}
+
\text{Residual Add/Remove/Swap Policy}
+
\text{Repair/Safety Heads}
+
\text{Local Edge Handshake}
}
\]

简称：

\[
\boxed{\text{POMDP-QP-FAR}}
\]

其中：

- **POMDP**：actor 只能看到过时或部分 CSI，奖励/评估仍用真实当前信道；
- **QP**：用 quorum shortfall 构造势函数，填平不可行平台；
- **FAR**：以 local hysteresis 作为 deployable anchor，RL 只学习残差修复和剪枝。

---

# 3. 先选哪种时序改进方案

Claude 提出了四种方案：

1. 过时 / 部分 CSI；
2. 隐藏时间相关信道状态；
3. 随机移动 / 驾驶意图；
4. 非马尔可夫约束 / 递减预算。

本方案建议：

```text
第一阶段只做方案 1：Stale / Partial CSI。
第二阶段保留方案 2 和 3 的接口，但不和方案 1 同时上线。
方案 4 暂缓，只在需要长期资源约束时再做。
```

原因：

- 过时 / 部分 CSI 最真实、最小改动、最容易控制变量；
- 它只改变 actor observation，不改变真实物理和最终 evaluator；
- 它能直接让 memory 有价值：actor 需要用历史估计当前真实信道；
- 它不会引入额外的 mobility 意图模型和 HMM 参数争议；
- 它能和 QP-FAR 最干净地结合。

---

# 4. Stale / Partial CSI POMDP

## 4.1 真状态与观测

真实环境状态包含当前真实链路状态：

\[
g_t = \{g_{e,t}\}_{e\in E},
\]

例如：

```text
true link success probability
true deadline delivery probability
true latency
true energy
true SINR
true LOS/NLOS/NLOSv
```

真实 reward、PBFT reliability、energy、latency 一律使用：

\[
g_t.
\]

actor 只能看到观测：

\[
\hat g_t.
\]

其中 \(\hat g_t\) 是过时、部分、含 age 的 CSI。

---

## 4.2 固定延迟 CSI

最简单版本：

\[
\hat g_{e,t}=g_{e,t-\delta}.
\]

其中：

\[
\delta\in\{1,2,3\}.
\]

当 \(t-\delta<0\) 时，可使用 frame 0 的探测值或初始化探测值。

actor 输入增加：

```text
csi_age = delta
```

真实 evaluator 仍使用当前：

\[
g_{e,t}.
\]

---

## 4.3 部分探测 CSI

真实 V2X 不会每帧测全图所有链路。定义探测 mask：

\[
m_{e,t}\in\{0,1\}.
\]

若 \(m_{e,t}=1\)：

\[
\hat g_{e,t}=g_{e,t-\delta}.
\]

若 \(m_{e,t}=0\)：

\[
\hat g_{e,t}=\hat g_{e,t-1},
\qquad
age_{e,t}=age_{e,t-1}+1.
\]

若 \(m_{e,t}=1\)：

\[
age_{e,t}=\delta.
\]

探测 mask 可以来自：

```text
per-node probing budget
random Bernoulli probing
local top-uncertainty probing
round-robin probing
```

第一版建议使用最可控的：

```text
per-edge independent probe probability rho
```

然后再加 per-node probing budget。

---

## 4.4 noisy CSI

可选加入观测噪声：

\[
\hat g_{e,t}
=
g_{e,t-\delta}
+
\epsilon_{e,t}.
\]

对概率量建议在 logit domain 加噪声：

\[
\operatorname{logit}(\hat p_{e,t})
=
\operatorname{logit}(p_{e,t-\delta})
+
\epsilon_{e,t},
\qquad
\epsilon\sim\mathcal N(0,\sigma^2).
\]

第一阶段可以先不加噪声，只做 delay + partial observation，避免混淆。

---

## 4.5 观测字段

actor edge features 不再直接给真实当前 CSI，而给：

```text
observed_link_success_probability
observed_deadline_delivery_probability
observed_latency
observed_energy
observed_sinr
csi_age
csi_observed_mask
last_observed_delta
```

可保留 motion features：

```text
relative_velocity_along_link
distance_delta
csi_delta_observed
```

但要明确：

```text
csi_delta_observed = observed CSI 的变化
```

而不是当前真实 CSI 的变化。

---

## 4.6 训练与评估边界

训练中允许：

- centralized critic 读取真实当前 CSI；
- evaluator 读取真实当前 CSI；
- SCQ / quorum potential 读取真实当前 CSI；
- diagnostics 读取真实当前 CSI。

部署 actor 只允许读取：

\[
\hat g_t,\ age_t,\ motion\ features,\ previous\ topology.
\]

这正是 CTDE 的合理用途：

```text
训练用真实全局物理教局部策略；
部署只用可获得观测。
```

---

# 5. 为什么 stale CSI 会让时序机制有意义

在当前完整 CSI 设定下：

\[
o_t\approx g_t.
\]

memoryless actor 已经知道当前链路好坏，历史作用很小。

加入 stale/partial CSI 后：

\[
o_t=\hat g_t\neq g_t.
\]

actor 需要利用历史：

\[
\hat g_{t-k:t},
\quad
velocity,
\quad
previous topology
\]

估计当前真实信道：

\[
p(g_t\mid \hat g_{0:t}).
\]

这变成一个滤波问题：

\[
b_t(g)=P(g_t=g\mid \hat g_{0:t}).
\]

recurrent actor 的 hidden state 可以近似：

\[
h_t\approx b_t.
\]

因此 recurrence 的作用从“锦上添花”变成“恢复隐藏当前状态”。

---

# 6. 其他时序方案是否要一起做

## 6.1 隐藏时间相关遮挡

该方案有价值，但不应和 stale CSI 第一轮同时做。

它需要新增 latent blockage state：

\[
z_{e,t}\in\{\text{clear},\text{blocked}\}.
\]

转移：

\[
P(z_{e,t+1}\mid z_{e,t}).
\]

观测：

\[
\hat y_{e,t}=f(z_{e,t},\gamma_{e,t})+\epsilon.
\]

真实 evaluator 用 \(z_{e,t}\) 决定真实信道，actor只能看到 noisy observation。

优点：

- 更真实；
- recurrence / belief state 价值更明显；
- 适合 urban NLOS 场景。

缺点：

- 改动更大；
- 需要校准转移概率；
- 会改变真实物理数据分布；
- 很难和 stale CSI 的效果解耦。

建议作为第二阶段：

```text
after stale CSI effect is measured
```

## 6.2 随机移动 / 意图

该方案也有价值，但不应第一阶段做。

当前车辆是 constant velocity，速度可观测时未来基本可预测。若加入路口随机转向：

\[
P(a_{\text{turn}}\mid lane,intersection)
\]

则未来位置有不确定性，历史可以帮助推断意图。

优点：

- 让规划和 topology hold 更有意义；
- 更接近城市交通。

缺点：

- 需要交通模型；
- 会让实验解释复杂；
- 难以判断收益来自“stale CSI”还是“意图预测”。

建议第三阶段加入。

## 6.3 非马尔可夫约束 / 递减预算

如累计切换预算、电池递减、探测预算等。

若剩余预算完全可观测，则任务仍是 MDP，不一定需要 recurrence。若剩余预算部分可观测，memory 才更有价值。

该方案适合后期加入真实长期资源约束，但不应作为第一阶段时间机制。

---

# 7. QP-FAR：解决可行域平台

## 7.1 问题不是 reward 非稠密，而是不可行区域平台

当前 reward：

\[
r=c-\tau
\]

确实是连续的。

但 PBFT reliability：

\[
c=C(x)
\]

是多阶段 quorum-tail 的全网合取式结构。当多个节点或阶段同时失败时：

\[
C(x)\approx 0
\]

且局部编辑：

\[
C(x\oplus e)-C(x)\approx0.
\]

所以问题是：

\[
\boxed{\text{dense reward but flat sub-feasible plateau}}
\]

---

# 8. Quorum shortfall feasibility distance

## 8.1 单接收者短缺

对阶段 \(h\)、接收者 \(j\)，设：

\[
S_{j,h}
=
\sum_{i\ne j}Y_{ij}^{h},
\qquad
Y_{ij}^{h}\sim Bernoulli(p_{ij}^{h}).
\]

法定人数需求是：

\[
q_h.
\]

定义 expected shortfall：

\[
\delta_{j,h}
=
\mathbb E[(q_h-S_{j,h})_+].
\]

展开为：

\[
\delta_{j,h}
=
\sum_{k=0}^{q_h-1}(q_h-k)P(S_{j,h}=k).
\]

这可以用 Poisson-binomial DP 精确计算。

直觉：

```text
一个节点平均还差几票才能达到 quorum
```

这比最终全网 \(C\) 更能区分不可行平台中的进步。

---

## 8.2 全网 quorum deficit

定义：

\[
D_{\text{quorum}}(x)
=
CVaR_\alpha
\left(
\{\delta_{j,h}(x)\}_{j,h}
\right).
\]

也可先用：

\[
D_{\max}(x)=\max_{j,h}\delta_{j,h}(x).
\]

推荐第一版同时记录：

```text
D_quorum_mean
D_quorum_max
D_quorum_cvar
worst_phase
worst_receiver
```

---

## 8.3 Proxy alignment test

在将 \(D_{\text{quorum}}\) 用于训练前，必须验证：

对局部编辑：

\[
x'=x\oplus e,
\]

计算：

\[
\Delta C=C(x')-C(x),
\]

\[
\Delta D=D_{\text{quorum}}(x)-D_{\text{quorum}}(x').
\]

报告：

```text
Spearman(ΔD, ΔC)
Top-k ΔD repairs hit true ΔC improvements?
ΔD improves but C worsens rate
ΔD improves but energy explodes rate
```

如果 \(\Delta D\) 与真实 \(C\) 不对齐，不得用于训练 reward。

---

# 9. Potential-based shaping

## 9.1 势函数

动态状态中包含上一拓扑：

\[
s_t=(g_t,\hat g_t,x_{t-1},\ldots).
\]

定义：

\[
\Phi_t(s_t)
=
-\eta_t D_{\text{quorum}}(x_{t-1},g_t).
\]

其中 \(g_t\) 是真实当前信道，训练环境可见；actor不可见。

有限时域要求：

\[
\Phi_T(s_T)=0.
\]

可以令：

\[
\eta_T=0.
\]

## 9.2 Shaped reward

\[
r'_t
=
r_t
+
\lambda_\Phi
[
\gamma\Phi_{t+1}(s_{t+1})
-
\Phi_t(s_t)
].
\]

总 shaping 回报 telescopes：

\[
\sum_{t=0}^{T-1}\gamma^t
[
\gamma\Phi_{t+1}-\Phi_t
]
=
-\Phi_0(s_0)+\gamma^T\Phi_T(s_T).
\]

若 \(\Phi_T=0\)，且初始状态固定，则最优策略不变。

## 9.3 POMDP 下的解释

即使 actor看不到 \(g_t\)，shaping仍是环境真实状态上的 reward transformation。对每条轨迹都满足 telescoping，所以不会改变底层MDP的最优轨迹排序。

但必须清楚：

```text
actor不能看到Phi或真实g_t
critic/训练环境可以用Phi减少学习难度
评估不使用Phi
```

---

# 10. Feasible-anchored residual learning

## 10.1 Anchor

选择 deployable local heuristic：

\[
H(o)=local\_hysteresis(o)
\]

它只用本地特征和上一拓扑，不调用 evaluator。

## 10.2 Residual action

策略不再生成完整拓扑，而是生成残差：

\[
x_t
=
H(\hat o_t,x_{t-1})
\oplus
\Delta_\theta(\hat o_t,h_t,H_t).
\]

其中：

```text
add edges
remove edges
swap edges
```

## 10.3 为什么 residual 更适合当前瓶颈

local hysteresis 已经接近可行流形。RL 从它附近学习：

```text
少量修复
少量剪枝
少量换边
```

比从全图 BCSP 搜索容易很多。

---

# 11. Repair / Safety heads

## 11.1 Repair head

对不可行或近可行拓扑，训练：

\[
\hat r_e^{repair}
\approx
D_{\text{quorum}}(x)
-
D_{\text{quorum}}(x\oplus e).
\]

含义：

```text
加这条边能让 quorum deficit 降多少
```

## 11.2 Safety head

对可行拓扑，训练：

\[
\hat r_e^{risk}
\approx
D_{\text{quorum}}(x\setminus e)
-
D_{\text{quorum}}(x).
\]

含义：

```text
删这条边会让可行性风险增加多少
```

## 11.3 使用方式

不可行时：

```text
优先 add / swap repair score 高的边
```

可行时：

```text
只 remove risk 低的边
```

---

# 12. Local edge handshake

## 12.1 当前问题

mutual activation：

\[
x_{ij}=1[e\in S_i]1[e\in S_j].
\]

若两端独立：

\[
P(x_{ij}=1)=P_i(e)P_j(e).
\]

关键边容易因为两端不同步而丢失。

## 12.2 两轮 handshake

### Round 1

节点 \(i\) 计算本端分数：

\[
s_{i\to j}.
\]

发送给邻居 \(j\)。

### Round 2

双方计算相同边分数：

\[
s_{ij}
=
\frac{s_{i\to j}+s_{j\to i}}{2}
+
\alpha\log \hat p_{ij}
+
\beta 1[e\in x_{t-1}].
\]

然后双方都用 \(s_{ij}\) 排序预算内 incident edges。

这仍是完全分布式，只需要邻居交换 scalar。

## 12.3 Correlated sampling

对每条边定义公共随机数：

\[
\xi_e=Hash(e,t,seed).
\]

两端对同一边使用相同随机扰动，减少独立采样错位。

---

# 13. 组合后的训练流程

## Stage A：Stale-CSI 预测体检

先不训练控制策略，训练或评估：

```text
memoryless vs recurrent
predict current true link p from stale observations
```

若 recurrent无法更好预测当前真实CSI，说明stale机制或特征有问题。

## Stage B：Anchor imitation

训练 actor 复现 local_hysteresis。

指标：

```text
proposal F1
decoded topology F1
held feasibility
switches/frame
```

若不能复现，不进入RL。

## Stage C：Quorum deficit alignment

验证：

```text
D_quorum vs true C
ΔD vs ΔC
```

若不对齐，不接shaping。

## Stage D：Add-only repair

从anchor失败场景开始，只允许加边。

目标：

\[
-D_{\text{quorum}}
\]

看是否提高可行率。

## Stage E：Conservative prune

从anchor可行场景开始，只允许删除低risk边。

目标：

```text
energy/latency down
feasibility retained
```

## Stage F：Full residual + PBRS

接入：

```text
add/remove/swap
potential shaping
trust-region anchor
handshake
```

## Stage G：再测PNA

PNA是当前唯一趋势正向机制，应在residual框架中重新测试。

---

# 14. 实验矩阵

## 14.1 CSI矩阵

```text
true current CSI
delay-1 CSI
delay-2 CSI
partial CSI rho=0.5
partial CSI rho=0.25
delay + partial
```

## 14.2 策略矩阵

```text
local_hysteresis anchor
anchor imitation actor
anchor + add repair
anchor + prune
anchor + full residual
anchor + full residual + PBRS
anchor + full residual + PBRS + PNA
```

## 14.3 时序矩阵

```text
memoryless
recurrent
memoryless + velocity
recurrent + velocity
```

---

# 15. 评价指标

## 15.1 最终指标

```text
C true closed-form PBFT reliability
per-frame feasibility
episode success
energy
latency
switches
return
```

## 15.2 可行域发现指标

```text
D_quorum
worst_receiver_shortfall
repair success rate
anchor failure repair rate
retention rate
```

## 15.3 局部-全局协调指标

```text
one-sided proposal rate
mutual acceptance rate
critical-edge disagreement
shared-score agreement
bridge-edge activation
```

## 15.4 时序指标

```text
CSI prediction error
stale-to-current link-p error
recurrent improvement over memoryless
temporal value under stale CSI
```

---

# 16. 必须避免的错误

1. 不要同时上线 stale CSI、hidden blockage、random turn 和 battery decay。
2. 不要让actor看到真实当前CSI，然后声称stale CSI实验有效。
3. 不要用proxy shaping结果做最终评估。
4. 不要在静态T=1任务中声称PBRS有效。
5. 不要把central teacher带入部署。
6. 不要只从可行域剪枝，忽略repair任务。
7. 不要用local_hysteresis作为baseline，却不把它作为anchor验证。
8. 不要在D_quorum未对齐真实C前将它加入reward。
9. 不要让residual policy破坏anchor而不报告retention。
10. 不要再从全动作空间重新开始训练复杂MARL。

---

# 17. 最终目标

本方案完成后，应能回答：

1. 过时/部分CSI是否让recurrence真正有价值？
2. \(D_{\text{quorum}}\) 是否能为不可行平台提供有用方向？
3. local_hysteresis anchor 是否能被actor稳定复现？
4. residual repair 是否能修复anchor失败场景？
5. safety/prune 是否能降低能耗/时延且不掉可行性？
6. PNA在可行锚定残差框架下是否仍有正向趋势？
7. learned policy是否能超过deployable heuristic，而不是从零落后于它？
