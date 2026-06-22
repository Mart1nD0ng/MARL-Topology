# MARL-Topology 技术架构设计说明

> 文档类型：技术支持文档  
> 项目：`Mart1nD0ng/MARL-Topology`  
> 日期：2026-06-22  
> 目标架构：**CTDE Graph-Counterfactual PPO + SCQ 反事实监督 + 可靠性约束 + 去中心化分布式执行**

---

## 1. 文档目的

本文给出 MARL-Topology 后续重构所需的统一技术依据，覆盖：

- V2X 动态组网任务应采用单步还是长时序建模；
- 物理层、链路层、路由、PBFT 可靠性、能耗和时延的数学定义；
- 大尺度场景的可解性判定；
- CTDE 与部署阶段完全去中心化之间的边界；
- actor、critic、decoder、evaluator、searcher 等组件的职责；
- Graph-Counterfactual PPO、SCQ 局部反事实监督和可靠性约束的组合方式；
- actor 和 critic 的神经网络结构；
- 训练、验证、部署及泛化评价方案。

本文是技术设计规范，不是对旧实现的兼容承诺。若旧代码、旧实验结论或旧文档与数学正确性冲突，应以修复后的定义为准，并重新生成数据、训练模型和报告结果。

---

# 2. 核心架构决策

## 2.1 训练可以中心化，部署必须完全去中心化

项目真正需要守住的边界是：

\[
\boxed{\text{Centralized Training with Fully Decentralized Execution}}
\]

训练阶段允许：

- 中央仿真环境；
- 全局 PBFT evaluator；
- centralized graph critic；
- 联合观测历史；
- 联合动作；
- 全局可靠性、能耗和时延监督；
- 训练期反事实模拟；
- 集中式参数更新。

部署阶段仅允许：

\[
a_{i,t}\sim \pi_\theta(a_i\mid o_{i,t},m_{i,t},\omega)
\]

其中节点 \(i\) 只能使用：

- 自身局部观测 \(o_{i,t}\)；
- 自身历史状态；
- 真实可交换的邻居消息 \(m_{i,t}\)；
- 公开协议参数 \(n,f,q,\tau\)；
- 部署偏好 \(\omega\)。

部署阶段禁止：

- critic；
- 全局状态；
- 中央 edge 排序；
- 中央 solver；
- evaluator 验证候选；
- 集中式预算分配；
- 依赖所有节点 logits 的不可分布式 decoder。

因此，**中央 critic 合法，中央 decoder 不合法**。

---

## 2.2 主方法定位

最终方法建议定位为：

> 在两时间尺度 V2X Dec-POMDP 中，以参数共享的 recurrent directional PNA actor 进行有限轮邻居通信和局部链路提议；使用 local mutual-acceptance decoder 形成预算可行拓扑；训练阶段由 centralized graph-temporal vector critic 完成长时价值估计，并以 COMA 风格反事实信用和 SCQ 闭式 PBFT 反事实监督校准；可靠性采用 chance/CVaR 约束，能耗和时延通过 preference-conditioned Pareto policy 优化。

推荐名称：

> **CTDE Graph-Counterfactual PPO with SCQ Supervision**

SCQ 全称：

> **Self-Certifying Quorum-Counterfactual Multi-Agent Reinforcement Learning**  
> 自证可行性—法定人数反事实多智能体强化学习

---

# 3. 当前项目的任务性质

## 3.1 当前主干实际是单步任务

当前训练过程本质上是：

\[
s \rightarrow x \rightarrow (C,E,L) \rightarrow \text{terminal}
\]

其中：

- 输入一个静态场景；
- actor 一次性选择拓扑；
- evaluator 计算可靠性、能耗和时延；
- episode 立即结束。

这属于：

> **多智能体共享奖励的结构化 contextual bandit**

而不是完整的长时序 MDP。

当前实现中缺少：

- \(s_{t+1}\)；
- 上一拓扑的持续状态；
- 拓扑切换成本；
- 电池累计；
- 队列累计；
- PBFT view-change 状态；
- 路由和调度表持续状态；
- 长期 return；
- 跨时间 credit assignment。

---

## 3.2 V2X 高动态不自动意味着 MDP

若满足：

\[
P(s_{t+1}\mid s_t,a_t)=P(s_{t+1}\mid s_t)
\]

并且：

- 每个时刻可以零成本重建拓扑；
- 当前拓扑不影响下一时刻；
- 没有队列、电池、切换、重传历史；
- 只优化当前共识轮；

则即使场景变化很快，每一时刻仍可视为独立 contextual bandit。

当存在下列任一机制时，任务才真正成为 MDP/POMDP：

- 当前拓扑在未来若干共识轮中继续使用；
- 重配置需要控制消息、认证或握手；
- 频繁切换产生能耗与时延；
- 当前发送影响后续电量；
- 当前决策影响队列；
- 当前失败触发 view change；
- 当前拓扑影响下一时刻链路测量和调度；
- 策略应利用速度、方向和历史链路预测未来可用性。

因此，任务是否需要时序，必须由**动作是否影响未来成本**决定，而不能只由“车辆移动很快”决定。

---

## 3.3 推荐最终建模：两时间尺度有限时域 Dec-POMDP

### 微时间尺度

固定拓扑下执行：

- 无线发送；
- 重传；
- STDMA slot；
- 中继；
- PBFT pre-prepare；
- PBFT prepare；
- PBFT commit。

### 宏时间尺度

每隔固定拓扑控制周期：

\[
\Delta T_{\mathrm{topo}}
\]

重新决策一次拓扑。

每个宏步内保持拓扑运行：

\[
H_{\mathrm{PBFT}}
\]

个共识轮，或固定数量的无线微步。

### Episode

一段真实车辆轨迹：

\[
t=0,1,\ldots,T-1.
\]

---

## 3.4 Dec-POMDP 定义

定义：

\[
\mathcal M=
\langle
\mathcal I,
\mathcal S,
\{\mathcal O_i\},
\{\mathcal A_i\},
P,
\mathbf r,
c,
\gamma
\rangle.
\]

### 全局状态

\[
s_t=
\{
p_{i,t},
v_{i,t},
h_{ij,t}^{\mathrm{channel}},
q_{i,t}^{\mathrm{queue}},
b_{i,t}^{\mathrm{battery}},
x_{t-1},
z_t^{\mathrm{PBFT}},
z_t^{\mathrm{MAC}}
\}.
\]

### 节点局部观测

\[
o_{i,t}=
\{
p_{i,t},
v_{i,t},
\text{本地邻居测量},
\text{本地信道历史},
\text{节点角色},
\text{剩余预算},
\text{局部队列},
x_{t-1}^{(i)}
\}.
\]

### 节点动作

节点 \(i\) 输出对邻居 \(j\) 的有向 bid：

\[
z_{i\rightarrow j,t}.
\]

节点在预算内选择提议集合：

\[
A_{i,t}\subseteq \mathcal N_i,
\qquad
|A_{i,t}|\leq b_i.
\]

### 去中心化 decoder

无向边激活条件：

\[
x_{ij,t}
=
\mathbf 1
\left[
j\in A_{i,t}
\land
i\in A_{j,t}
\right].
\]

该规则可由两端独立执行，只需交换 proposal/accept bit。

---

## 3.5 拓扑稳定性应转化为物理成本

不建议添加一个没有物理解释的“稳定性奖励”。

将拓扑切换计入真实成本：

\[
E_t^{\mathrm{total}}
=
E_t^{\mathrm{PBFT}}
+
E_t^{\mathrm{relay}}
+
E_t^{\mathrm{MAC}}
+
E_t^{\mathrm{policy\_comm}}
+
E_t^{\mathrm{reconfig}}.
\]

\[
L_t^{\mathrm{total}}
=
L_t^{\mathrm{consensus}}
+
L_t^{\mathrm{reconfig}}.
\]

例如：

\[
E_t^{\mathrm{reconfig}}
=
e_{\mathrm{edge}}
|E_t\triangle E_{t-1}|,
\]

\[
L_t^{\mathrm{reconfig}}
=
l_{\mathrm{edge}}
|E_t\triangle E_{t-1}|.
\]

于是频繁改变拓扑会自然增加能耗和时延，无需额外手调稳定性权重。

---

## 3.6 必须执行 Temporal Value Test

保留单步 contextual-bandit 作为强基线。

小图上比较：

### Myopic policy/oracle

\[
x_t^{\mathrm{myopic}}
=
\arg\min_x J(s_t,x).
\]

### 有限时域 policy/oracle

\[
x_{t:t+H}^{\mathrm{horizon}}
=
\arg\min
\sum_{k=0}^{H-1}
\gamma^k
J(s_{t+k},x_{t+k}).
\]

定义时间价值：

\[
\Delta_H
=
J_{\mathrm{myopic}}
-
J_{\mathrm{horizon}}.
\]

若真实切换成本、队列和动态信道下：

\[
\Delta_H\approx 0,
\]

则最终可保留 contextual bandit。

若：

\[
\Delta_H>0,
\]

则必须使用时序 actor/critic。

---

# 4. 环境数学

# 4.1 链路有限块长模型

设：

- SINR：\(\gamma_e\)；
- 带宽：\(B_e\)；
- 发送时长：\(t_e\)；
- blocklength：
  \[
  n_e=B_et_e;
  \]
- payload：\(D\) bits。

有限块长近似错误率：

\[
\epsilon_e
=
Q\left(
\frac{
n_e\log_2(1+\gamma_e)
-D+\frac12\log_2n_e
}{
\sqrt{n_eV(\gamma_e)}
}
\right),
\]

其中：

\[
V(\gamma)
=
\left(1-(1+\gamma)^{-2}\right)(\log_2e)^2.
\]

单次发送成功率：

\[
p_e=1-\epsilon_e.
\]

若 deadline 内最多尝试：

\[
K_e
=
\left\lfloor
\frac{T_e^{\mathrm{deadline}}}
{t_e^{\mathrm{attempt}}}
\right\rfloor,
\]

则 deadline 内交付率：

\[
d_e
=
1-(1-p_e)^{K_e}.
\]

期望尝试次数：

\[
\bar K_e=
\begin{cases}
d_e/p_e,&p_e>0,\\
K_e,&p_e=0.
\end{cases}
\]

链路期望时延：

\[
L_e
=
\bar K_e t_e^{\mathrm{attempt}}.
\]

链路期望能耗：

\[
E_e
=
\bar K_e
\left[
P_{\mathrm{tx}}t_{\mathrm{tx}}
+
P_{\mathrm{rx}}t_{\mathrm{tx}}
+
P_{\mathrm{proc}}t_{\mathrm{proc}}
\right].
\]

---

# 4.2 路由与 relay

必须确保只有一层多跳语义。

推荐构建真正的一跳矩阵：

\[
M_{ij}^{(1,h)}
=
P(
i\rightarrow j
\text{ 在 PBFT 阶段 }h\text{ deadline 内直接送达}
).
\]

最多 \(H\) 跳的端到端可靠性：

\[
M_{ij}^{(\leq H,h)}
=
\max_{\substack{r:i\leadsto j\\|r|\leq H}}
\prod_{e\in r}
M_e^{(1,h)}.
\]

不能先使用 unrestricted route 获得多跳端到端概率，再将该概率作为“一跳边”执行第二次 relay DP。

必须通过：

```text
A -- B -- C

relay_hops=1: P(A->C)=0
relay_hops=2: P(A->C)>0
```

回归测试。

---

# 4.3 PBFT 参数安全性

经典 PBFT：

\[
n=3f+1,
\qquad
q=2f+1=n-f.
\]

若允许任意验证者数 \(n\)，法定人数必须满足：

\[
2q-n>f
\]

以保证两个 quorum 交集中至少存在一个诚实节点，同时满足：

\[
q\leq n-f
\]

以保证剩余诚实节点可形成 quorum。

最小安全整数可定义为：

\[
q_{\min}
=
\left\lfloor
\frac{n+f}{2}
\right\rfloor+1,
\]

同时验证 liveness。

项目必须明确支持：

- `classic_exact`：
  \[
  n=3f+1,\ q=2f+1;
  \]
- 或 `safe_generalized`：
  对任意 \(n\) 计算安全 \(q\)。

不得继续对任意 N 固定：

\[
f=1,\ q=3.
\]

---

# 4.4 Poisson-binomial quorum-tail

设：

\[
Y_i\sim\operatorname{Bernoulli}(p_i).
\]

法定人数尾概率：

\[
Q_q(\mathbf p)
=
\Pr\left(
\sum_iY_i\geq q
\right).
\]

生成多项式：

\[
G(z)
=
\prod_i
\left[
(1-p_i)+p_iz
\right].
\]

若：

\[
G(z)=\sum_ka_kz^k,
\]

则：

\[
Q_q(\mathbf p)
=
\sum_{k=q}^{m}a_k.
\]

应同时保留：

- reference float DP；
- Torch differentiable DP；
- batch 版本；
- exact parity tests；
- gradcheck。

---

# 4.5 Quorum-tail 可微性

对第 \(i\) 个输入：

\[
\frac{\partial Q_q(\mathbf p)}
{\partial p_i}
=
\Pr\left(
\sum_{j\neq i}Y_j=q-1
\right).
\]

含义：

> 当其他消息恰好差一票形成 quorum 时，第 \(i\) 个消息最关键。

该量定义 quorum sensitivity：

\[
\chi_i
=
\left|
\frac{\partial Q_q}
{\partial p_i}
\right|.
\]

它可用于：

- 反事实候选选择；
- critic 辅助监督；
- 关键链路诊断；
- 高价值数据采样。

---

# 4.6 三阶段 PBFT 可靠性

给定 primary \(p\)。

### Pre-prepare

\[
\alpha_j^{(1,p)}
=
\begin{cases}
1,&j=p,\\
M_{pj}^{\mathrm{pre}},&j\neq p.
\end{cases}
\]

### Prepare

\[
u_{ij}^{(2,p)}
=
\alpha_i^{(1,p)}
M_{ij}^{\mathrm{prepare}}.
\]

\[
\alpha_j^{(2,p)}
=
\alpha_j^{(1,p)}
Q_{q_{\mathrm{prepare}}}
\left(
\{u_{ij}^{(2,p)}\}_{i\neq j}
\right).
\]

### Commit

\[
u_{ij}^{(3,p)}
=
\alpha_i^{(2,p)}
M_{ij}^{\mathrm{commit}}.
\]

\[
\alpha_j^{(3,p)}
=
\alpha_j^{(2,p)}
Q_{q_{\mathrm{commit}}}
\left(
\{u_{ij}^{(3,p)}\}_{i\neq j}
\right).
\]

### 全局成功

\[
C_p
=
Q_{q_{\mathrm{global}}}
\left(
\{\alpha_j^{(3,p)}\}_j
\right).
\]

primary 分布为 \(\rho_p\) 时：

\[
C
=
\sum_p\rho_pC_p.
\]

---

# 4.7 固定 Byzantine 故障集合

不能在不同接收者或不同阶段分别删除概率最大的 \(f\) 个节点。

应定义：

\[
B\subseteq V,
\qquad
|B|\leq f,
\]

并在整个协议中保持同一 \(B\)。

鲁棒可靠性：

\[
C_{\mathrm{robust}}(x)
=
\min_{\substack{B\subseteq V\\|B|\leq f}}
C(x;B).
\]

\(f\) 较小时精确枚举。

训练时需要平滑可采用：

\[
\operatorname{softmin}_{\beta}
\{C(x;B)\}
=
-\frac1\beta
\log\sum_Be^{-\beta C(x;B)}.
\]

最终评估必须使用 hard min。

---

# 4.8 PBFT 消息计划

必须为三个阶段生成不同消息集合。

### Pre-prepare

\[
\mathcal M_{\mathrm{pre}}
=
\{p\rightarrow j\mid j\in V_{\mathrm{val}},j\neq p\}.
\]

### Prepare

\[
\mathcal M_{\mathrm{prepare}}
=
\{i\rightarrow j\mid
i,j\in V_{\mathrm{val}},i\neq j,
i\text{ 已 prepared-ready}
\}.
\]

### Commit

\[
\mathcal M_{\mathrm{commit}}
=
\{i\rightarrow j\mid
i,j\in V_{\mathrm{val}},i\neq j,
i\text{ 已 commit-ready}
\}.
\]

coverage-gated clients：

- 不发送 PBFT vote；
- 可以作为 relay；
- 仅计算真实转发成本。

---

# 4.9 能耗

总能耗：

\[
E_t^{\mathrm{total}}
=
E_t^{\mathrm{protocol}}
+
E_t^{\mathrm{relay}}
+
E_t^{\mathrm{retrans}}
+
E_t^{\mathrm{MAC-control}}
+
E_t^{\mathrm{policy-comm}}
+
E_t^{\mathrm{reconfig}}
+
E_t^{\mathrm{view-change}}.
\]

协议消息能耗：

\[
E_t^{\mathrm{protocol}}
=
\sum_h
\sum_{m\in\mathcal M_h}
\sum_{e\in r_m}E_e.
\]

actor K 轮邻居通信成本：

\[
E_t^{\mathrm{policy-comm}}
=
\sum_{k=1}^{K}
\sum_{(i,j)\in G_{\mathrm{comm}}}
E_{ij}^{\mathrm{ctrl}}(d_m).
\]

控制通信不能被当作免费资源。

---

# 4.10 Quorum-completion latency

当前 max-all-pairs latency 容易在 phase budget 上饱和，应改成达到 quorum 的完成时间。

设消息 \(i\rightarrow j\) 在阶段 \(h\) 于时间 \(t\) 前到达的 CDF：

\[
F_{ij}^{h}(t).
\]

接收者 \(j\) 在时间 \(t\) 前达到阶段 quorum 的概率：

\[
F_{j,h}(t)
=
Q_{q_h}
\left(
\{
\alpha_i^{h-1}
F_{ij}^{h}(t)
\}_{i\neq j}
\right).
\]

阶段在时间 \(t\) 前全局完成：

\[
F_{T_h}(t)
=
Q_{q_{\mathrm{global}}}
\left(
\{F_{j,h}(t)\}_j
\right).
\]

timeout-aware 期望阶段时延：

\[
\mathbb E[\min(T_h,B_h)]
=
\int_0^{B_h}
\left[
1-F_{T_h}(t)
\right]dt.
\]

协议总时延：

\[
L_t^{\mathrm{consensus}}
=
\sum_h
\mathbb E[\min(T_h,B_h)].
\]

至少报告：

- expected；
- P50；
- P95；
- CVaR；
- timeout rate。

失败拓扑不能获得零时延，必须支付 timeout。

---

# 5. 大尺度场景可解性

# 5.1 三状态定义

\[
z_s\in
\{
\mathrm{W},
\mathrm{I},
\mathrm{U}
\}.
\]

- W：`witness_feasible`
- I：`certified_infeasible`
- U：`unknown`

有限搜索失败只能得到 U，不能直接得到 I。

---

# 5.2 Lower Bound 与 Upper Bound

对场景 \(s\)：

\[
LB(s)
=
\max_{\text{已发现合法拓扑 }x}
C(s,x).
\]

\[
UB(s)
=
\max_{x\in\mathcal X_{\mathrm{relaxed}}}
C_{\mathrm{upper}}(s,x).
\]

判定：

- 若
  \[
  LB(s)\geq\tau,
  \]
  则 W；
- 若
  \[
  UB(s)<\tau,
  \]
  则 I；
- 否则 U。

upper bound 必须来自有证明的乐观放松，不能来自神经网络预测。

---

# 5.3 Witness memory

训练场景维护：

\[
W_s=
(x_s^\star,C_s^\star,E_s^\star,L_s^\star).
\]

若策略、局部反事实或搜索器发现更优拓扑，则更新。

一旦发现：

\[
C_s^\star\geq\tau,
\]

该场景永久具有可行 witness。

训练、validation、held/test 的 witness memory 必须隔离，held/test 发现不得反馈训练。

---

# 5.4 分布级可靠性约束

不能要求所有场景逐个满足可靠性，因为真实不可行场景会使约束不可满足。

使用：

\[
J_C(\pi)
=
\mathbb E
\left[
\frac1T
\sum_t
\mathbf 1[C_t<\tau]
\right]
\leq\delta.
\]

或：

\[
\operatorname{CVaR}_{\alpha}
\left(
\sum_t(\tau-C_t)_+
\right)
\leq d.
\]

---

# 6. 优化目标

# 6.1 能耗—时延 Pareto

可靠性作为约束。

目标向量：

\[
\mathbf J(\pi)
=
\left(
\mathbb E[E^{\mathrm{total}}],
\mathbb E[L^{\mathrm{total}}]
\right).
\]

偏好：

\[
\omega=(\omega_E,\omega_L),
\qquad
\omega_E+\omega_L=1.
\]

归一化：

\[
\tilde E
=
E/E_{\mathrm{budget}},
\]

\[
\tilde L
=
L/L_{\mathrm{SLA}}.
\]

标量效用：

\[
R_t^\omega
=
-\omega_E\tilde E_t
-\omega_L\tilde L_t.
\]

不得使用 SA teacher energy 作为归一化尺度。

---

# 6.2 Chance constraint

\[
g_C(\theta)
=
\Pr(C<\tau)-\delta.
\]

对偶更新：

\[
\lambda_{k+1}
=
\left[
\lambda_k+\eta_\lambda\hat g_C
\right]_+.
\]

residual 必须可正可负，\(\lambda\) 必须能升能降。

---

# 6.3 CVaR constraint

短缺：

\[
D=\tau-C.
\]

\[
\operatorname{CVaR}_\alpha(D)
=
\min_{\nu}
\left[
\nu
+
\frac1{1-\alpha}
\mathbb E[(D-\nu)_+]
\right].
\]

可使用 distributional/quantile critic 建模尾部。

---

# 6.4 Pareto checkpoint archive

validation 上维护：

\[
\mathcal A
=
\{
(\theta_k,\rho_k,E_k,L_k,H_k)
\}.
\]

选择顺序：

1. 可靠性风险满足；
2. 最小 violation；
3. energy-latency 非支配；
4. constrained hypervolume；
5. 稳定性破平局。

不得只按 raw feasibility 保存 checkpoint。

---

# 7. Actor 设计

# 7.1 总体结构

推荐：

> **Preference-conditioned Recurrent Directional PNA Actor**

包含：

1. local temporal encoder；
2. directional message passing；
3. PNA aggregation；
4. recurrent shared update；
5. directed bid head；
6. local budget sampler。

---

# 7.2 动作图与通信图分离

定义：

\[
G_{\mathrm{act}}=(V,E_{\mathrm{act}})
\]

为可提议链路图。

定义：

\[
G_{\mathrm{comm}}=(V,E_{\mathrm{comm}})
\]

为部署时可实际交换 actor 消息的图。

必须：

\[
E_{\mathrm{comm}}
\subseteq
E_{\mathrm{physically\ observable}}.
\]

不能因为 candidate graph 为完全图，就允许一轮 actor 通信看到全局。

---

# 7.3 时间编码

\[
h_{i,t}^{\mathrm{temp}}
=
\operatorname{GRU}
\left(
h_{i,t-1}^{\mathrm{temp}},
\phi_o(o_{i,t})
\right).
\]

只编码本地历史。

---

# 7.4 Directional message

\[
m_{i\rightarrow j}^{(k)}
=
\phi_m
\left(
h_i^{(k)},
h_j^{(k)},
e_{i\rightarrow j},
\omega,
n,f,q,\tau
\right).
\]

允许：

\[
m_{i\rightarrow j}\neq m_{j\rightarrow i}.
\]

---

# 7.5 PNA aggregation

使用：

\[
[
\operatorname{mean},
\operatorname{max},
\operatorname{min},
\operatorname{std},
\operatorname{sum}
].
\]

degree scaler：

\[
s_{\mathrm{id}}(d)=1,
\]

\[
s_{\mathrm{amp}}(d)
=
\frac{\log(d+1)}
{\mathbb E[\log(D+1)]},
\]

\[
s_{\mathrm{att}}(d)
=
\frac{\mathbb E[\log(D+1)]}
{\log(d+1)}.
\]

---

# 7.6 参数共享的 recurrent message passing

\[
h_i^{(k+1)}
=
\operatorname{GRU}_{\mathrm{msg}}
\left(
h_i^{(k)},
\operatorname{PNA}
\{m_{j\rightarrow i}^{(k)}\}
\right).
\]

跨轮共享参数。

训练时随机通信轮数：

\[
K\sim\mathcal U(K_{\min},K_{\max}).
\]

部署时可使用更大 K，但必须计入通信成本。

---

# 7.7 Directed bid head

\[
z_{i\rightarrow j}
=
\phi_{\mathrm{bid}}
(h_i,h_j,e_{i\rightarrow j},\omega).
\]

每个 endpoint 独立输出 bid。

节点使用本地 Plackett–Luce 或 Gumbel-top-\(b_i\) 选择 proposal set。

采样器必须返回：

- \(a_i\)；
- \(\log\pi_i(a_i\mid o_i)\)；
- 真实动作 entropy；
- proposal set；
- budget rejection；
- gate rejection；
- mutual rejection。

---

# 7.8 Actor 特征

### 节点

- role embedding；
- 位置与速度；
- 本地预算及剩余比例；
- 局部队列；
- 电池；
- 局部候选度；
- validator/client role；
- 上一拓扑局部边；
- \(f/n\)、\(q/n\)、\(\tau\)；
- \(\omega\)。

### 有向边

- direction-specific delivery；
- SINR；
- LOS/NLOS/NLOSv；
- 距离；
- expected attempts；
- expected energy；
- expected latency；
- local MAC conflict degree；
- 上一时刻激活状态；
- mutual rejection 历史；
- bridge/local-connectivity 特征。

禁止：

- 节点 ID 数字编码；
- 全局 edge 排名；
- 全局可靠性作为 actor 输入；
- teacher/oracle 标签；
- future outcome。

---

# 8. Centralized critic 设计

# 8.1 Critic 职责

critic 仅训练使用，负责：

- 估计长期 energy return；
- 估计长期 latency return；
- 估计可靠性 cost；
- 提供 PPO advantage；
- 提供 per-agent counterfactual credit；
- 跨时间 bootstrap；
- 学习 SCQ 反事实差分。

---

# 8.2 输入

推荐输入：

- 所有节点联合观测历史；
- action graph；
- communication graph；
- 所有节点 proposals；
- mutual decoder 后拓扑；
- 上一拓扑；
- PBFT/MAC 状态；
- \(\omega,n,f,q,\tau\)。

可加入 simulator true state 作为辅助分支，但应与 joint-observation-history critic 做消融，避免 critic 依赖 actor 永远不可见的信息。

---

# 8.3 Graph-temporal encoder

建议：

1. per-node temporal GRU；
2. action-conditioned edge encoder；
3. 4–6 层 Graph Transformer 或 PNA；
4. permutation-invariant attention pooling；
5. vector value/Q heads。

不得 flatten 固定 N 输入。

---

# 8.4 Vector heads

\[
V_E(H_t),
\qquad
V_L(H_t),
\qquad
V_C(H_t).
\]

action-conditioned：

\[
Q_E(H_t,\mathbf a_t),
\]

\[
Q_L(H_t,\mathbf a_t),
\]

\[
Q_C(H_t,\mathbf a_t).
\]

标量化放在 critic 外：

\[
Q^{\omega,\lambda}
=
-\omega_EQ_E
-\omega_LQ_L
-\lambda Q_C.
\]

这样 critic 可被不同偏好复用。

---

# 8.5 Distributional cost head

若优化 CVaR，可靠性 cost critic 可输出 quantiles：

\[
Z_C^{(\tau_1)},\ldots,Z_C^{(\tau_M)}.
\]

由 quantile return 估计尾部风险。

该功能应在基础 vector critic 稳定后再启用。

---

# 9. Graph-Counterfactual PPO

# 9.1 MAPPO 基础

节点 PPO ratio：

\[
\rho_{i,t}
=
\frac{
\pi_\theta(a_{i,t}\mid o_{i,t})
}{
\pi_{\theta_{\mathrm{old}}}(a_{i,t}\mid o_{i,t})
}.
\]

若仅使用 centralized value：

\[
A_t
=
\hat R_t-V_\phi(H_t),
\]

所有节点共享 \(A_t\)。

这是 Graph-MAPPO baseline，但信用仍较粗。

---

# 9.2 COMA 风格 counterfactual advantage

对节点 \(i\)：

\[
A_{i,t}^{E}
=
Q_E(H_t,\mathbf a_t)
-
\mathbb E_{\tilde a_i\sim\pi_i}
Q_E(H_t,\tilde a_i,\mathbf a_{-i}).
\]

时延与可靠性同理。

组合：

\[
A_{i,t}^{\omega,\lambda}
=
-\omega_EA_{i,t}^{E}
-\omega_LA_{i,t}^{L}
-\lambda A_{i,t}^{C}.
\]

---

# 9.3 组合动作近似

单节点动作空间：

\[
|\mathcal A_i|
=
\sum_{k=0}^{b_i}
\binom{|\mathcal N_i|}{k}.
\]

无法完整枚举。

使用独立 actor samples：

\[
\tilde a_i^{(1)},\ldots,\tilde a_i^{(K_{\mathrm{cf}})}
\sim\pi_i(\cdot\mid o_i).
\]

baseline：

\[
b_i
\approx
\frac1{K_{\mathrm{cf}}}
\sum_{k=1}^{K_{\mathrm{cf}}}
Q(H_t,\tilde a_i^{(k)},\mathbf a_{-i}).
\]

这些反事实样本必须独立于实际动作，确保 baseline 不依赖当前 \(a_i\)。

---

# 9.4 PPO actor loss

\[
\mathcal L_{\mathrm{actor}}
=
-\mathbb E
\left[
\frac1N
\sum_i
\min
\left(
\rho_{i,t}A_{i,t},
\operatorname{clip}
(\rho_{i,t},1-\epsilon,1+\epsilon)
A_{i,t}
\right)
\right]
-
\beta_H\mathcal H.
\]

entropy 必须对应真实 local subset policy，不能用 Bernoulli entropy 代替 Plackett–Luce 或 Gaussian policy entropy。

---

# 9.5 Critic loss

vector value loss：

\[
\mathcal L_V
=
\sum_{k\in\{E,L,C\}}
\left(
V_k(H_t)-\hat R_{k,t}
\right)^2.
\]

action-Q loss：

\[
\mathcal L_Q
=
\sum_k
\left(
Q_k(H_t,\mathbf a_t)-\hat Y_{k,t}
\right)^2.
\]

可加入 clipped value loss、Huber loss、target normalization。

---

# 10. SCQ 反事实监督

# 10.1 SCQ 不是另一个 decoder

SCQ 只在训练时：

- 选择高价值局部反事实；
- 调用真实 simulator/evaluator；
- 获得 exact 局部差分；
- 校准 centralized Q critic；
- 更新 witness memory。

部署不使用 SCQ evaluator。

---

# 10.2 单步 exact difference

对节点 \(i\)：

\[
\Delta R_i^{\mathrm{env}}
=
R(\mathbf a)
-
R(\tilde a_i,\mathbf a_{-i}).
\]

critic consistency：

\[
\mathcal L_{\mathrm{SCQ}}
=
\left[
Q_\phi(H,\mathbf a)
-
Q_\phi(H,\tilde a_i,\mathbf a_{-i})
-
\Delta R_i^{\mathrm{env}}
\right]^2.
\]

---

# 10.3 时序 one-step counterfactual target

从同一 simulator state fork 两条转移：

\[
(s_t,\mathbf a_t)
\rightarrow
(r_t,s_{t+1}),
\]

\[
(s_t,\tilde a_i,\mathbf a_{-i})
\rightarrow
(\tilde r_t,\tilde s_{t+1}).
\]

差分 target：

\[
\Delta y_i
=
\left[
r_t+\gamma V(H_{t+1})
\right]
-
\left[
\tilde r_t+\gamma V(\tilde H_{t+1})
\right].
\]

\[
\mathcal L_{\mathrm{SCQ}}
=
(\Delta Q_\phi-\Delta y_i)^2.
\]

---

# 10.4 Sensitivity-guided top-M

候选评分：

\[
s_e^{\mathrm{cf}}
=
\alpha|\chi_e|
+
\beta\,\operatorname{boundary}(z_e)
+
\gamma\,\operatorname{mutualConflict}(e)
+
\eta\,\operatorname{bridgeScore}(e)
-
\zeta\,\operatorname{cost}(e).
\]

选择：

- add；
- remove；
- swap；
- mutual repair；
- bridge candidate。

top-M targeted counterfactual 主要用于：

- critic 监督；
- witness discovery；
- failure diagnosis。

若用于 policy baseline，必须确保候选构造不依赖实际采样动作，或使用正确重要性修正。

---

# 11. RLOO 的正确定位

RLOO 仅作为 critic-free baseline。

对同一场景采样 \(M\) 个联合动作：

\[
b^{(k)}
=
\frac1{M-1}
\sum_{\ell\neq k}R^{(\ell)}.
\]

\[
A^{(k)}
=
R^{(k)}-b^{(k)}.
\]

机制激活要求：

\[
M\geq2.
\]

配置声明 `baseline_type=rloo` 且 \(M=1\) 时必须直接报错。

研究配置下 M 应通过相同 evaluator-call budget 的方差/效率实验确定，不能随意使用 smoke 参数。

---

# 12. Evaluator、Critic 与 Searcher

## Evaluator / Environment

给定状态和动作，产生：

\[
(s_{t+1},C_t,E_t,L_t).
\]

它是 ground truth simulator，不要求可微，不给最优动作。

## Critic

学习：

\[
V_\phi(H_t),
\qquad
Q_\phi(H_t,\mathbf a_t).
\]

它是近似器，不是 ground truth。

## Searcher

多次调用 evaluator 以寻找高质量拓扑。

搜索器：

- 不是 RL 必需组件；
- 不能作为大尺度可解性真值；
- 可以用于 witness discovery；
- 可以作为训练诊断或 baseline；
- 不能进入部署主方法。

---

# 13. 训练数据流

```text
动态 V2X 环境
    │
    ▼
各节点局部观测与本地历史
    │
    ▼
共享 Recurrent Directional PNA Actor
    │
    ▼
每节点 local subset action + logp_i
    │
    ▼
Local Mutual-Acceptance Decoder
    │
    ▼
拓扑保持 H 个 PBFT 微轮
    │
    ▼
全局 C / E / L / next-state
    │
    ├── Central Graph-Temporal V/Q Critic
    ├── Actor-sampled COMA counterfactual
    ├── SCQ top-M simulator counterfactual
    ├── Reliability dual
    ├── Witness memory
    └── Pareto validation archive
```

---

# 14. 部署数据流

```text
节点本地观测
    │
    ▼
本地 temporal state
    │
    ▼
K 轮邻居消息传递
    │
    ▼
本地 directed bids
    │
    ▼
本地 top-b proposal
    │
    ▼
双方交换 accept bit
    │
    ▼
mutual edge activation
```

部署时不存在：

- centralized critic；
- evaluator；
- searcher；
- global decoder；
- global argsort；
- centralized budget repair。

---

# 15. 必须保留的算法基线

保持相同 actor、decoder、环境、数据和 evaluator-call budget：

1. REINFORCE-EMA；
2. REINFORCE-RLOO，且 \(M\geq2\)；
3. Graph-MAPPO；
4. Graph-Counterfactual PPO；
5. Graph-Counterfactual PPO + SCQ；
6. myopic contextual-bandit actor；
7. temporal CTDE actor。

比较：

- sample efficiency；
- wall-clock；
- reliability risk；
- energy；
- P95 latency；
- constrained hypervolume；
- OOD N；
- critic OOD error；
- unknown-to-witness discovery；
- train–deploy gap。

---

# 16. 机制激活契约

每个声称启用的机制必须有运行时 assertion 和日志证据。

## RLOO

- \(M\geq2\)；
- 每场景 reward variance；
- RLOO baseline variance；
- evaluator calls。

## Temporal

- episode length \(>1\)；
- 状态变化非零；
- previous topology 有效；
- reconfiguration 或其他动作—未来耦合非零。

## Counterfactual

- \(K_{\mathrm{cf}}\geq1\)；
- 反事实动作与实际动作有差异；
- Q 差分方差非零；
- actor 使用 per-agent advantage。

## SCQ

- top-M \(>0\)；
- simulator fork 成功；
- exact difference 被 critic loss 使用；
- witness memory 更新可追踪。

## Dual

- constraint target 明确；
- residual 可正可负；
- \(\lambda\) 可升可降。

## Pareto

- validation archive 大小；
- 非支配点数量；
- hypervolume；
- checkpoint 选择原因。

---

# 17. 配置层级

## Smoke

只验证：

- imports；
- shape；
- forward/backward；
- rollout；
- checkpoint；
- evaluator integration。

不得产生科研结论。

## Pilot

验证：

- 机制真正激活；
- loss 非零；
- 信号方差；
- 小规模方向性结果。

不得作为最终 headline。

## Research

必须：

- 正式数据；
- 正式参数；
- ≥5 独立 seed；
- held-out；
- OOD N；
- CI；
- 完整 artifacts。

---

# 18. 验收指标

## 协议正确性

- quorum intersection property；
- liveness；
- fixed fault set；
- Torch/reference parity；
- exact small-N parity。

## 环境正确性

- relay hop 语义；
- phase-specific messages；
- client/validator accounting；
- timeout-aware latency；
- control communication cost；
- reconfiguration cost。

## 学习

- critic explained variance；
- counterfactual rank correlation；
- SCQ difference error；
- PPO KL；
- dual convergence；
- witness discovery。

## 泛化

训练：

\[
N\in\{8,12,16\}.
\]

评估：

- held in-range；
- N=24；
- 资源允许时 N=32/48；
- 功率；
- RSU 数量；
- 密度；
- 遮挡；
- \(f/n\)；
- 偏好 \(\omega\)。

## 主要结果

- reliability violation rate；
- CVaR reliability shortfall；
- energy；
- expected/P95 latency；
- reconfiguration cost；
- policy communication cost；
- constrained hypervolume；
- unknown-to-witness discovery；
- multi-seed CI。

---

# 19. 最终结论

项目最终不应继续以“critic-free”作为身份核心，而应以：

\[
\boxed{
\text{训练期高效全局信用}
+
\text{部署期严格局部执行}
}
\]

为核心。

推荐主架构：

\[
\boxed{
\text{CTDE Graph-Counterfactual PPO}
+
\text{SCQ exact counterfactual supervision}
+
\text{Recurrent Directional PNA Actor}
+
\text{Vector Graph-Temporal Critic}
+
\text{Chance/CVaR reliability constraint}
+
\text{Preference-conditioned Pareto policy}
}
\]

中央 critic 负责训练效率和长期价值；local actor 与 mutual decoder 保证部署阶段完全去中心化、分布式执行。
