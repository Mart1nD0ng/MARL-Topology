# MARL-Topology 技术架构设计说明（第二版）

> 文档类型：技术支持文档  
> 项目：`Mart1nD0ng/MARL-Topology`  
> 初版日期：2026-06-22  
> 第二版修订：2026-06-23  
> 目标架构：**CTDE Graph-Counterfactual PPO + SCQ 反事实监督 + 预算条件化无序子集策略 + 可靠性约束 + 去中心化分布式执行**

---

## 0. 第二版修订范围与当前状态

本版在第一版基础上加入 Phase 0–7 的代码审查结论，并以这些结论覆盖旧设计中不再成立的部分。

### 0.1 当前阶段状态

| 阶段 | 当前状态 | 第二版判断 |
|---|---|---|
| Phase 0 | CTDE 治理边界已建立，旧 contract 大量退役 | 基本完成，但正式 config tier 与统一 manifest 仍需补齐 |
| Phase 1 | safe quorum 与 Torch quorum-tail 已实现；fixed-\(B\) 已接入生产 | 部分完成；必须修复 effective \(f\) 的生产记录、故障集合大小与 honest-primary 条件平均的单调性问题 |
| Phase 2 | one-hop matrix + relay DP 已在 production operating point 启用 | 基本完成；应删除错误 legacy default，并补齐 latency-aware relay integration |
| Phase 3 | tri-state 字段、状态分类和 witness memory primitive 已存在 | integration 未完成；训练仍按旧 `feasible_exists` 过滤 unknown |
| Phase 4 | quorum-completion latency primitive 已存在并启用 | integration 未完成；三阶段仍复用同一组 all-pairs records，尚不是严格 phase-specific accounting |
| Phase 5 | 存在 trajectory 数据结构和车辆运动生成 substrate | **尚未建立动态任务**；主训练仍是 \(T=1\) contextual bandit，没有拓扑持久性、重配置成本、动作影响未来的 transition 或 Temporal Value Test |
| Phase 6 | per-agent action API 与 Gumbel 修复已实现 | 必须 REVISE；ordered Plackett–Luce 将同一无序子集的 \(k!\) 个排列误作不同动作 |
| Phase 7 | centralized graph critic 与 PPO 分支已写入主干 | WIP/BLOCKED；除排列熵爆炸外，critic 前向被 `no_grad` 包裹、PPO 使用 joint ratio，均需修复 |

### 0.2 第二版最重要的新增决策

1. **废弃生产路径中的 ordered top-\(k\) Plackett–Luce 动作定义。**
2. 节点动作改成预算约束下的**无序 proposal subset**。
3. 使用 **Budget-Conditioned Unordered Subset Policy（BCSP）**：
   \[
   \pi_i(S)
   =
   \frac{
   \mathbf 1[|S|\le b_i]\exp(\sum_{e\in S}\theta_e)
   }{
   \sum_{|A|\le b_i}\exp(\sum_{e\in A}\theta_e)
   }.
   \]
4. 精确 log-probability、采样和 entropy 由 \(O(m_ib_i)\) 动态规划完成；当 \(b_i\ge m_i\) 时退化为 \(O(m_i)\) 独立 Bernoulli。
5. PPO 使用 **per-agent importance ratio**，不再使用随智能体数量指数放大的 joint ratio。
6. local mutual acceptance 继续作为唯一训练—部署一致 decoder。
7. 第一优先级不是继续 Phase 8，而是修复并重新验证 Phase 0–7。

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

不能在不同接收者或不同阶段分别删除概率最大的 \(f\) 个节点。应定义一个贯穿整个协议的固定故障集合：

\[
B\subseteq V,
\qquad
|B|\le f.
\]

理论目标为：

\[
C_{\mathrm{robust}}(x)
=
\min_{\substack{B\subseteq V\\|B|\le f}}
C(x;B).
\]

## 4.7.1 必须明确 primary 语义

若 scalar reliability 定义为对诚实 primary 的条件平均：

\[
C_{\mathrm{honest}}(B)
=
\frac{1}{|V\setminus B|}
\sum_{p\notin B}C_p(B),
\]

则不能未经证明地假设：

\[
B_1\subset B_2
\Longrightarrow
C_{\mathrm{honest}}(B_2)\le C_{\mathrm{honest}}(B_1).
\]

原因是：将一个低可靠性 primary 加入 \(B\) 会把它从平均分母中删除，条件平均可能反而上升。因此，仅枚举 \(|B|=f\) 并声称最坏情况必然在那里，在当前语义下并不充分。

生产实现必须选择以下一种可审计语义：

### 方案 A：固定原始 primary 分布

\[
C(B)
=
\sum_{p\in V}\rho_p C_p(B),
\]

并显式定义 faulty primary 的 view-change/timeout 结果。该定义更适合比较不同 \(B\) 的绝对风险。

### 方案 B：诚实 primary 条件平均

继续使用 \(C_{\mathrm{honest}}(B)\)，但必须真实计算：

\[
\min_{0\le r\le f}
\min_{\substack{B\subseteq V\\|B|=r}}
C_{\mathrm{honest}}(B),
\]

不得只枚举 \(|B|=f\)。

## 4.7.2 精确、近似与证书必须区分

- exact enumeration：给出真实 hard minimum；
- branch-and-bound：只有在上下界闭合时才是 exact；
- greedy fixed-\(B\)：只找到一个合法 \(B\)，因此
  \[
  C(B_{\mathrm{greedy}})
  \ge
  \min_B C(B).
  \]
  它是对最坏风险的**乐观近似**，不能无误差界地称为 certified robust reliability；
- softmin：只用于训练平滑，最终评估回到 hard min。

每次生产评价必须记录：

```text
validator_count
configured_fault_tolerance
effective_fault_tolerance
quorum
external_quorum
fault_strategy
fault_set_count
enumeration_exact
worst_case_fault_set
```

不能再从节点数推断一个并未真正传入 evaluator 的 \(f\)。

## 4.7.3 测试要求

必须增加：

1. 非对称 primary reliability 下的 \(|B|\le f\) 反例测试；
2. production config 中 effective \(f,q\) 的 integration test；
3. exact 与 greedy 的 gap 统计；
4. perfect-link、single-weak-primary、single-critical-relay 等机制测试；
5. N=24 真实 operating-point 的 strategy/complexity artifact。

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

# 4.11 当前环境 integration 的强制修复项

现有 primitive 已显著改进，但生产 evaluator 仍需完成以下闭环：

1. `pre_prepare`、`prepare`、`commit` 不能继续复用同一组 all-pairs network records。
2. pre-prepare 必须是 primary-to-backups 消息计划。
3. prepare/commit 必须由 validators 发送，clients 不产生 vote。
4. clients 可以作为 relay，并只计实际转发成本。
5. quorum-completion latency 必须以各阶段真实消息计划为输入。
6. policy communication、MAC control、reconfiguration 和 view-change 成本需显式记录。
7. corrected environment flags 应成为 production default；legacy 语义只能显式请求。
8. tri-state `unknown` 不能继续通过旧 `feasible_exists=False` 被训练主干静默删除。

在以上 integration 完成前，不能将 Phase 1–4 标记为完全完成。

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


# 7. Actor、无序子集策略与去中心化 Decoder

# 7.1 环境真实动作是无序 proposal subset

节点 \(i\) 的局部动作应定义为：

\[
S_i\subseteq\mathcal E_i,
\qquad
|S_i|\le b_i,
\]

其中 \(\mathcal E_i\) 是节点 \(i\) 可提议的 incident edge 集合。

旧 ordered Plackett–Luce 动作：

\[
o_i=(e_{i_1},e_{i_2},\ldots,e_{i_k})
\]

会把同一集合 \(S_i\) 的 \(k!\) 个排列当成不同动作，但 decoder 只使用集合：

\[
S_i=\{e_{i_1},\ldots,e_{i_k}\}.
\]

当 \(k=m_i\) 时，所有 \(m_i!\) 个排列都映射到唯一集合 \(\mathcal E_i\)。此时真实集合动作 entropy 为 0，而 ordered entropy 可接近 \(\log(m_i!)\)。这既造成错误的探索信号，也造成 factorial 计算爆炸。

因此，**order 只能是采样实现细节，不能进入 PPO action probability。**

---

# 7.2 预算条件化无序子集策略（BCSP）

令 actor 为每条 incident edge 输出局部自然参数：

\[
\theta_{i,e}=\frac{z_{i,e}}{T}.
\]

定义：

\[
\boxed{
\pi_i(S_i\mid o_i)
=
\frac{
\mathbf1[|S_i|\le b_i]
\exp\left(\sum_{e\in S_i}\theta_{i,e}\right)
}{
Z_i(\theta)
}
}
\]

其中：

\[
Z_i(\theta)
=
\sum_{\substack{A\subseteq\mathcal E_i\\|A|\le b_i}}
\exp\left(\sum_{e\in A}\theta_{i,e}\right).
\]

这是独立 Bernoulli exponential-family 分布在 cardinality constraint：

\[
|S_i|\le b_i
\]

下的条件分布。它直接在真实无序动作空间上定义概率。

---

# 7.3 MAP 动作与部署 decoder 一致

BCSP 的 MAP 动作为：

\[
S_i^\star
=
\arg\max_{|S|\le b_i}
\sum_{e\in S}\theta_{i,e}.
\]

因此：

1. 不选择负 logit 边；
2. 对正 logit 降序；
3. 最多选择前 \(b_i\) 条。

这与当前部署规则完全一致：

```text
logit >= 0
local top-b
mutual acceptance
```

训练阶段不再使用 hard gate 截断概率支持；部署时使用 MAP decoder。这样 PPO 内不需要“冻结 gate”，也不会因 logits 跨越 0 而改变动作支持。

---

# 7.4 Partition function 的 \(O(mb)\) 动态规划

设节点有 \(m\) 条 incident edges。

定义：

\[
A_{j,r}
=
\log
\sum_{\substack{
S\subseteq\{1,\ldots,j\}\\
|S|=r
}}
\exp\left(\sum_{e\in S}\theta_e\right).
\]

初始化：

\[
A_{0,0}=0,
\qquad
A_{0,r>0}=-\infty.
\]

递推：

\[
A_{j,r}
=
\operatorname{logaddexp}
\left(
A_{j-1,r},
A_{j-1,r-1}+\theta_j
\right).
\]

最终：

\[
\log Z
=
\operatorname{logsumexp}_{0\le r\le \min(b,m)}
A_{m,r}.
\]

计算复杂度：

\[
O(mb),
\]

空间可压缩为：

\[
O(b).
\]

不得通过枚举：

\[
\frac{m!}{(m-k)!}
\]

个顺序计算 log-probability 或 entropy。

---

# 7.5 \(b\ge m\) 的线性快路径

当预算不构成约束：

\[
b\ge m,
\]

则所有子集合法：

\[
Z
=
\prod_{e=1}^{m}(1+e^{\theta_e}).
\]

因此：

\[
\log Z
=
\sum_e\operatorname{softplus}(\theta_e).
\]

策略退化为独立 Bernoulli：

\[
P(e\in S)=\sigma(\theta_e).
\]

复杂度：

\[
O(m).
\]

因此当前真实 blocker 中的：

\[
m=15,\quad b=64
\]

应是线性场景，而不是 \(15!\) 场景。

---

# 7.6 精确 log-probability

对记录的无序集合 \(S_i\)：

\[
\boxed{
\log\pi_i(S_i)
=
\sum_{e\in S_i}\theta_{i,e}
-
\log Z_i
}
\]

PPO rollout 只保存：

- local candidate edge ids；
- accepted subset bitmask/indices；
- old subset log-probability；
- local budget；
- temperature。

不保存或重评分 accepted order。

---

# 7.7 精确采样

先采样 cardinality：

\[
P(K=r)
=
\exp(A_{m,r}-\log Z).
\]

给定 \(K=r\)，从 \(j=m\) 向前进行 backward sampling：

\[
P(x_j=1\mid r)
=
\exp
\left(
\theta_j+A_{j-1,r-1}-A_{j,r}
\right).
\]

若选择第 \(j\) 条边，则令 \(r\leftarrow r-1\)。

预处理复杂度：

\[
O(mb),
\]

单次采样复杂度：

\[
O(m).
\]

实现必须使用稳定 log-domain DP，并支持 batch、autograd 与 device-preserving tensor 创建。

---

# 7.8 精确 entropy

因为：

\[
\log\pi(S)
=
\theta(S)-\log Z,
\qquad
\theta(S)=\sum_{e\in S}\theta_e,
\]

所以：

\[
H(\pi)
=
\log Z-\mathbb E_\pi[\theta(S)].
\]

边缘包含概率为：

\[
\mu_e
=
P(e\in S)
=
\frac{\partial\log Z}{\partial\theta_e}.
\]

因此：

\[
\boxed{
H(\pi)
=
\log Z-\sum_e\theta_e\mu_e
}
\]

可通过 DP 的 autograd 精确获得，复杂度仍为：

\[
O(mb).
\]

也可用 expectation semiring 直接计算。定义：

\[
Z_{j,r}
=
\sum_{|S|=r}e^{\theta(S)},
\]

\[
G_{j,r}
=
\sum_{|S|=r}e^{\theta(S)}\theta(S),
\]

递推：

\[
Z_{j,r}
=
Z_{j-1,r}+e^{\theta_j}Z_{j-1,r-1},
\]

\[
G_{j,r}
=
G_{j-1,r}
+
e^{\theta_j}
\left(
G_{j-1,r-1}
+
\theta_jZ_{j-1,r-1}
\right).
\]

最终：

\[
H
=
\log Z-\frac{G}{Z}.
\]

生产实现必须使用 log-domain 或 rescaling，避免数值溢出。

---

# 7.9 Entropy 的规模归一化

固定 entropy coefficient 不能直接乘场景 joint entropy，因为其尺度会随 N、degree 和 budget 增长。

节点合法动作数：

\[
|\mathcal A_i|
=
\sum_{r=0}^{\min(b_i,m_i)}
\binom{m_i}{r}.
\]

最大 entropy：

\[
H_i^{\max}
=
\log|\mathcal A_i|.
\]

定义：

\[
\bar H_i
=
\begin{cases}
H_i/H_i^{\max},&H_i^{\max}>0,\\
0,&\text{otherwise}.
\end{cases}
\]

actor loss 使用 agent-average normalized entropy：

\[
\bar H
=
\frac{1}{\sum_sN_s}
\sum_{s,i}\bar H_{s,i}.
\]

这样 entropy coefficient 在 N=8 与 N=48 上具有可比较含义。

---

# 7.10 稀疏性来自目标，不来自人工硬上限

BCSP 允许：

\[
|S_i|=0,1,\ldots,b_i,
\]

不会像旧策略那样只要通过 gate 就选满预算。

若一条边的边际可靠性贡献不足以抵消：

- 发送/接收能耗；
- MAC 时隙；
- 干扰；
- 时延；
- 重配置成本；

actor 会学习负 logit，形成稀疏 proposal。

不能通过以下方式伪造扩展性：

- 限制最大节点度；
- 固定小候选 top-\(K\)；
- 人为缩小物理预算；
- 用 permutation threshold 切换数学定义。

应报告：

\[
|E_{\mathrm{policy}}|,
\quad
|E_{\mathrm{witness}}|,
\quad
\text{edge redundancy},
\quad
\Delta C_e.
\]

---

# 7.11 Local mutual acceptance

节点独立采样 \(S_i\)，最终边：

\[
x_{ij}
=
\mathbf1[
(i,j)\in S_i
\land
(i,j)\in S_j
].
\]

联合 proposal policy：

\[
\pi_\theta(\mathbf S\mid\mathbf o)
=
\prod_i\pi_i(S_i\mid o_i).
\]

joint log-probability可用于诊断：

\[
\log\pi(\mathbf S)
=
\sum_i\log\pi_i(S_i),
\]

但 PPO 更新必须使用 per-agent ratio，不能把 joint ratio 作为主 actor ratio。

mutual decoder 是确定性环境映射：

\[
x=D_{\mathrm{mutual}}(\mathbf S).
\]

训练、验证与部署必须调用同一语义。

---

# 7.12 Actor 网络主体

BCSP 只替换动作分布，不替代 actor encoder。

最终 actor 仍建议使用：

> **Preference-conditioned Recurrent Directional PNA Actor**

包含：

1. local temporal encoder；
2. directional message passing；
3. PNA aggregation；
4. recurrent shared update；
5. directed bid/logit head；
6. BCSP subset sampler；
7. local mutual-acceptance decoder。

动作图：

\[
G_{\mathrm{act}}=(V,E_{\mathrm{act}})
\]

与通信图：

\[
G_{\mathrm{comm}}=(V,E_{\mathrm{comm}})
\]

必须分离，且部署消息只能沿物理可通信邻居传递。

---

# 7.13 Actor 特征与实现要求

节点特征：

- role embedding；
- 位置、速度、本地历史；
- 剩余预算；
- queue/battery；
- validator/client role；
- 上一拓扑局部状态；
- \(f/n,q/n,\tau,\omega\)。

有向边特征：

- direction-specific delivery；
- SINR；
- LOS/NLOS/NLOSv；
- expected attempts；
- energy；
- latency；
- MAC conflict；
- previous activation；
- bridge/local-connectivity 特征。

实现必须：

- 使用 `tensor.new_zeros()` 等 device-preserving API；
- 支持 CPU/CUDA；
- 支持 variable N/E batching；
- 不编码节点 ID；
- 不输入全局可靠性、teacher label 或 future outcome。


# 8. Centralized Critic 设计

# 8.1 Critic 的训练边界

critic 仅在 CTDE 训练中使用，部署时完全删除。

它负责：

- scene-level value baseline；
- 后续 per-agent counterfactual Q；
- energy/latency/reliability vector return；
- SCQ exact difference calibration；
- 时序任务中的 bootstrap。

critic 与 actor：

- 参数完全独立；
- optimizer 完全独立；
- encoder 不共享；
- critic gradient 不得进入 actor；
- actor 部署 artifact 不得依赖 critic。

---

# 8.2 输入与图结构

critic 可读取：

- 全局 candidate/action graph；
- 所有节点联合观测或 joint observation history；
- proposal subsets；
- mutual decoder 后 topology；
- \(n,f,q,\tau,\omega\)；
- PBFT/MAC 状态；
- dynamic task 下的 previous topology。

不得把 teacher topology、search result 或 held/test future outcome 当作 critic 输入。

critic 应为 variable-N graph model，不得 flatten 固定节点数。

---

# 8.3 网络结构

建议：

1. node/edge encoder；
2. 4–6 层 PNA 或 Graph Transformer；
3. action-conditioned edge channel；
4. masked mean/max/attention readout；
5. vector V/Q heads。

单步 Phase 7 可先使用：

\[
V_\phi(s).
\]

Phase 8 增加：

\[
Q_\phi(s,\mathbf S).
\]

后续 vector heads：

\[
V_E,V_L,V_C,
\qquad
Q_E,Q_L,Q_C.
\]

---

# 8.4 Critic 梯度契约

训练用 critic forward 不能位于：

```python
@torch.no_grad()
```

上下文中。

应分成：

```text
critic_value_train(...)   # grad enabled
critic_value_rollout(...) # no_grad wrapper
```

必须有端到端测试证明：

1. 通过正式 trunk helper 计算 value；
2. critic loss backward；
3. optimizer step；
4. 至少一个 critic 参数发生变化；
5. actor 参数保持不变。

shape test 或直接调用 critic module 的 test 不能替代该 integration test。

---

# 8.5 Vector heads 与标量化

\[
Q^{\omega,\lambda}
=
-\omega_EQ_E
-\omega_LQ_L
-\lambda Q_C.
\]

标量化应在 critic heads 外完成，使同一 critic 支持不同 preference。

CVaR 阶段可添加 quantile cost head，但必须在 scalar/vector critic 稳定后再启用。

---

# 8.6 Batching、设备与 checkpoint

必须支持：

- padded graph batch 或 PyG-style graph batch；
- PPO minibatch；
- critic minibatch；
- CUDA device；
- critic state；
- critic optimizer state；
- scheduler state；
- normalization state；
- resume 后指标连续。

当前逐场景循环只能作为 correctness pilot，不能作为最终规模化实现。


# 9. Graph-MAPPO 与 Graph-Counterfactual PPO

# 9.1 Phase 7 Graph-MAPPO baseline

单步任务中：

\[
A_s
=
R_s-V_\phi(s).
\]

Phase 7 可以暂时让同一 scene 内所有 agent 共享 \(A_s\)，但 action ratio 必须是 per-agent ratio。

---

# 9.2 禁止 joint PPO ratio

旧实现使用：

\[
\rho_s
=
\exp
\left(
\sum_i
[
\log\pi_i^{\mathrm{new}}(S_i)
-
\log\pi_i^{\mathrm{old}}(S_i)
]
\right).
\]

随着 agent 数增加，log-ratio 方差累积，\(\rho_s\) 会快速接近 0 或爆大，clip fraction 容易趋近 1。

生产实现使用：

\[
\boxed{
\rho_{s,i}
=
\exp
\left(
\log\pi_i^{\mathrm{new}}(S_{s,i})
-
\log\pi_i^{\mathrm{old}}(S_{s,i})
\right)
}
\]

Phase 7 actor loss：

\[
\mathcal L_{\mathrm{PPO}}
=
-
\frac{1}{\sum_sN_s}
\sum_{s,i}
\min
\left(
\rho_{s,i}A_s,
\operatorname{clip}(\rho_{s,i},1-\epsilon,1+\epsilon)A_s
\right).
\]

Phase 8 再将 \(A_s\) 替换为 \(A_{s,i}^{\mathrm{CF}}\)。

---

# 9.3 Per-agent KL 与 clip diagnostics

记录：

\[
\operatorname{KL}_{\mathrm{approx}}
=
\frac{1}{\sum_sN_s}
\sum_{s,i}
\left[
(\rho_{s,i}-1)
-
\log\rho_{s,i}
\right].
\]

clip fraction 也按 agent-action 统计。

joint log-probability只用于：

- debug；
- factorization check；
- scene likelihood diagnostic。

不得用于主 PPO clipping。

---

# 9.4 Graph-Counterfactual advantage

Phase 8 使用 action-conditioned critic：

\[
A_{i}^{E}
=
Q_E(s,\mathbf S)
-
\mathbb E_{\tilde S_i\sim\pi_i}
Q_E(s,\tilde S_i,\mathbf S_{-i}),
\]

时延和可靠性同理：

\[
A_i^{\omega,\lambda}
=
-\omega_EA_i^E
-\omega_LA_i^L
-\lambda A_i^C.
\]

反事实 subset 必须是无序集合，并重新经过同一 mutual decoder。

用于无偏 policy baseline 的反事实样本必须在给定 \(o_i,\mathbf S_{-i}\) 后独立于实际 \(S_i\)。

---

# 9.5 BCSP 下的独立反事实采样

对节点 \(i\)，从同一 BCSP 额外采样：

\[
\tilde S_i^{(1)},\ldots,\tilde S_i^{(K_{\mathrm{cf}})}
\sim\pi_i.
\]

baseline：

\[
b_i
\approx
\frac{1}{K_{\mathrm{cf}}}
\sum_{k=1}^{K_{\mathrm{cf}}}
Q(s,\tilde S_i^{(k)},\mathbf S_{-i}).
\]

SCQ sensitivity-guided add/remove/swap 可用于 critic supervision 和 witness discovery；若其候选依赖实际 \(S_i\)，则不能未经修正直接充当无偏 policy baseline。

---

# 9.6 Actor loss

\[
\mathcal L_{\mathrm{actor}}
=
\mathcal L_{\mathrm{PPO}}
-
\beta_H
\frac{1}{\sum_sN_s}
\sum_{s,i}\bar H_{s,i}.
\]

其中 \(\bar H_{s,i}\) 是按合法 subset action count 归一化的 entropy。

---

# 9.7 Critic loss

单步 scalar value：

\[
\mathcal L_V
=
\mathbb E[(R-V_\phi(s))^2].
\]

vector heads：

\[
\mathcal L_V
=
\sum_{k\in\{E,L,C\}}
w_k
(V_k-\hat R_k)^2.
\]

action-conditioned Q：

\[
\mathcal L_Q
=
\sum_k
w_k
(Q_k-\hat Y_k)^2.
\]

critic coefficient 与独立 critic learning rate 不应同时无审计地调节；需要记录实际 gradient norm 和 effective step size。

---

# 9.8 公平预算

比较 EMA、RLOO、Graph-MAPPO、Counterfactual PPO 时至少统一：

- evaluator calls；
- environment steps；
- actor architecture；
- decoder；
- data split；
- seed；
- checkpoint rule。

不能只统一 update 数。

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

# 10.5 SCQ 与无序 subset action

SCQ 的局部反事实必须操作无序集合：

- add edge：
  \[
  \tilde S_i=S_i\cup\{e\};
  \]
- remove edge：
  \[
  \tilde S_i=S_i\setminus\{e\};
  \]
- swap：
  \[
  \tilde S_i=(S_i\setminus\{e_{\mathrm{out}}\})\cup\{e_{\mathrm{in}}\}.
  \]

不能枚举同一 subset 的不同 order。

SCQ top-\(M\) 的计算预算以真实 evaluator calls 计量，并应记录：

```text
counterfactual_calls
unique_subset_count
duplicate_topology_count
witness_updates
critic_difference_error
```

若多个 proposal subsets 经 mutual decoder 映射到相同最终 topology，可共享 evaluator cache。

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


# 16. 机制激活与规模契约

每个声称启用的机制必须有 runtime assertion、日志和 production-scale 测试。

## RLOO

- \(M\ge2\)；
- 若 `baseline=rloo` 且 \(M=1\)，fail fast；
- 记录 reward variance、RLOO variance 和 evaluator calls。

## BCSP 无序子集策略

- production action path 禁止 `itertools.permutations`；
- small \(m\) 与 exhaustive subset enumeration 对齐；
- \(b\ge m\) 与 independent Bernoulli 对齐；
- MAP 与 local deterministic decoder 对齐；
- \(m=15,b=64\) 在严格时间预算内完成；
- \(m=128,b=64\) 展示 polynomial scaling；
- logp/entropy gradcheck；
- two orders mapping to same set 只能得到一个 environment action；
- \(b=m\) 时不再出现 \(\log(m!)\) 的虚假 order entropy。

## Graph-MAPPO

- critic 正式 helper 前向具有梯度；
- optimizer step 后 critic 参数变化；
- actor 参数不被 critic update 改变；
- PPO ratio 在 epoch 0 对每个 agent 等于 1；
- per-agent KL/clip fraction 被记录；
- real operating-point shard smoke 退出 0；
- critic/checkpoint resume 可复现。

## Temporal

- episode length \(>1\)；
- previous topology 生效；
- action 影响 future return；
- reconfiguration 或 queue/battery 等状态耦合非零；
- Temporal Value Test 已执行。

## Counterfactual / SCQ

- \(K_{\mathrm{cf}}\ge1\)；
- unique subset count 非零；
- per-agent advantage 不全相同；
- SCQ exact evaluator difference 真正进入 critic loss；
- held/test 不反馈 train witness。

## Dual

- target 明确；
- residual 可正可负；
- \(\lambda\) 可升可降。

## Pareto

- 至少两个 preference；
- archive 非空；
- constrained hypervolume；
- checkpoint 由 validation archive 选择。

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

项目目标保持为：

\[
\boxed{
\text{训练期高效全局信用}
+
\text{部署期严格局部执行}
}
\]

第二版确认的主架构为：

\[
\boxed{
\text{BCSP 无序子集 Actor}
+
\text{Local Mutual Acceptance}
+
\text{Per-Agent Graph-MAPPO}
+
\text{Graph-Counterfactual PPO}
+
\text{SCQ Exact Supervision}
}
\]

并进一步结合：

\[
\boxed{
\text{Recurrent Directional PNA}
+
\text{Vector Graph-Temporal Critic}
+
\text{Chance/CVaR Reliability Constraint}
+
\text{Preference-Conditioned Pareto Policy}
}
\]

关键原则：

1. central critic 合法，central decoder 非法；
2. agent 动作是无序 proposal subset，而不是其排列；
3. 预算约束通过 BCSP 与 local decoder 精确处理，不靠人工缩小 degree/budget；
4. PPO 使用 per-agent ratio；
5. 稀疏性应由可靠性—能耗—时延目标自然产生；
6. Phase 0–7 修复验证完成前，不继续宣称后续算法有效；
7. 当前项目尚未建立真正动态任务，必须在 Phase 5 完成两时间尺度 transition 与 Temporal Value Test 后再决定最终采用 bandit 还是 Dec-POMDP。


---

## 附录 A：第二版重点验证清单

```text
[ ] production effective f/q/fault strategy 被记录
[ ] fixed-B 对 |B|<=f 的语义经过非对称反例测试
[ ] tri-state 真正控制训练数据选择
[ ] phase-specific PBFT message plan 接入 evaluator
[ ] dynamic task transition 已建立或明确由 Temporal Value Test 否决
[ ] ordered Plackett-Luce 从生产动作概率中删除
[ ] BCSP exact DP / sampling / entropy 完成
[ ] per-agent PPO ratio 完成
[ ] critic formal helper 可训练
[ ] real-shard Graph-MAPPO smoke 完成
[ ] CUDA/device tests 完成
[ ] evaluator-call-fair A/B 完成
```
