# MARL-Topology 技术支持文档：Belief-Guided Evidence-Gated Residual PPO

> 文档类型：技术支持文档  
> 目标：修复 stale CSI 下时序模块无效、residual policy 被 anchor clamp 或 collapse、PPO/critic 未进入 residual 路径、oracle 模仿可能锁死上限等问题。  
> 推荐方法：**Belief-Guided Evidence-Gated Residual PPO**  
> 中文名：**信念预测引导的证据门控残差 PPO**  
> 核心思想：让 GRU 明确学习“从过时/部分 CSI 恢复当前真实 CSI”，让 residual policy 只在“有证据的局部编辑空间”中通过 PPO 小步、安全地偏离 local_hysteresis anchor。

---

# 1. 当前失败机制的统一解释

上一轮 Q14 给出了一个比“MARL 不行”更精确的诊断：

```text
stale CSI 确实造成感知损失；
GRU 模块存在且前向活跃；
但 GRU 对最终 topology decision 的影响被 logit 饱和、离散 MAP residual decode、缺少 CSI 预测监督、REINFORCE 高方差共同抹掉；
residual policy 放松 anchor 后没有朝更好方向走，而是进入 clamp / collapse 两个 basin。
```

因此，本轮不是简单“加 PPO”或“加 oracle imitation”，而是同时修复四条链：

```text
时序信号链：stale history -> GRU -> current CSI belief -> edge/residual score
训练稳定链：residual action -> old/new logp -> PPO clip/KL -> CTDE critic
方向监督链：central repair/prune -> beneficial edit labels -> repair/safety heads
安全探索链：anchor -> evidence-gated residual candidates -> adaptive KL/safety constraint
```

---

# 2. Stale CSI 下时序模块的正确角色

## 2.1 真实状态与 actor 观测

真实环境在时刻 \(t\) 有当前信道：

\[
g_t = \{g_{e,t}\}_{e\in E}.
\]

真实 PBFT 可靠性、能耗、时延永远用 \(g_t\) 计算。

actor 只能看到：

\[
\hat g_t,
\]

其中 \(\hat g_t\) 是 stale / partial / noisy CSI，例如：

\[
\hat g_{e,t}=g_{e,t-\delta}
\]

或：

\[
\hat g_{e,t}=
\begin{cases}
g_{e,t-\delta}, & m_{e,t}=1,\\
\hat g_{e,t-1}, & m_{e,t}=0.
\end{cases}
\]

并附带：

```text
csi_age
observed_mask
relative_velocity
distance_delta
observed_csi_delta
previous_topology
```

## 2.2 时序模块不是“直接输出 topology”，而是“形成当前信道信念”

GRU hidden state 应近似 belief：

\[
h_t \approx b_t = P(g_t \mid \hat g_{0:t}, a_{0:t-1}, x_{0:t-1}).
\]

因此，actor 应显式输出 belief prediction：

\[
\tilde p_{e,t} = f_{\theta}^{belief}(h_t,e),
\]

用于预测当前真实 link success：

\[
p_{e,t}^{true}.
\]

这一步把“隐式时序建模”改成“被明确监督的滤波任务”。

---

# 3. CSI belief auxiliary loss

## 3.1 目标

在训练时，simulator / evaluator 知道当前真实链路成功率：

\[
p_{e,t}^{true}.
\]

actor 不读取该值，只把它作为训练 label。

在 logit 域训练更稳定：

\[
y_{e,t}^{true}
=
\operatorname{logit}
(
\operatorname{clip}(p_{e,t}^{true},\epsilon,1-\epsilon)
).
\]

模型输出：

\[
\tilde y_{e,t}.
\]

损失：

\[
L_{\text{CSI}}
=
\frac{1}{Z}
\sum_{t,e}
w_{e,t}
\operatorname{Huber}
(
\tilde y_{e,t}
-
y_{e,t}^{true}
).
\]

## 3.2 权重

不建议平均所有边，因为大量无关边会稀释学习信号。使用：

\[
w_{e,t}
=
1
+
\alpha\mathbf 1[e\in x_{t-1}]
+
\beta\mathbf 1[e\in x_H]
+
\gamma\chi_{e,t}^{quorum}
+
\eta\mathbf 1[e\in \mathcal C_{\text{residual}}].
\]

其中：

- \(x_H\)：local_hysteresis anchor；
- \(\chi^{quorum}\)：quorum criticality 或 \(D_{\text{quorum}}\) 相关重要性；
- \(\mathcal C_{\text{residual}}\)：add/remove/swap候选边集合。

## 3.3 验收指标

```text
true-current psucc prediction MSE
logit-psucc Huber loss
top-k critical-link prediction recall
recurrent vs memoryless prediction improvement
belief prediction improvement under delay1/delay2/partial
```

若 recurrent 在这个任务上仍无法优于 memoryless，则不应继续声称“时序模块已有效”。

---

# 4. 解决 logit 饱和

## 4.1 当前饱和问题

当前 actor 使用：

\[
z=10\tanh(raw/10).
\]

当：

\[
|raw|\gg 10
\]

时：

\[
\frac{\partial z}{\partial raw}\approx0.
\]

即使 GRU hidden 改变了 raw，最终 logit 也可能被量化到同一个 \(\pm 10\)。

## 4.2 技术修改

### 方案 A：独立 residual head

不要让 residual policy 复用全拓扑 activation head。新增：

```text
belief_head
repair_head
safety_head
residual_policy_head
```

residual policy head 输出小幅 logits：

\[
z_{\Delta,e}\in[-z_{\max}^{res},z_{\max}^{res}]
\]

其中：

\[
z_{\max}^{res}\in[2,4]
\]

而不是 10。

### 方案 B：raw-logit L2

\[
L_{\text{raw}}
=
\lambda_{\text{raw}}
\mathbb E[raw^2].
\]

### 方案 C：饱和率监控

必须记录：

```text
raw_abs_mean
raw_abs_p95
frac_abs_raw_gt_30
frac_abs_logit_gt_9_5
recurrent_memoryless_logit_delta
recurrent_memoryless_action_delta
```

### 方案 D：输入标准化修复

stale/partial CSI 特征不应只用 frame 0 统计。使用：

```text
all train frames
```

计算标准化，或对 csi_age / mask / probability / logit-psucc 使用物理固定归一化。

---

# 5. Evidence-gated residual action

## 5.1 Anchor

基础策略是可部署的 local_hysteresis：

\[
x_H = H(o_t).
\]

该策略不调用 evaluator，是 deployable anchor。

## 5.2 Residual edit

最终拓扑：

\[
x_t = x_H \oplus \Delta_t.
\]

其中：

\[
\Delta_t = \Delta^{add}_t \cup \Delta^{remove}_t \cup \Delta^{swap}_t.
\]

上一轮的错误在于：所有 edge flip 都进入 Bernoulli residual action，然后用统一 flip penalty 压住它。这会产生：

```text
强约束 -> 不动
弱约束 -> 崩溃
```

本轮改为 evidence-gated candidate set。

---

# 6. Repair / Safety / Utility heads

## 6.1 Quorum deficit

使用训练时 evaluator 计算：

\[
D_{\text{quorum}}(x).
\]

它来自 per-phase/per-receiver expected quorum shortfall：

\[
\delta_{j,h}
=
\mathbb E[(q_h-S_{j,h})_+].
\]

全局聚合可用：

\[
D_{\text{quorum}}
=
\operatorname{CVaR}_{\alpha}
(\{\delta_{j,h}\}_{j,h}).
\]

## 6.2 Repair gain

对于 add / swap-in 边：

\[
g_e^{repair}
=
D_{\text{quorum}}(x)
-
D_{\text{quorum}}(x\oplus e).
\]

若：

\[
g_e^{repair}>0
\]

说明该边让 topology 更接近可行。

## 6.3 Safety risk

对于 remove / swap-out 边：

\[
r_e^{risk}
=
D_{\text{quorum}}(x\setminus e)
-
D_{\text{quorum}}(x).
\]

若：

\[
r_e^{risk}\gg0
\]

说明删除该边会破坏可靠性边界。

## 6.4 Utility gain

综合真实目标：

\[
u_{\Delta}
=
J(x_H\oplus \Delta)
-
J(x_H)
\]

其中：

\[
J = \text{discounted episode return or per-frame objective}.
\]

只对正增益编辑做 imitation：

\[
w_{\Delta}=[u_{\Delta}]_+.
\]

## 6.5 监督损失

\[
L_{\text{repair}}
=
\sum_e
\operatorname{Huber}
(
\hat g_e^{repair}-g_e^{repair}
).
\]

\[
L_{\text{safety}}
=
\sum_e
\operatorname{Huber}
(
\hat r_e^{risk}-r_e^{risk}
).
\]

\[
L_{\text{edit}}
=
-
\sum_{\Delta}
w_{\Delta}
\log\pi_\theta(\Delta\mid o,x_H).
\]

注意：

```text
不模仿完整 central oracle topology；
只模仿相对 anchor 有正增益的局部 edits。
```

---

# 7. Evidence gate

## 7.1 Add gate

允许 add 候选进入 action space 当：

\[
\hat g_e^{repair} - \lambda_c \widehat{cost}_e > m_{add}.
\]

## 7.2 Remove gate

允许 remove 当：

\[
\hat r_e^{risk}<m_{risk}
\]

且：

\[
\widehat{\Delta E/L}_e > m_{cost}.
\]

## 7.3 Swap gate

允许 swap 当：

\[
\hat g_{e_{in}}^{repair}
-
\hat r_{e_{out}}^{risk}
-
\lambda_c cost(e_{in},e_{out})
>
m_{swap}.
\]

## 7.4 Gated residual policy

策略只在：

\[
\mathcal A_{\text{gated}}(x_H)
\]

上采样或排序，而不是对所有候选边做 independent Bernoulli flip。

---

# 8. Residual PPO

## 8.1 旧方式

旧 residual trainer：

\[
L
=
-\sum_t \log \pi_\theta(\Delta_t)(G_t-b)
+
\lambda_{\text{flip}}\mathbb E[\#flip].
\]

问题：

```text
无 value critic
无 PPO clip
无 KL
无 entropy
统一惩罚所有 flip
高方差
容易 clamp/collapse
```

## 8.2 新方式

记录 rollout 时：

\[
\log\pi_{\theta_{old}}(\Delta_t).
\]

训练时重算：

\[
\log\pi_{\theta}(\Delta_t).
\]

ratio：

\[
\rho_t
=
\exp
(
\log\pi_{\theta}(\Delta_t)
-
\log\pi_{\theta_{old}}(\Delta_t)
).
\]

PPO loss：

\[
L_{\text{PPO}}
=
-
\mathbb E
\min
(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
).
\]

Advantage：

\[
A_t
=
G_t
-
V_{\phi}(s_t).
\]

centralized critic 可读取训练时全局真实 state / true CSI / previous topology / action summary，但部署不使用 critic。

---

# 9. KL 与 anchor 的正确关系

需要三种约束分开：

## 9.1 PPO KL

限制新旧策略：

\[
D_{KL}(\pi_{\theta_{old}}\|\pi_\theta)
\le \epsilon_{\text{ppo}}.
\]

作用：防止训练 collapse。

## 9.2 Adaptive anchor KL

限制偏离 anchor policy：

\[
D_{KL}(\pi_\theta\|\pi_H)
\le \epsilon_H(t).
\]

但 \(\epsilon_H\) 应自适应：

```text
retention下降 -> 收紧
validation改善且retention稳定 -> 放松
```

不要用固定 flip-count penalty。

## 9.3 Safety constraint

约束可行 anchor 不被破坏：

\[
P(C(x_t)<\tau \mid C(x_H)\ge\tau)
\le \delta.
\]

训练时估计：

```text
retention_rate
unsafe_edit_rate
anchor_feasible_to_policy_infeasible
```

---

# 10. Entropy 与去极化

为防止 flip 概率到 0 或 1：

\[
L
=
L_{\text{PPO}}
-
\beta_H H(\pi_\theta)
+
\lambda_{\text{raw}}L_{\text{raw}}
+
\lambda_{\text{CSI}}L_{\text{CSI}}
+
\lambda_{\text{edit}}L_{\text{edit}}
+
\lambda_{\text{repair}}L_{\text{repair}}
+
\lambda_{\text{safety}}L_{\text{safety}}.
\]

必须记录：

```text
entropy
flip_prob_mean
flip_prob_p05/p50/p95
zero-flip rate
all-flip rate
logit saturation rate
```

---

# 11. Oracle imitation 不应锁死上限

## 11.1 禁止

禁止直接学习：

\[
\pi_\theta(o)\approx x^{oracle}.
\]

这可能把策略上限锁死在 oracle，并可能让部署 actor依赖不可观测信息。

## 11.2 允许

允许学习：

\[
\Delta^{oracle}
=
\arg\max_{\Delta\in \mathcal A_{local}}
J(x_H\oplus \Delta)-J(x_H).
\]

并只模仿：

\[
[J(x_H\oplus\Delta)-J(x_H)]_+>0
\]

的编辑。

teacher 是：

```text
better-than-anchor edit generator
```

不是：

```text
final policy target
```

teacher loss 必须退火：

\[
\lambda_{\text{edit}}(t)\downarrow0.
\]

最终评估不使用 oracle。

---

# 12. 与 stale CSI 的交互

## 12.1 正确顺序

不要直接让 stale observation 模仿 true-CSI oracle topology。

正确顺序：

```text
stale observation history -> belief CSI prediction
belief + anchor -> edit prediction
residual policy -> topology
```

## 12.2 训练目标

\[
L_{\text{total}}
=
L_{\text{PPO}}
+
\lambda_{\text{CSI}}L_{\text{CSI}}
+
\lambda_{\text{edit}}L_{\text{edit}}
+
\lambda_{\text{safety}}L_{\text{safety}}
+
\lambda_{\text{repair}}L_{\text{repair}}
+
\lambda_{\text{raw}}L_{\text{raw}}
-
\beta_H H.
\]

## 12.3 验证

必须证明：

```text
belief prediction improves under recurrence
belief prediction improves topology edits
policy uses belief, not leaked true CSI
```

---

# 13. 评价指标

## 13.1 标准指标

```text
per-frame feasibility
episode return
energy
latency
switches
timeout
```

## 13.2 时序指标

```text
current-CSI prediction error
recurrent vs memoryless belief MSE
recurrent vs memoryless topology difference
logit saturation rate
hidden-to-logit sensitivity
```

## 13.3 residual指标

```text
policy edit rate
zero-flip rate
all-flip rate
repair success
retention
safe remove precision
unsafe edit rate
ΔC over anchor
ΔD_quorum over anchor
ΔE/L over anchor
```

## 13.4 training指标

```text
PPO KL
clip fraction
critic EV
entropy
anchor KL
edit-supervision hit rate
raw logit norm
```

---

# 14. 最小实验矩阵

## Arm 0：anchor

```text
local_hysteresis
```

## Arm 1：residual PPO only

```text
PPO clip/KL
CTDE value critic
entropy
raw-logit L2
no edit supervision
```

## Arm 2：belief auxiliary only

```text
stale CSI
belief prediction loss
no residual edit
```

## Arm 3：supervised edit only

```text
beneficial oracle-edit supervision
no RL
```

## Arm 4：belief + supervised edit

```text
CSI belief + edit prediction
```

## Arm 5：belief + supervised edit + residual PPO

```text
full method
```

必须先证明 Arm 2 / Arm 3 有效，再跑 Arm 5。

---

# 15. 终止条件

如果出现以下情况，应停止 RL 微调，回到诊断：

```text
actor cannot predict true CSI better than memoryless
actor cannot supervised-learn beneficial edits
residual PPO still has zero edit rate
residual PPO unsafe edit rate high
retention collapses
critic EV negative
logit saturation remains high
```

---

# 16. 预期结果解释

## 若 residual PPO only 有效

说明主要问题是 REINFORCE / flip penalty。

## 若 supervised edit only 有效但 PPO 后变差

说明 RL trainer仍有问题。

## 若 belief auxiliary有效但控制无收益

说明时序信息存在，但不影响可行拓扑决策。

## 若 edit supervision无效

说明局部观测不足以模仿 central repair，可能需要更强 handshake / communication 或接受 heuristic主导。

## 若 full method超过anchor

说明当前路线成立。

---

# 17. 开发硬约束

1. 不得让 actor 读取 true current CSI。
2. 不得模仿完整 central oracle topology。
3. 不得用 fixed flip penalty 作为唯一 trust region。
4. 不得声称 graph_mappo PPO 已覆盖 residual trainer，除非 residual trainer实际调用 PPO loss。
5. 不得在 logit saturation 未解决时评价 recurrence失败。
6. 不得只报 feasibility，必须报 edit/retention/saturation/CSI预测指标。
