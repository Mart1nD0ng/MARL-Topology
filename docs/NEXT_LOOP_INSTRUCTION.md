# 下一轮 `/loop` 指令（2026-06-21 定稿，证据驱动）

> 角色：MARL 研究工程师，处于「模型设计 & 训练」阶段（非写论文）。每轮：定位瓶颈 → 单一假设（一个变量）→ 最小实现 → 多配置训练+held-out+多种子+基线 → 真实数据评估 → 记 `URBAN_V2X_RESEARCH_LOG.md` → keep/rollback。卡 3–5 轮或只有违反硬约束才能达标时，停下来汇报。

---

## 1. 现状诊断（工件核对结论，非 log 叙述）

- **主干已就位且合规**：去中心化 constrained-RL（`scripts/train/train_decentralized_rl.py`）= 无 critic 的 REINFORCE+RLOO，闭式 PBFT quorum-tail 共识作奖励（`quorum_tail.py:33-73`，零 MC），Ng-Harada 势 `r=(c−τ)` + 预算对偶，τ=0.9 硬冻结，`local_mutual_assemble` 可行-by-construction 解码。
- **最强工件（已逐字段验证）**：N=24 冷启动、无 oracle，held raw 0.769–0.808 / cond 0.933 **> SA 教师天花板 0.577**——在训练范围外尺度反超中心化 oracle。
- **真实瓶颈（按severity）**：
  1. 🔴 **头条单种子**：N=24 held 仅 26 场景(15 可解)，一场景翻转≈0.04–0.05 raw，val 在 0.58–0.71 震荡 → 超出 0.577 的边际可能含采样噪声。**最致命、最廉价可修。**
  2. 🟠 **in-range RL 价值脆弱**：sparse2 的 RL>BC 只在末步策略、只在 conditional、keep-best 回退；op 只追平 ceiling。
  3. 🟡 **奖励信号粗糙**：全局 quorum-tail 当前是**单一场景标量**共享给所有节点，无 per-node 信用分配（懒节点塌缩）；dense 模式下共识对偶 lam_c 是死项。
- **创新性诚实定位**：优化机理（REINFORCE+RLOO、Lagrangian、GNN 尺度不变、3GPP env）全是标准；真正新颖的是**被优化的对象（闭式共识可靠性奖励 C1）+ 解码器/探索器对（C6/C3）+ 冷启动反超 oracle 的经验姿态（C5）**。

---

## 2. 不可降级的硬约束（INVARIANTS，2026-06-21 修订）

> **变更：原 #2「模型必须使用时序特征」已退役**——经多轮测试时序贡献为 null（5-seed GRU A/B +0.000），owner 裁决移除。其余五条不得以任何形式降级、代理或偷换：

- **I1 完全去中心化执行 AND 学习**：执行策略只用局部/邻域信息（无全局状态泄漏）；学习无中心 critic、无 CTDE gap、无全局矩阵编译。
- **I3 共识可靠性阈值 τ≥0.9**：训练奖励与评估同一常量，绝不放宽或换松代理。
- **I4 泛化/可扩展**：域随机化 + held-out + 变 N + **多种子带 CI**；禁止只在训练配置/单工作点报成功。
- **I5 单一原则化奖励**：势函数整形 + 约束式 RL/Lagrangian 对偶；**禁止堆叠加权奖励项**。
- **I6 闭式全局共识失败概率**：必须是可计算闭式（Poisson-binomial quorum tail），全网 quorum，非 MC/经验频率/局部代理；既作约束也作评估。

---

## 3. 创新点设计（SOTA，非保守）

> 目标不是"补多种子"，而是把项目从"标准机理+新对象"升级为**有方法学贡献的 SOTA 工作**。核心主张：
> **"Exact decentralized credit assignment for a global closed-form reliability objective" —— 对全局闭式共识可靠性做精确的 per-agent 分解，使无 critic 的去中心化策略获得逐 agent 梯度。** 语料 40 篇无一对闭式 Poisson-binomial 共识 tail 做精确分解（Zhang&Guo 只做邻域和的局部代理），这是真正的新机制。

三层设计（按依赖顺序）：

- **创新 A（头条新机制）—— 共识 tail 的精确 per-node 分解奖励**：对每个节点 j 计算其对全局 quorum tail 的**精确边际贡献**（leave-one-out：`c − c_{\j}`，或 Shapley 值近似），作为该节点的局部势函数奖励。关键约束守恒：**全局闭式 tail 仍是评估指标与约束（I6 不变）；per-node 分解必须解析自同一全局 tail，sum-of-attributions 重构全局值**（单测钉死），绝不退化为局部代理。预期：锐化信用分配、降梯度方差、攻懒节点塌缩 → 抬升并稳定 N=24。

- **创新 B（让约束式 RL 名副其实）—— 共识对偶变活**：借 MACPO 的 dense/sparse 成本分解，把当前死项 lam_c 接到**稀疏 episode 级共识违约成本**（二值 feasible/infeasible），与 dense 势函数结构性区分。结果：reward = 一个 per-node 势 + 两个分别作用于不同约束的活对偶（共识、预算），**仍是 I5 单一原则化结构，不是加权袋**；且"对共识+预算做对偶上升"从此名副其实。**严禁**借 MACPO 的中心 cost-critic / trust-region（破 I1）。

- **创新 C（从单工作点到 Pareto 前沿）—— 偏好条件的可靠性-能耗前沿**：actor 输入拼接每-episode 采样的权重 ω~simplex（仅 actor 侧，借 MOMA-AC/MO-MIX，**丢弃其 CTDE critic**），用单次去中心训练描出 reliability(c) vs radio-budget/energy(β·er) 的 Pareto 前沿，顺带让 β>0 命中 DoD 低能耗目标。ω 只标量化"一势+对偶"，不引入加权奖励袋（I5）。

> 可借鉴但需改造以合规的技术（每条标注受限 INVARIANT）：固定 k-邻域窗口（LNS2+RL/Naderializadeh，给 N 外推架构级论证；I1：top-k 只用局部链路特征）；REDA 去中心拍卖作更强可行解码器（I1：用市场/竞价变体非中心矩阵）；CoLight index-free GAT 局部注意力（I1：仅 k-hop 邻域，非全局自注意力）。

---

## 4. 最终交付目标

一个**可发表（top-venue）级别**的去中心化 V2X 拓扑规划 RL 系统与证据包：

1. **方法贡献**：闭式共识可靠性奖励 + 其精确 per-node 分解 + 可行-by-construction 去中心解码器/门界探索器，全程无 critic、无全局泄漏。
2. **决定性实证**：冷启动、无 oracle 的去中心化学习者在**训练范围外尺度（N≥24）反超中心化 SA oracle**，**多种子带 CI**，统计上站得住。
3. **泛化广度**：多配置（密度×功率×RSU×尺度）held-out + 变 N + Pareto 可靠性-能耗前沿。
4. **干净仓库**：单一主干（旧 MAPPO/critic 已移除），README/AGENTS 与代码一致，测试套件绿。

---

## 5. 完成判据（DoD）

- **D1 头条坐实（最高优先）**：N=24 冷启动 ≥3 独立种子（不同 `--seed` 且不同 `--split-seed`），报告 raw/cond 的 `mean±CI95`。**通过 = `raw_mean − 0.577` 的 CI95 下界 > 0**；否则诚实改述为"范围外追平/逼近 oracle"。op/sparse2 同样多种子。
- **D2 创新 A 有效**：per-node 分解 A/B vs 全局标量（单变量），N=24 多种子 cond 显著上升**或** val 震荡幅度下降；**I6 守恒单测通过**（sum-of-attributions==全局闭式 tail，1e-9）。
- **D3 创新 B 名副其实**：共识对偶 lam_c 在训练中真实活跃且收敛；硬场景 feasibility 不低于创新 A 基线；reward 结构仍为"一势+两对偶"（I5 单测）。
- **D4 创新 C**：单次训练产出 ≥5 个 ω 点的 Pareto 前沿（reliability vs energy），β>0 时能耗较 β=0 显著下降而 cond 不破 τ。
- **D5 合规闸**：每轮自检 I1/I3/I4/I5/I6 未被降级；任何借鉴的中心化部件已改造为去中心或被丢弃。
- **D6 仓库**：单一主干，README/AGENTS 更新，`tests/unit` 绿、契约套件随 doc 恢复转绿。

> 每个改进单变量落地、记 log、keep/rollback；卡 3–5 轮或只有违反 INVARIANT 才能达标 → 停下汇报。
