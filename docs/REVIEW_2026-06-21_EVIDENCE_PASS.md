# 项目审查报告 —— 证据驱动 pass（2026-06-21）

> 方法：以工件为准，不以 log 叙述为准。每条结论指向具体 `文件:行` 或 `result_save/*.json` 字段。
> 证据基座：通读 `literature review/` 全部 40 篇文献 + 6 项 INVARIANT 源码核对 + 3 项实验工件核对（52 个子代理，约 2.6M token，873s）。
> 关联记忆：`campaign-4090-audit`、`decentralized-rl-trunk-state`、`marl-topology-gate-constraints`。

---

## 净判定（先给结论）

- **真正完成的**：一个 INVARIANT 大体合规的**去中心化 constrained-RL 主干**（无 critic、闭式共识、单目标势函数奖励、τ=0.9 硬冻结），代码层面经得起核对。最强工件是 **N=24 冷启动、无 oracle、击败 SA 教师**——已对 JSON 逐字段验证为真。
- **没完成 / 被高估的**：
  1. 该 N=24 头条结果 **单种子、单切分、held 仅 26 场景**，统计功效不足（最大短板）。
  2. **INVARIANT #2（时序）按字面未满足**——主干无任何时序特征，且"证明时序为 null"的 GRU 模块已从代码树删除，null 已不可复现。
  3. 工作树 **182 个 docs 被删（未暂存）→ 契约测试 257 失败/162 通过**。
  4. 4090 campaign 的"raw 0.82 / planner 增益 / actor≥SA"等多项**与工件矛盾**。

---

## (a) Loop 完成判定：声称 → 状态 → 证据

| # | INVARIANT / 声称 | 状态 | 证据 |
|---|---|---|---|
| #6 | 共识可靠性=**闭式全局**共识失败概率，非 MC/经验/局部代理 | ✅ 已验证 | `protocol/quorum_tail.py:33-73` 异构 Poisson-binomial 生成多项式 DP；grep `random/monte/trials/np.random` 两文件**零命中**；全局量 over 所有 node（`pbft_reliability.py:246-250`）；既是训练奖励(`train_decentralized_rl.py:142,147,152`)又是评估指标(`metrics/registry.py:33-42`)。瑕疵：registry 文案仍写"Stage 2 deterministic estimate"（陈旧误导，非违规） |
| #1 | 完全去中心化**执行+学习**（无全局泄漏、无 critic 假装去中心化） | ✅ 已验证（带保留） | 执行仅消费局部特征，`message_passing_graph_edge_scorer.py:103` `global_topology_used=False`；解码 `local_mutual_assemble`；学习 `loss=-(adv.detach()*logp).mean()`(`train_decentralized_rl.py:427`)+RLOO(411)，**无 value 网络**。保留：旧 MAPPO/critic 主干 `production_mappo_adapter.py:652` **未删、仍被本脚本 import（仅造数据）**，README/AGENTS 仍称其 canonical |
| #2 | 模型必须**真正使用时序**特征 | ❌ 违反（按字面）→ 经 owner 决定**退役** | `grep nn.GRU/LSTM/RNN` over `src/` **零命中**；log 引用的 `models/local_temporal_gnn_edge_scorer.py` **代码树不存在（已删）**；`train_decentralized_rl.py:31` "Temporal stays OFF"。证明 null 的 GRU 已删→null 不可复现。**owner 2026-06-21 裁决：经测试时序贡献不存在，#2 从不可降级清单移除** |
| #3 | τ≥0.9，从不放宽 | ✅ 已验证 | 单源 `decentralized_distillation.py:34 TAU=0.9`，奖励与评估同一常量导入；硬护栏 `requirement_feasibility_diagnosis.py:395 raise if <0.9`。瑕疵：0.9 字面在 ~10 模块重复（去同步风险），主干路径用同一 TAU |
| #5 | 单一原则化奖励（势函数+Lagrangian），非堆叠加权 | ✅ 已验证（带保留） | `train_..:152` dense `r=(c-TAU)-lam_b*g_b-(beta*er if feasible)`，决定性 run `--beta 0`→`r=(c-0.9)-lam_b*g_b`。Ng-Harada 势 + 预算对偶上升(434) + solvable-only mask(391)。保留：**dense 下 lam_c 计算但不进奖励**→共识全靠势函数，**只有预算对偶是活的**，"共识+预算双对偶"言过其实 |
| #4 | 泛化：域随机化+held-out+变 N+**多种子带方差** | 🟡 部分 | held-out ✅、多配置 ✅、变 N ✅(8/12/16/20/24)；**多种子 ❌**：所有 `dec_rl*/rl_result.json` 全 `seed:0`，头条 N=24 单种子。campaign CI95 是 5 个 actor checkpoint 复用同数据/切分的方差，非独立训练种子 |

### 头条结果工件核对（逐字段 VERIFIED）
- **N=24**（`dec_rl_scale24/rl_result.json`）：held=26（solvable=15），seed=0，raw 0.769(keep-best)/0.808(final)，cond 0.933，ceiling 0.577，energy 0.754/0.778，与 `run_n24_cold.log:55-56` 逐行一致。✅ 真，**单种子**。
- **op 冷启动**（`dec_rl_cold`）：raw 0.688(==ceiling)，cond 0.894(keep-best)/**0.909(final)**。"0.909"是末步策略；"~97% of BC"里的 BC 来自 log 而非本 JSON。
- **sparse2 "RL>BC"**：🟡 PARTIAL — 仅 conditional(0.864→0.886,+0.022)且**仅末步策略**；**keep-best 回退到 BC**，raw 几乎不动(0.448→0.458)。i12 "首次 RL>BC" 成立但**边际且选择脆弱**。

### 4090 campaign 与计划的矛盾
- ❌ "raw≈0.80-0.82"：E1 实测 **0.6625±0.079**，0.82 被驳倒。
- ❌ "actor≥SA-teacher"：actor raw 0.6625 **<** SA 0.6875；actor 只赢全图/空图/随机(均 0.0)。
- ❌ "planner 比 SA +0.05-0.06"：E8 vs E1 实际 **+0.006**。
- 🟡 Route B 半成立：rich26(高功率)raw 0.94 ✅，rich6r(密 RSU)raw 0.79 ❌。
- ✅ 去中心化成本≈0(0.004)；E6 zero-shot 退化(0.49)、E7 OOD(0.68)/retrain(0.72)。⚠️ E6/E7 用 `split=all`(240) **非 held-out**。

### 仓库/测试健康：🔴 工作树红
- **182 个 docs/\*.md 删除（未暂存，可 `git restore docs/` 恢复）** → `tests/contract` **257 失败/162 通过**（纯因 doc 缺失，如 `test_project_contract_files.py` FileNotFound `PROJECT_STATE.md`）。
- `tests/unit` **496 通过/1 失败/1 xfail**，唯一失败 `test_stage24_critic_integration` 是与本次无关的陈旧断言（干净 checkout 也失败）。
- run 目录 gitignore 正确；src/tests 对 HEAD 干净——删 doc 是唯一改动。

---

## (b) 项目真实状态 + 排序后的真实瓶颈

**真实图景**：物理环境真实合规（3GPP TR 37.885/38.901 + scheduled MAC + relay + coverage-gated PBFT，已接入主干评估器，#6 闭式指标驱动奖励）；模型主干是**静态 K-round 消息传递 GNN + 局部互接受解码器**，去中心化执行/学习成立；最强证据 N=24 击败 oracle 真实但单种子。

**按严重度排序的真实瓶颈**：
1. **🔴 统计功效（头条单种子）** —— held 仅 26 场景(15 可解)，一场景翻转≈0.04-0.05 raw；val 轨迹震荡 0.58-0.71(best 0.711)，而报告 0.769/0.808 高于该轨迹 → 超出 0.577 的边际**可能被采样噪声解释**。评审致命点，但修复最廉价。
2. **🟠 时序 #2 已退役**（owner 裁决）——更新硬约束清单。
3. **🟠 in-range 价值脆弱** —— sparse2 的 RL>BC 只在末步策略、只在 conditional、keep-best 回退；op 只追平 ceiling。RL"价值"目前**只在 out-of-range(N=24)成立**。
4. **🟡 奖励叙述 vs 实现的缺口** —— dense run 中 lam_c 死项；"共识+预算双对偶"言过其实。
5. **🟡 主干二义性 + 仓库红** —— MAPPO/critic 主干与 RL 主干并存未退役；README 指旧主干；182 doc 删除使契约套件红。
6. **🟢 信用分配粗糙** —— 全局 quorum-tail 当前作单一场景标量共享给所有节点，未做 per-node 分解（懒节点塌缩风险）。

---

## (c) 文献结构化分析 + 创新性定位 + 改进方向

### 40 篇笔记（按主题；"中心化?"决定能否照搬而不破 #1）

**① NCO**：1 MACSIM(去中心,置换不变 set-CE)、2 PARCO(**中心**,agent-only 通信层)、3 MAPT(**中心**+critic)、4 LNS2+RL(CTDE,**8-agent 固定邻域窗口**)、5 Satellite(去中心,**Q-矩阵+Hungarian**)、6 Equity-Transformer(去中心,**REINFORCE+共享 MC 基线=我们 RLOO 的 prior art**)、7 SCRIMP(去中心,可扩展通信编码器)。

**② 拓扑规划**：8 MARL2GRID-TR(CTDE,**idle 门控**)、9 Centrally-Coord(**中心**,优先级课程)、10 Mesh(去中心)、11 **MAGNNETO(CTDE,邻域-only MPNN=我们 actor 的 prior art)**、12 Grid(CTDE,分层动作)。

**③ V2X**：13 Wang-benchmark(去中心,**拓扑多样课程**)、15 优先级(CTDE)、17 AoI(去中心,多任务局部 critic)、18 Federated(去中心,**SUMO+NR-V2X Mode2 管线**)、19 Mean-Field、20 两分支优先切换。

**④ ITS/CAV**：21 factor-graph(CTDE)、22 Han et al.(去中心,**截断 Q+γ^(κ/2) 衰减界**)、23 社交注意力、24 CoPO(LCF)、25 **CoLight(去中心,index-free GAT)**、26 **MPLight(去中心,max-pressure 闭式局部奖励)**、27 **MACPO(中心,dense+sparse 成本分解)**。

**⑤ 无线 RRM**：28 离线 CQL、**29 Multi-cell(high,去中心,局部邻域奖励)**、30 可学习边权 GNN、31 RSSI-only、32 top-K 固定维观测、33 SINR 零奖励硬约束、34 干扰外部性定价、35 D2LT 特征。

**⑥ MO-MARL**：36 效用异质(CTDE,无可借)、37 PCMA(局部观测偏好)、38 **MOMA-AC(actor 侧 ω 采样)**、39 MOMAland(权重分解 Pareto)、40 MO-MIX(偏好条件局部 Q)。

### 创新性定位（诚实，不注水）

**真正新颖（窄但成立）**：
- **C1 闭式 PBFT quorum-tail 作为 RL 奖励** —— 40 篇无一优化共识可靠性目标（全是 sum-rate/delay/AoI/PRR/pressure）。**目标/领域贡献**。
- **C6 `local_mutual_assemble` + C3 门界高斯探索** —— 预算可行-by-construction 双端 AND 规则，**无 solver/无全局矩阵/无全局排序**，零噪声极限**逐字节等于部署**。REDA/PARCO 都需 solver 或中心协调。**语料中不存在**——干净的模型贡献。
- **C5 经验姿态** —— 冷启动去中心化学习者在**训练范围外 N=24 超越中心 oracle**(0.769-0.808 vs 0.577)。对照论文都是"逼近/追平"中心参考。**全项目最强**，但单种子。

**标准/已有（非创新）**：REINFORCE+RLOO 无 critic（Equity-Transformer/PARCO/MACSIM）；DTDE 去中心学习（Han/CoLight/MPLight/Zhang&Guo）；Lagrangian+势函数 shaping（MACPO/LagrMAPPO）；GNN 尺度不变（MAGNNETO/CoLight/LNS2+RL）；3GPP V2X env（Saad/Wang-benchmark）；结构化可行解（REDA/PARCO）。

> **博士级判断**：把论文叙事锚定在"指标+奖励+结果"三元组(C1/C6/C5)，把 C2 学习规则定位为"刻意选用最简 critic-free 估计器(零 CTDE gap)"而非算法创新。最大暴露面：MACPO（dense 势+硬约束已发表）、Zhang&Guo（奖励还没 per-node 分解）。

### 改进方向（收益×成本 排序，每条绑 INVARIANT）

| 优先级 | 方向 | 来源 | 收益 | 成本 | INVARIANT 约束 |
|---|---|---|---|---|---|
| **1** | **N=24(+op/sparse2)≥3 独立种子+CI95** | V2X RRA/MOMAland 规范 | 最高——把头条从单种子变带方差可辩护结果 | 低-中 | 无风险，强化 #4 |
| **2** | **全局 quorum-tail 的 per-node 精确分解(leave-one-out/Shapley)作 per-agent 奖励** | Zhang&Guo, Xu et al. | 锐化信用分配、降方差→抬升/稳定 N=24 | 中 | #6 保留全局闭式作评估/约束，只用精确 leave-one-out（sum 重构全局）；#5 每节点一势 |
| **3** | **让共识对偶变活：MACPO dense/sparse 分解**，lam_c 接稀疏 episode 级共识违约成本 | MACPO | 让"共识+预算双对偶"名副其实 | 低-中 | #5 一势+两结构不同对偶，非加权袋；#1 绝不借中心 cost-critic/trust-region |
| **4** | **固定 k-邻域窗口 decoder/actor** | LNS2+RL, Naderializadeh | 给 N 外推架构级论证，封顶算力 | 中 | #1 top-k 只用局部链路特征，不窥全局排序 |
| **5** | **REDA 去中心拍卖作更强可行解码器** | REDA | 更可证 conflict-free，硬场景抬 conditional | 中-高 | #1 用去中心市场/竞价变体，非中心矩阵编译 |
| **6** | **偏好条件 ω actor 描 reliability-vs-budget Pareto 前沿**（顺带 β>0 命中低能耗 DoD） | MOMA-AC/MO-MIX/MOMAland | 单训练给部署菜单，强化 #4 广度 | 中 | #1 只借 actor 侧 ω 采样，丢 CTDE critic；#5 ω 标量化一势+对偶 |

---

## (d) 下一轮 loop 计划

**优先：方向 1（多种子+CI），其次方向 2（per-node 精确奖励分解）。**

**为什么**：方向 1 是头条结果的承重修复且最廉价——不先坐实它，任何模型改进都无法判断是真信号还是噪声。方向 2 是最高上限的模型侧改动，直接打瓶颈 6 与瓶颈 1，且能在不破 #6 的前提下（精确 leave-one-out 而非局部代理）抬升/稳定 N=24。

**DoD**：
1. N=24/op/sparse2 各 ≥3 独立训练种子（不同 `--seed` 且不同 `--split-seed`），报告 `mean±CI95`；**判据**：N=24 的 `raw_mean − 0.577` 的 CI95 下界 **>0** 才算"击败 oracle"，否则诚实改述为"以 critic-free 去中心冷启动在范围外追平/逼近 oracle"。
2. per-node 分解 A/B vs 全局标量：N=24 多种子 conditional 显著上升**或** val 震荡幅度下降，且**全局闭式 tail 评估不变**（#6 守恒，sum-of-attributions==global 单测）。
3. 每轮记 `URBAN_V2X_RESEARCH_LOG.md`，单变量 keep/rollback。

**不可降级清单（2026-06-21 更新，#2 时序已退役）**：#1 去中心执行+学习 · #3 τ≥0.9 · #4 泛化+多种子 · #5 单一原则化奖励 · #6 闭式全局共识失败概率。方向 2/3/5/6 都触碰 #1 或 #6，照搬任何中心 critic/全局矩阵/局部代理即违规。
