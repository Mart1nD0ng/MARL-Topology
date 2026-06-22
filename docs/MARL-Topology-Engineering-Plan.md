# MARL-Topology 全面重构工程计划书

> 文档类型：工程实施计划  
> 项目：`Mart1nD0ng/MARL-Topology`  
> 日期：2026-06-22  
> 目标：将当前单步 critic-free REINFORCE 主干逐步重构为 **CTDE Graph-Counterfactual PPO + SCQ + 去中心化执行**，同时修复环境数学、可解性、能耗和时延定义。

---

## 1. 工程原则

1. **先修环境数学，再比较 MARL。**
2. **中央 critic 可以存在，中央部署 decoder 不可以存在。**
3. 保持一个生产主干，不建立长期并行的第二套系统。
4. 每轮只验证一个主要假设。
5. 所有机制必须有“激活契约”，防止配置使机制名存实亡。
6. smoke、pilot、research 配置严格隔离。
7. 旧结果在协议或指标变更后必须标记不可比较。
8. test/held 数据不得参与 checkpoint 选择或训练期 witness 更新。
9. 大尺度搜索失败只能标记 unknown。
10. 正确性和可复现性优先于旧 headline。

---

# 2. 目标交付物

最终生产主干应包含：

- 动态两时间尺度 V2X Dec-POMDP；
- 安全 PBFT \(n/f/q\)；
- fixed Byzantine fault-set robustness；
- 单一、无重复的 relay 语义；
- tri-state solvability；
- witness memory 与可行性上下界；
- phase-specific PBFT accounting；
- quorum-completion latency；
- 完整能耗与重配置成本；
- local recurrent directional PNA actor；
- local mutual-acceptance decoder；
- centralized graph-temporal vector V/Q critic；
- Graph-Counterfactual PPO；
- SCQ top-M exact counterfactual supervision；
- chance/CVaR reliability constraint；
- preference-conditioned Pareto policy；
- constrained Pareto checkpoint archive；
- 多 seed、跨 N、跨配置评估。

---

# 3. 代码治理

## 3.1 单一生产入口

保留一个正式训练入口，例如：

```text
scripts/train/train_ctde_scq.py
```

迁移完成后，将其设为唯一生产入口。

旧入口：

```text
scripts/train/train_decentralized_rl.py
```

在过渡期仅作为 baseline 或兼容包装，最终必须：

- 明确标记 baseline；
- 不与生产主干竞争；
- 不继续承载旧的误导性 invariant。

---

## 3.2 推荐目录

```text
src/marl_topology/
    protocol/
        quorum_spec.py
        torch_quorum_tail.py
        fault_set_robustness.py
        pbft_message_plan.py
        quorum_completion_latency.py

    solvability/
        status.py
        witness_memory.py
        optimistic_bound.py
        exact_small_graph.py

    models/
        recurrent_directional_pna_actor.py
        centralized_graph_temporal_critic.py
        distributional_cost_head.py

    policies/
        local_bid_sampler.py
        decentralized_mutual_acceptance.py

    training/
        ctde_rollout.py
        graph_mappo.py
        graph_counterfactual_ppo.py
        scq_counterfactual.py
        constrained_dual.py
        pareto_archive.py
        mechanism_contracts.py

    evaluation/
        risk_metrics.py
        solvability_metrics.py
        constrained_pareto.py
        temporal_value_test.py
```

模块名称可根据当前代码风格调整，但职责必须保持清晰。

---

# 4. 总体阶段图

```text
Phase 0  冻结现状与文档治理
Phase 1  PBFT quorum 和 fault semantics
Phase 2  Route / relay 修复
Phase 3  Tri-state solvability
Phase 4  PBFT accounting 与非退化 latency
Phase 5  动态两时间尺度环境
Phase 6  去中心化 actor action API
Phase 7  Graph-MAPPO baseline
Phase 8  Graph-Counterfactual PPO
Phase 9  SCQ exact counterfactual supervision
Phase 10 可靠性约束与 Pareto
Phase 11 Recurrent Directional PNA actor
Phase 12 正式泛化 campaign
Phase 13 文档、迁移和发布
```

P0–P4 未完成前，不对模型创新做最终有效性判断。

---

# 5. Phase 0：冻结现状与治理

## 目标

- 保存当前 HEAD 和现有结果；
- 确认实际生产入口；
- 记录当前测试状态；
- 修正文档中的错误架构边界；
- 建立后续实验治理。

## 工作项

1. 记录：
   - git commit；
   - Python/Torch 版本；
   - 数据 manifest；
   - 当前 smoke；
   - 当前 unit/contract 结果；
   - 当前主结果。
2. 创建：
   ```text
   docs/CURRENT_HEAD_STATUS.md
   ```
3. 更新 AGENTS.md：
   - 删除“无 critic 是不可降级硬约束”；
   - 改成“部署期严格去中心化”；
   - 明确 central critic 合法、central decoder 非法。
4. 增加配置层级：
   ```text
   configs/smoke/
   configs/pilot/
   configs/research/
   ```
5. 增加统一 run manifest。

## 测试

- 当前 smoke；
- 当前 unit；
- 配置解析；
- manifest schema。

## 退出条件

- 所有旧结果可追溯；
- 文档不再混淆 CTDE 与 centralized execution；
- smoke 参数不能被研究脚本默认采用。

---

# 6. Phase 1：PBFT quorum 与故障语义

## 目标

修复跨尺度 PBFT 安全性。

## 工作项

1. 实现 `PBFTQuorumSpec`：
   - `classic_exact`；
   - `safe_generalized`。
2. 检查：
   \[
   n\geq3f+1
   \]
   \[
   2q-n>f
   \]
   \[
   q\leq n-f.
   \]
3. 移除任意 N 下固定 \(f=1,q=3\)。
4. 实现 fixed fault set：
   \[
   C_{\mathrm{robust}}
   =
   \min_{|B|\leq f}C(B).
   \]
5. 实现 Torch quorum-tail。
6. 保留 reference float DP。

## 测试

- quorum intersection property tests；
- liveness property tests；
- classic PBFT fixtures；
- generalized fixtures；
- fixed fault set 跨阶段一致；
- Torch/reference 随机 parity；
- gradcheck；
- small-N exhaustive parity。

## 退出条件

- 所有生产配置满足安全性；
- 无动态“删除当前最大概率”替代固定故障集合；
- 旧数据标记为 protocol-incompatible。

---

# 7. Phase 2：Route / Relay 修复

## 目标

消除重复多跳语义。

## 工作项

1. 画出当前调用链；
2. 选择：
   - 一跳矩阵 + relay DP；
   - 或 route evaluator + max_hops；
3. 删除另一层；
4. 明确 hop latency；
5. 明确 relay energy；
6. 修复 deadline 传播。

## 测试

```text
A -- B -- C
H=1: A->C = 0
H=2: A->C > 0
```

以及：

- direct path；
- disconnected path；
- multiple path max-product；
- schedule deadline；
- energy sum；
- no double counting。

## 退出条件

- hop 参数语义可被单测证明；
- route 和 message matrix 不重复组合多跳。

---

# 8. Phase 3：Tri-state solvability

## 目标

替代错误的二值 `feasible_exists` 真值。

## 工作项

1. 引入：
   ```text
   witness_feasible
   certified_infeasible
   unknown
   ```
2. 实现 train-only witness memory；
3. 实现 small-graph exact solver；
4. 设计严格乐观 upper bound；
5. 旧 searcher 输出改名为 witness；
6. 删除默认 skip `feasible_exists=False`；
7. unknown 使用单独 curriculum，不直接当 constraint violation 真值。

## 数据字段

```text
solvability_status
witness_source
witness_topology
witness_reliability
upper_bound
certificate_source
```

## 测试

- witness 单调；
- 搜索失败保持 unknown；
- exact small graph truth；
- valid upper-bound certificate；
- train/val/test memory 隔离；
- policy discovery 更新 LB。

## 退出条件

- 训练不再依赖伪不可解标签；
- conditional 指标不再称为“provably solvable”。

---

# 9. Phase 4：PBFT Accounting

## 目标

使 energy 和 latency 对真实 PBFT 拓扑有辨识度。

## 工作项

1. 实现 phase-specific message plan；
2. 区分 validators 与 clients；
3. clients 只作为 relay；
4. 计算：
   - protocol；
   - relay；
   - retransmission；
   - MAC control；
   - policy communication；
   - view change；
5. 实现 quorum-completion latency；
6. 失败支付 timeout；
7. 增加 expected/P50/P95/CVaR。

## 测试

- pre-prepare 消息数；
- prepare/commit 消息数；
- client 无 vote；
- relay energy；
- timeout；
- quorum latency；
- topology sensitivity；
- phase budget。

## 退出条件

- latency 不再集中在 0.03 s；
- 不可达拓扑不再获得零时延优势；
- energy 不再是三次重复 all-pairs。

---

# 10. Phase 5：动态两时间尺度环境

## 目标

建立真实的时序决策能力，并科学验证是否需要 temporal policy。

## 工作项

1. 定义：
   - topology control interval；
   - PBFT micro-steps；
   - episode length；
2. 添加：
   - 位置和速度演化；
   - 信道演化；
   - previous topology；
   - queue/battery 可选状态；
   - reconfiguration energy/latency；
3. 实现 simulator state fork；
4. 保留 static contextual-bandit mode；
5. 实现 temporal value test。

## 测试

- deterministic trajectory；
- same seed reproducibility；
- action affects future cost；
- topology persists；
- reconfiguration cost；
- state fork equivalence；
- \(T=1\) 与旧单步语义对齐。

## 决策门

若：

\[
\Delta_H
\]

在多场景、多 seed 下接近零，则保留单步主任务。

若显著大于零，则启用 temporal actor/critic。

## 退出条件

- 是否需要时序由实验决定；
- temporal 机制不再在无动作—未来耦合的 toy 环境中测试。

---

# 11. Phase 6：Actor Action API 与 Decoder

## 目标

为 MAPPO/COMA/SCQ 提供真正 per-agent 动作。

## 工作项

1. actor 输出 directed bids；
2. sampler 返回：
   - `action_i`；
   - `logp_i`；
   - `entropy_i`；
   - proposal diagnostics；
3. 强制主 rollout 使用 local mutual decoder；
4. global argsort 只留 ablation；
5. 分离 action graph 与 communication graph；
6. 实现 batch 前向与逐节点消息交换等价测试。

## 测试

- per-agent logp sum；
- budget feasibility；
- mutual symmetry；
- no global sort；
- local observability boundary；
- batch/distributed equivalence；
- action entropy 与真实 policy 匹配。

## 退出条件

- centralized decoder 无法进入生产 rollout；
- 部署路径可逐节点执行。

---

# 12. Phase 7：Graph-MAPPO Baseline

## 目标

建立第一个合法 CTDE 强基线。

## 工作项

1. centralized graph-temporal value critic；
2. vector heads：
   - energy；
   - latency；
   - reliability cost；
3. PPO clipped update；
4. GAE 或 n-step return；
5. critic 与 actor encoder 完全分离；
6. local decoder 参与全部 rollout。

## 对照

- REINFORCE-EMA；
- RLOO \(M\geq2\)；
- Graph-MAPPO。

## 指标

- sample efficiency；
- evaluator calls；
- critic explained variance；
- PPO KL；
- OOD N critic error；
- reliability/energy/latency。

## 退出条件

- CTDE 不改变部署图；
- critic 在 validation 和 OOD 上具有可接受误差；
- 公平 evaluator-call budget A/B 完成。

---

# 13. Phase 8：Graph-Counterfactual PPO

## 目标

从全局 advantage 升级到 per-agent credit。

## 工作项

1. action-conditioned vector Q critic；
2. actor-sampled independent counterfactuals；
3. COMA-style baseline；
4. per-agent PPO advantage；
5. 记录 counterfactual rank correlation。

## 重要约束

用于 policy baseline 的反事实样本必须独立于实际动作，避免 action-dependent baseline 偏差。

## 测试

- counterfactual action only changes one agent；
- other actions fixed；
- decoder recomputation correct；
- per-agent advantages differ；
- no gradient through baseline sample selection；
- unbiasedness toy test。

## 对照

- Graph-MAPPO；
- Graph-Counterfactual PPO。

## 退出条件

- credit variance下降或 sample efficiency提升；
- 无隐藏 centralized decoder。

---

# 14. Phase 9：SCQ Exact Counterfactual Supervision

## 目标

用真实 PBFT evaluator 差分校准 Q critic。

## 工作项

1. quorum sensitivity；
2. top-M add/remove/swap；
3. simulator fork；
4. exact immediate/one-step difference；
5. SCQ critic consistency loss；
6. witness discovery；
7. evaluator-call budget control。

## 测试

- sensitivity 与有限差分；
- top-M 可复现；
- exact delta；
- fork state isolation；
- SCQ loss 真正参与 critic 更新；
- held/test 无反馈。

## 对照

- Counterfactual PPO；
- Counterfactual PPO + random counterfactual；
- Counterfactual PPO + SCQ top-M。

## 退出条件

- critic difference error下降；
- witness discovery或策略效果改善；
- 增益大于额外 evaluator 成本。

---

# 15. Phase 10：可靠性约束与 Pareto

## 目标

从手调 weighted reward 转为正式 constrained multi-objective RL。

## 工作项

1. chance constraint；
2. CVaR 可选；
3. dual residual：
   \[
   \Pr(C<\tau)-\delta;
   \]
4. preference-conditioned actor；
5. vector critic；
6. 固定物理 normalization；
7. Pareto validation archive；
8. constrained hypervolume。

## 测试

- dual 可升可降；
- constraint target；
- \(\omega\) 条件影响动作；
- extreme preferences；
- non-dominated archive；
- no teacher normalization；
- validation-only selection。

## 退出条件

- energy 和 latency 都真实进入训练；
- checkpoint 不再只按 raw feasibility；
- 单模型输出非平凡 Pareto 点。

---

# 16. Phase 11：Recurrent Directional PNA Actor

## 目标

在环境和训练机制稳定后升级 actor。

## 工作项

1. local temporal GRU；
2. directional messages；
3. PNA aggregators；
4. degree scalers；
5. recurrent shared rounds；
6. randomized K；
7. control communication cost；
8. directed bid head。

## 公平对照

相同：

- critic；
- decoder；
-数据；
- objective；
- seed；
- evaluator calls；
-训练步数。

比较：

1. 当前 sum GNN；
2. mean/normalized GNN；
3. directional PNA；
4. recurrent directional PNA。

## 退出条件

- OOD N 稳定性提升；
- 通信成本被显式报告；
- batch/distributed execution parity。

---

# 17. Phase 12：正式泛化 Campaign

## 数据维度

训练：

\[
N\in\{8,12,16\}.
\]

评估：

- held in-range；
- N=24；
- N=32/48；
- 不同 power；
- 不同 RSU count；
- 不同 density；
- 不同 blockage；
- 不同 \(f/n\)；
- 不同 topology hold interval；
- 不同 preference。

## 统计

- ≥5 independent seeds；
- paired seed/split；
- raw per-seed；
- mean；
- std；
- 95% CI；
- paired bootstrap；
- effect size。

## 主要报告

- reliability violation；
- CVaR shortfall；
- energy；
- expected/P95 latency；
- reconfiguration；
- policy communication；
- constrained hypervolume；
- witness discovery；
- critic OOD error；
- train–deploy gap。

## 退出条件

- 所有 headline 有可复核 artifacts；
- 单 seed 结果不进入正式结论。

---

# 18. Phase 13：迁移与发布

## 工作项

1. 将新入口设为唯一主干；
2. 旧 REINFORCE 保留为 baseline；
3. 更新：
   - README；
   - AGENTS；
   - TRUNK_MAP；
   - research log；
4. 标记旧结果：
   ```text
   retired_due_to_protocol_metric_change
   ```
5. 清理 scratch；
6. 固化 dependency versions；
7. 完整 reproduction command。

## 退出条件

- 文档、代码和工件一致；
- 不存在第二生产 trunk；
- fresh checkout 可复现 smoke 和主要结果。

---

# 19. 每轮迭代工作流

每轮只能有一个主要假设。

## 19.1 开始前

写 `experiment_plan.md`：

```text
Hypothesis
Changed variable
Controlled variables
Success criterion
Failure criterion
Compute budget
Seeds
Splits
Required tests
```

## 19.2 实施

1. 先写失败测试或诊断；
2. 最小实现；
3. targeted tests；
4. smoke；
5. 受影响 unit suite；
6. 机制激活检查；
7. pilot；
8. 通过后 multi-seed research A/B。

## 19.3 结束

写 `decision.md`：

```text
Result
Per-seed values
Confidence interval
Failure cases
Mechanism activation evidence
Compute cost
Keep / Revise / Rollback
Next single hypothesis
```

连续三轮无收益时，优先复查：

- evaluator；
- label；
- action reachability；
- train/deploy mismatch；
- critic error；
- checkpoint selection；
- gradient variance；
- mechanism activation。

不能直接堆叠下一模块。

---

# 20. 机制激活防线

## RLOO

声明 RLOO 时：

\[
M\geq2.
\]

否则 fail fast。

## Temporal

声明 temporal 时：

- \(T>1\)；
- state transition 非零；
- previous topology 有效；
- action affects future return。

## COMA

声明 counterfactual 时：

- \(K_{\mathrm{cf}}\geq1\)；
- per-agent advantage 非全相同；
- counterfactual actions 与真实动作有差异。

## SCQ

声明 SCQ 时：

- top-M \(>0\)；
- exact evaluator delta 被计算；
- SCQ loss 非零；
- witness 更新日志存在。

## Dual

声明 constraint 时：

- target 非空；
- residual 有正有负；
- \(\lambda\) 可升可降。

## Pareto

声明 Pareto 时：

- 至少两个偏好；
- archive 非空；
- hypervolume 被计算；
- checkpoint 选择使用 archive。

---

# 21. 公平实验预算

不同算法必须在至少一种统一预算下比较：

## Evaluator-call budget

\[
B_{\mathrm{eval}}
\]

相同。

适用于比较 RLOO、SCQ 和搜索型方法。

## Environment-step budget

\[
B_{\mathrm{step}}
\]

相同。

## Wall-clock budget

额外报告，但不替代 evaluator-call 公平性。

## Parameter budget

actor 保持相同；critic 参数单独报告。

不能以相同 update 数比较每次 update 调用 evaluator 次数完全不同的方法。

---

# 22. 实验工件

每个正式 run 保存：

```text
config.yaml
manifest.json
git_revision.txt
environment_version.json
data_manifest.json
seed_manifest.json
train_history.jsonl
critic_metrics.json
dual_history.json
validation_metrics.json
held_metrics.json
pareto_archive.json
solvability_report.json
failure_cases.json
mechanism_activation.json
decision.md
```

---

# 23. 风险登记

## 风险 1：Critic 使用全局信息过强

缓解：

- joint-observation-history critic；
- true-state critic ablation；
- OOD N critic error；
- critic dropout；
- graph-size randomization。

## 风险 2：SCQ evaluator 成本过高

缓解：

- top-M；
- near-threshold only；
- vectorized evaluator；
- cache；
- schedule；
- random control。

## 风险 3：Temporal 环境仍无时序价值

缓解：

- Temporal Value Test；
- 保留 bandit baseline；
- 不强行启用 GRU。

## 风险 4：Pareto 条件未真正生效

缓解：

- extreme preference tests；
- preference-action mutual information；
- archive coverage。

## 风险 5：隐藏中央 decoder

缓解：

- production imports 禁止 global decoder；
- boundary tests；
- deployment replay；
- distributed-equivalence test。

## 风险 6：机制名存实亡

缓解：

- mechanism activation assertions；
- smoke/pilot/research 配置分离；
- formal run manifest。

---

# 24. Definition of Done

只有同时满足以下条件才能宣布重构完成：

1. PBFT quorum 和 fault semantics 通过属性测试；
2. relay 无重复多跳；
3. tri-state solvability 替代伪二值真值；
4. phase-specific accounting 正确；
5. latency 非退化且失败支付 timeout；
6. 动态任务是否必要已由 Temporal Value Test 决定；
7. 主 rollout 只使用 local mutual decoder；
8. centralized critic 不进入部署；
9. Graph-MAPPO、Counterfactual PPO、SCQ 有公平消融；
10. RLOO 正式基线 \(M\geq2\)；
11. reliability constraint 可升可降；
12. energy 与 latency 均进入训练；
13. Pareto archive 工作；
14. actor 支持变 N 和局部通信；
15. 控制通信与重配置成本进入指标；
16. ≥5 seed OOD 结果有 CI；
17. README、AGENTS、TRUNK_MAP 与代码一致；
18. fresh checkout 可复现。

---

# 25. 推荐最小落地路径

在算力和时间有限时，优先完成：

```text
1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 6
7. Phase 7
8. Phase 8
9. Phase 9
10. Phase 10
```

随后再决定：

- Phase 5 时序环境是否开启；
- Phase 11 actor 是否升级。

不能先做大网络，再回头修 evaluator。
