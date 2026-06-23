# MARL-Topology 全面重构工程计划书（第二版）

> 文档类型：工程实施计划  
> 项目：`Mart1nD0ng/MARL-Topology`  
> 初版日期：2026-06-22  
> 第二版修订：2026-06-23  
> 当前阻塞提交语境：Phase 7 Graph-MAPPO WIP  
> 目标：**先修复并重新验收 Phase 0–7，再继续 Phase 8–13。**

---

# 1. 当前状态与动态任务回答

## 1.1 项目是否已经开发“建立动态任务”

**结论：没有完成。**

仓库已经存在：

- 车辆位置随时间推进；
- `TrajectoryFrame` / `ProductionTrajectorySpec`；
- 多帧场景生成；
- 一些历史 temporal actor 研究 substrate。

但当前正式训练和 Phase 7 `graph_mappo.py` 仍明确是：

\[
T=1
\]

的 contextual bandit：

- 没有 topology persistence；
- 没有 previous topology 进入主 rollout；
- 没有 reconfiguration energy/latency；
- 没有 queue/battery 等动作—未来状态耦合；
- 没有 \(s_{t+1}\) rollout；
- 没有 GAE/bootstrap；
- 没有 simulator state fork；
- 没有 Temporal Value Test。

因此，现状只能称为：

> **已开发动态场景/轨迹数据底座，尚未建立动态强化学习任务。**

Phase 5 必须在 Phase 0–7 修复阶段补完，随后用 Temporal Value Test 决定最终主任务是否保持单步。

---

# 2. 第二版工程原则

1. 先修 Phase 0–7，再继续 Phase 8。
2. 不通过 permutation threshold、degree cap 或 budget cap 规避复杂度。
3. agent action 是无序 subset，不是 ordered permutation。
4. central critic 可以存在，central decoder 不可以存在。
5. local mutual acceptance 是唯一生产 rollout/deployment decoder。
6. 每个机制有 runtime activation contract。
7. smoke、pilot、research 配置严格隔离。
8. evaluator-call budget 是算法公平性的首要预算。
9. production-scale smoke 是单元测试的必要补充。
10. 旧结果在协议、动作分布或指标变更后必须标记不可比较。
11. test/held 不参与训练、checkpoint 或 witness 更新。
12. 一个阶段只有在 primitive、integration、production-scale 三层测试均通过后才算完成。

---

# 3. 阶段状态总表

| Phase | 当前状态 | 第二版动作 |
|---|---|---|
| 0 | 治理边界已修，suite 已转绿 | 补 config tiers、manifest、机制激活 artifacts |
| 1 | safe quorum/Torch DP 完成，fixed-B 接入 | 修 effective \(f\)、\(|B|\le f\) 语义、greedy 证据边界 |
| 2 | one-hop relay 已启用 | 移除错误 legacy default、补 latency-aware relay |
| 3 | tri-state 字段完成 | 训练主干改用 tri-state，接入 witness/unknown curriculum |
| 4 | quorum latency primitive 完成 | phase-specific message/energy/latency integration |
| 5 | trajectory substrate 存在 | 真正建立动态任务并运行 Temporal Value Test |
| 6 | per-agent API 完成但 ordered action 错误 | 用 BCSP 无序 subset policy 重写 |
| 7 | Graph-MAPPO WIP，运行挂起 | 修 critic gradient、per-agent ratio、batch/checkpoint、real-shard smoke |
| 8–13 | 未完成 | Phase 0–7 验收后继续 |

---

# 4. 修复闸门 R0–R7

# R0：Phase 0 治理与实验基础设施

## 目标

使每次 run 的代码、环境、参数和机制状态可追溯。

## 工作项

1. 建立：
   ```text
   configs/smoke/
   configs/pilot/
   configs/research/
   ```
2. 统一 run manifest：
   ```text
   git_revision
   environment_math_version
   action_distribution_version
   dataset_manifest
   seed/split
   mechanism_activation
   evaluator_call_budget
   ```
3. 明确：
   - old EMA baseline；
   - RLOO baseline；
   - Graph-MAPPO arm；
   - production candidate。
4. 文档和代码中删除仍把 `c-tau` 无证明称为 policy-invariant shaping 的描述。
5. 正式记录当前 action-policy blocker 和 Phase 6 REVISE 决策。

## 测试

- config schema；
- research config 不得继承 smoke 参数；
- manifest round-trip；
- mechanism activation artifact 存在。

## 退出条件

- fresh run 可完全复现；
- smoke/pilot 结果不能被 headline 脚本读取。

---

# R1：Phase 1 PBFT quorum / fixed-B 修复

## 目标

确保生产 evaluator 实际执行的 \(n,f,q,B\) 与文档一致。

## 工作项

1. 生产评价记录：
   ```text
   validator_count
   configured_f
   effective_f
   q
   q_external
   fault_strategy
   fault_set_count
   exact/approx
   ```
2. 检查 `PhysicsRegime -> Stage21ObjectiveStackConfig` 是否真正传入 fault tolerance。
3. 修复 honest-primary 条件平均下的单调性问题：
   - 显式 view-change + 固定 primary 分布；或
   - 枚举所有 \(|B|\le f\)。
4. greedy strategy 明确标记 optimistic approximation，不作为 certificate。
5. 小 N exact；大 N 记录 approximation gap。
6. 重新核实 N=24 的实际计算复杂度，禁止继续无证据使用 \(\binom{24}{7}\) 解释 production cost。

## 先失败测试

```text
test_production_effective_f_q_are_logged
test_fault_set_min_checks_all_sizes_when_required
test_asymmetric_primary_breaks_naive_monotonicity
test_greedy_is_not_marked_exact_or_certified
```

## 退出条件

- production \(f,q\) 无歧义；
- robust reliability 的 exact/approx 语义诚实；
- 非对称反例通过。

---

# R2：Phase 2 route/relay 完成化

## 目标

将 corrected semantics 变成唯一默认。

## 工作项

1. `one_hop_relay=True` 成为 production/default corrected mode。
2. legacy mode 必须显式请求。
3. 删除或翻转旧 strict xfail。
4. relay probability、latency、energy 同一条路径语义。
5. deadline 沿 relay path 正确传播。
6. vectorized/reference evaluator parity。

## 测试

- A-B-C H=1/2；
- multiple paths；
- scheduled latency；
- energy；
- legacy explicit-only；
- production config integration。

## 退出条件

- 无静默双重多跳路径；
- 无默认错误语义。

---

# R3：Phase 3 tri-state integration

## 目标

让 tri-state 真正影响训练和评价。

## 工作项

1. 删除训练主干：
   ```python
   if not feasible_exists: continue
   ```
2. 使用：
   ```text
   witness_feasible
   certified_infeasible
   unknown
   ```
3. unknown：
   - 进入探索；
   - 不直接当 certified violation；
   - 可由策略更新 witness。
4. train/val/test witness memory 隔离。
5. conditional metric 改名为 witness recall。
6. 实现 optimistic UB 或明确 certified-infeasible 暂不可用。

## 测试

- finite search miss -> unknown；
- unknown 不被 skip；
- policy discovery U->W；
- test witness 不反馈 train；
- old label 不再控制主训练。

## 退出条件

- 训练不再依赖伪不可解标签。

---

# R4：Phase 4 PBFT accounting integration

## 目标

完成真实三阶段消息、能耗和 latency 闭环。

## 工作项

1. `pbft_message_plan.py`：
   - pre-prepare；
   - prepare；
   - commit。
2. validators 产生 vote，clients 只 relay。
3. 不再将同一 `records` 字典复制到三个 phase。
4. quorum-completion latency 使用真实 phase message maps。
5. 失败支付 timeout。
6. energy 包括：
   - protocol；
   - relay；
   - retransmission；
   - MAC control；
   - policy communication；
   - reconfiguration；
   - view-change。
7. 输出 expected/P50/P95/CVaR/timeout rate。

## 先失败测试

```text
test_pre_prepare_has_primary_to_backup_messages_only
test_clients_do_not_emit_pbft_votes
test_prepare_commit_message_plans_are_phase_specific
test_failed_phase_pays_timeout
test_control_and_relay_energy_are_counted_once
```

## 退出条件

- latency/energy 对 topology 有辨识度；
- 三阶段语义真实接入 production evaluator。

---

# R5：Phase 5 动态任务建立与决策

## 目标

从“轨迹数据”升级为真正动作影响未来的环境。

## 工作项

1. 定义 macro topology interval：
   \[
   \Delta T_{\mathrm{topo}}.
   \]
2. 每个 topology 保持多个 PBFT micro-round。
3. 状态加入：
   - previous topology；
   - velocity/channel history；
   - reconfiguration state；
   - 可选 queue/battery。
4. 成本加入：
   \[
   E^{\mathrm{reconfig}},
   \qquad
   L^{\mathrm{reconfig}}.
   \]
5. 实现：
   \[
   (s_t,a_t)\rightarrow(r_t,s_{t+1}).
   \]
6. 实现 simulator state fork。
7. 保留 \(T=1\) static mode。
8. 执行 Temporal Value Test。

## 测试

```text
test_episode_length_gt_one
test_topology_persists_for_hold_interval
test_action_changes_future_cost
test_reconfiguration_cost_nonzero
test_state_fork_isolation
test_T1_matches_static_mode
```

## 决策门

若多场景、多 seed：

\[
\Delta_H\approx0,
\]

则 static bandit 为主任务，dynamic 作为扩展。

若：

\[
\Delta_H>0,
\]

则后续 critic/actor 必须支持 temporal return。

## 退出条件

- 对“是否需要时序”有实验结论，而不是仅有轨迹类。

---

# R6：Phase 6 无序子集 Action API

## 目标

用 BCSP 替换 ordered Plackett–Luce，并消除 factorial 复杂度。

## 工作项

1. 新建 `budget_conditioned_subset.py`。
2. 实现：
   \[
   \pi_i(S)\propto
   \mathbf1[|S|\le b_i]
   e^{\sum_{e\in S}\theta_e}.
   \]
3. 实现 \(O(mb)\)：
   - log partition；
   - exact logp；
   - exact sampling；
   - exact entropy。
4. 实现 \(b\ge m\) 的 independent Bernoulli \(O(m)\) 快路径。
5. API 改成：
   ```text
   accepted_subset
   subset_logp
   normalized_subset_entropy
   budget
   ```
6. order 不进入 probability。
7. local mutual acceptance 保持不变。
8. 所有 tensor device-preserving。

## 先失败测试

```text
test_order_aliases_are_one_subset_action
test_k_equals_m_has_no_order_entropy
test_bcsp_matches_exhaustive_small_graph
test_bcsp_b_ge_m_matches_bernoulli
test_bcsp_map_matches_local_decoder
test_bcsp_sampling_frequency_matches_probability
test_bcsp_gradcheck
test_bcsp_m15_b64_runtime
test_bcsp_m128_b64_polynomial_scaling
test_no_permutations_in_production_action_path
test_cuda_device_path_if_available
```

## 明确禁止

- permutation threshold；
- first-step entropy surrogate 作为最终数学定义；
- degree cap；
- budget cap；
- fixed candidate top-K。

## 退出条件

- real op-point action sampling/entropy 不再挂起；
- exact small-case 与 polynomial large-case 同时成立。

---

# R7：Phase 7 Graph-MAPPO 完成化

## 目标

建立第一个可运行、可训练、可扩展的 CTDE baseline。

## 工作项

1. 移除训练 critic helper 上的 `@torch.no_grad()`。
2. 分离：
   ```text
   critic_forward_train
   critic_forward_rollout
   ```
3. PPO 改为 per-agent ratio。
4. Phase 7 暂用 shared scene advantage：
   \[
   A_s=R_s-V(s).
   \]
5. entropy 使用 BCSP normalized entropy。
6. actor/critic graph batching。
7. critic optimizer、checkpoint、resume 完整。
8. 修复 CUDA device。
9. 记录：
   - critic parameter delta；
   - EV；
   - per-agent KL；
   - clip fraction；
   - normalized entropy；
   - subset cardinality；
   - active edge count；
   - evaluator calls；
   - runtime/memory。
10. real corrected shard smoke。
11. EMA/RLOO/Graph-MAPPO 公平 A/B。

## 先失败测试

```text
test_critic_formal_helper_has_grad
test_critic_optimizer_changes_parameters
test_actor_unchanged_by_critic_step
test_per_agent_ratio_epoch0_is_one
test_joint_ratio_not_used_by_actor_loss
test_graph_mappo_real_shard_smoke
test_graph_mappo_checkpoint_resume
test_graph_mappo_cuda_if_available
test_entropy_scale_is_agent_normalized
```

## 正式对照

| 方法 | evaluator calls/scene |
|---|---:|
| EMA | 1 |
| RLOO | \(M\ge2\) |
| Graph-MAPPO + BCSP | 1 |

## 退出条件

- unit/contract/integration/real-shard 四层均通过；
- critic 确实学习；
- PPO 不因 N 增长发生 joint-ratio 爆炸；
- 5-seed pilot 至少完成并诚实报告。

---

# 5. R0–R7 总验收闸门

只有全部满足后才进入 Phase 8：

```text
[ ] production f/q/fault strategy 可追溯
[ ] fixed-B 语义无错误单调性假设
[ ] relay corrected semantics 为默认
[ ] tri-state 真正接入训练
[ ] phase-specific PBFT accounting 完成
[ ] dynamic task 已建立并完成 Temporal Value Test
[ ] BCSP 替换 ordered PL
[ ] Graph-MAPPO real-shard 可训练
[ ] critic 参数实际变化
[ ] PPO 使用 per-agent ratio
[ ] 无 factorial action path
[ ] 无 smoke 参数进入 research
```

---

# 6. 继续未完成阶段

# Phase 8：Graph-Counterfactual PPO

## 目标

从 shared scene advantage 升级为 per-agent counterfactual credit。

## 工作项

1. action-conditioned vector Q critic；
2. BCSP 独立 subset counterfactual samples；
3. COMA-style per-agent baseline；
4. per-agent PPO advantage；
5. counterfactual rank correlation；
6. mutual decoder 后重新评价。

## 测试

- 只改变一个 agent subset；
- 其他 agent 固定；
- per-agent advantage 不全相同；
- baseline samples 独立于 actual subset；
- small game unbiasedness。

## 退出条件

- 相同 evaluator budget 下比 Graph-MAPPO 提升 credit/sample efficiency，或诚实 rollback。

---

# Phase 9：SCQ Exact Counterfactual Supervision

## 目标

用真实 PBFT evaluator 差分校准 Q critic。

## 工作项

1. quorum sensitivity；
2. top-M add/remove/swap subset；
3. evaluator cache；
4. exact immediate/one-step difference；
5. SCQ consistency loss；
6. witness discovery；
7. random counterfactual control。

## 测试

- sensitivity vs finite difference；
- unique subset；
- duplicate topology cache；
- state fork；
- SCQ loss 非零；
- held/test 无反馈。

## 退出条件

- critic difference error下降；
- 性能或 witness discovery 增益大于额外计算成本。

---

# Phase 10：可靠性约束与 Pareto

## 工作项

1. chance constraint：
   \[
   P(C<\tau)-\delta;
   \]
2. CVaR 可选；
3. dual 可升可降；
4. vector critic；
5. preference-conditioned actor；
6. fixed physical normalization；
7. Pareto archive；
8. constrained hypervolume。

## 退出条件

- energy/latency 均进入训练；
- checkpoint 不再只按 raw feasibility；
- 多 preference 产生非平凡前沿。

---

# Phase 11：Recurrent Directional PNA Actor

在 R5 Temporal Value Test 后决定是否启用 temporal GRU。

比较：

1. 当前 sum GNN；
2. normalized/mean GNN；
3. directional PNA；
4. recurrent directional PNA。

保持 critic、BCSP、decoder、objective、budget 和 seed 相同。

---

# Phase 12：正式泛化 Campaign

训练：

\[
N\in\{8,12,16\}.
\]

评估：

- held in-range；
- N=24；
- N=32/48；
- power；
- density；
- blockage；
- RSU count；
- \(f/n\)；
- topology hold interval；
- preference。

至少 5 independent seeds，报告：

- reliability violation；
- CVaR shortfall；
- energy；
- P95 latency；
- reconfiguration；
- policy communication；
- constrained hypervolume；
- witness discovery；
- critic OOD error；
- runtime/memory；
- CI。

---

# Phase 13：迁移与发布

1. 新 CTDE-SCQ 入口成为唯一生产 trunk；
2. EMA/RLOO 仅保留 baseline；
3. README/AGENTS/TRUNK_MAP/research log 一致；
4. 旧结果标记 retired；
5. dependency lock；
6. fresh checkout reproduction。

---

# 7. 每轮迭代工作流

## 开始前：`experiment_plan.md`

```text
Hypothesis
Changed variable
Controlled variables
Failure-first test
Success criterion
Failure criterion
Evaluator-call budget
Environment-step budget
Wall-clock budget
Seeds / splits
Mechanism activation evidence
```

## 实施顺序

1. 失败测试；
2. 最小实现；
3. targeted unit；
4. integration；
5. real-shard smoke；
6. mechanism activation；
7. pilot；
8. paired multi-seed A/B；
9. cost/scaling report。

## 结束：`decision.md`

```text
Result
Per-seed values
Confidence interval
Mechanism activation
Runtime / memory
Evaluator calls
Failure cases
Keep / Revise / Rollback
Next single hypothesis
```

连续三轮无收益时，复查：

- evaluator；
- parameter semantics；
- action reachability；
- critic gradient；
- train/deploy mismatch；
- checkpoint；
- mechanism activation；
- scaling。

不得继续堆模块。

---

# 8. 测试分层

每个阶段必须同时有：

1. **数学单元测试**：独立可计算真值；
2. **integration test**：正式调用链；
3. **production-scale complexity test**：真实 \(m,b,N\)；
4. **real-shard smoke**：正式数据；
5. **multi-seed research test**：有效性。

“同一函数的两个 wrapper 输出相等”只能算一致性测试，不能替代数学与 production-scale 测试。

---

# 9. 参数治理

## RLOO

- `baseline=rloo` 时 \(M\ge2\)；
- M 通过 evaluator-call efficiency 选择。

## BCSP

每次 run 记录：

```text
degree distribution
budget distribution
b>=m fast-path rate
DP cell count
subset cardinality distribution
normalized entropy
```

## PPO

记录：

```text
per-agent KL
per-agent clip fraction
actor gradient norm
critic gradient norm
critic parameter delta
explained variance
```

## Fault robustness

记录：

```text
effective_f
q
fault strategy
exact flag
evaluated fault-set count
approximation gap where available
```

## Dynamic task

记录：

```text
episode length
hold interval
reconfiguration cost
action-future coupling diagnostic
Temporal Value Test delta
```

---

# 10. 严禁捷径

严禁：

- permutation count threshold 作为最终解；
- first-step categorical entropy 替代真实 subset entropy；
- 限制最大 degree；
- 人为缩小物理 budget；
- 固定 candidate top-K；
- 关闭 entropy 后宣称动作问题解决；
- 用 joint PPO ratio；
- critic helper `no_grad` 却声称 critic 已训练；
- 继续按 `feasible_exists` 删除 unknown；
- 将 greedy fixed-B 称为 exact robust；
- 仅用小三角图证明 production scalability；
- 仅通过 shape/一致性测试宣称组件正确；
- test/held 选择 checkpoint；
- smoke 参数进入 headline；
- central decoder 进入 rollout/deployment；
- 同时加入多个机制后归因。

---

# 11. Definition of Done

只有同时满足以下条件才能宣布重构完成：

1. Phase 0–7 修复闸门全部通过；
2. dynamic task 已建立并由 Temporal Value Test 决定用途；
3. 无 factorial action path；
4. BCSP、per-agent PPO、local mutual acceptance 形成闭环；
5. critic 确实训练并可恢复；
6. tri-state、PBFT accounting、relay、fault semantics 正确；
7. Phase 8–10 有公平消融；
8. actor 支持 variable N；
9. 控制通信与重配置成本被计入；
10. ≥5 seed OOD 结果有 CI；
11. 文档、代码和 artifacts 一致；
12. fresh checkout 可复现。
