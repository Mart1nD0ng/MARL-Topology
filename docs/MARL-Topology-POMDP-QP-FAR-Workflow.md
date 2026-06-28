# POMDP-QP-FAR 工程工作流文档

> 目标：将 MARL-Topology 下一阶段从“全空间RL探索”转向“过时/部分CSI POMDP + 可行锚定残差学习”。  
> 原则：一次只改一个变量；先验证时序任务真的需要记忆，再验证可行域势函数，再训练残差策略。

---

# 0. 阶段总览

```text
Q0 冻结当前D13结果
Q1 实现stale/partial CSI observation model
Q2 做CSI预测与temporal value体检
Q3 实现quorum shortfall / D_quorum diagnostics
Q4 做D_quorum与真实C的alignment test
Q5 训练local_hysteresis imitation actor
Q6 实现anchor residual action space
Q7 add-only repair
Q8 conservative prune
Q9 full residual + potential-based shaping
Q10 local edge handshake / correlated sampling
Q11 PNA in residual framework
Q12 多seed、多CSI、多N campaign
Q13 文档与结果收口
```

Q1–Q4是前置条件；未通过前不得进入完整RL训练。

---

# Q0：冻结当前D13结果

## 目标

把当前负面结果作为baseline，不再覆盖。

## 工件

```text
result_save/dynamic_d13_campaign.json
docs/dynamic_repair/D13/decision.md
mechanism_activation files
per-seed raw results
```

## 退出条件

- 当前learned < deployable < central的结论可复现；
- PNA seed collapse被记录；
- D13 scope固定为N≤16、6帧、5 seed。

---

# Q1：Stale / Partial CSI Observation Model

## 目标

让actor在过时/部分CSI下决策，真实reward仍用当前CSI。

## 实现

新增：

```text
src/marl_topology/training/csi_observation_model.py
```

支持：

```text
mode = current | delay | partial | delay_partial
delay_frames
probe_probability
per_node_probe_budget
noise_std
```

DynamicScene.observation 增加参数：

```text
csi_observation_model
```

actor edge features替换为observed CSI：

```text
observed_psucc
observed_latency
observed_energy
csi_age
observed_mask
```

真实 context/evaluator 不变。

## 测试

```text
test_delay1_observation_uses_previous_frame_csi
test_current_mode_byte_identical
test_partial_csi_holds_last_observation
test_csi_age_increments_when_unobserved
test_reward_uses_true_current_csi_not_observed_csi
test_actor_never_receives_true_current_csi_under_stale_mode
```

## 退出条件

- stale CSI只影响actor observation；
- reward/evaluator仍用真实当前信道。

---

# Q2：CSI预测与Temporal Value体检

## 目标

确认stale CSI确实创造了可被history利用的POMDP。

## 实验

训练/评估简单预测头：

```text
input: observed stale CSI history
target: true current link psucc
```

比较：

```text
memoryless
recurrent
memoryless + velocity
recurrent + velocity
```

## 指标

```text
MSE(true psucc)
rank correlation
top-k critical link prediction
current topology feasibility
Temporal Value Delta_H under stale CSI
```

## 退出条件

- recurrent或velocity至少在预测当前CSI上优于memoryless；
- 若没有，说明stale参数太弱或观测仍太充分。

---

# Q3：Quorum Shortfall Diagnostics

## 目标

实现 \(D_{\text{quorum}}\)，但先不接入reward。

## 实现

新增：

```text
src/marl_topology/protocol/quorum_deficit.py
```

输出：

```text
per_phase_receiver_shortfall
D_quorum_mean
D_quorum_max
D_quorum_cvar
worst_phase
worst_receiver
```

短缺定义：

\[
\delta_{j,h}=\sum_{k=0}^{q_h-1}(q_h-k)P(S_{j,h}=k)
\]

## 测试

```text
test_shortfall_zero_when_quorum_always_met
test_shortfall_high_when_no_messages
test_shortfall_decreases_when_message_probability_increases
test_quorum_deficit_matches_bruteforce_small_case
test_deficit_uses_phase_receiver_structure
```

## 退出条件

- D_quorum数值可信；
- 可在Stage21 evaluator diagnostics中输出。

---

# Q4：D_quorum Alignment Test

## 目标

验证proxy与真实PBFT reliability对齐。

## 脚本

新增：

```text
scripts/diagnostics/quorum_deficit_alignment.py
```

对以下topology采样局部edit：

```text
random
local_hysteresis
teacher
learned
anchor-corrupted
```

计算：

```text
C
D_quorum
ΔC
ΔD
energy
latency
```

## 指标

```text
Spearman(ΔD, ΔC)
Top-k ΔD repairs true-positive rate
ΔD improves but C worsens rate
ΔD improves but energy explodes rate
```

## 退出条件

- 若alignment不足，不得接入reward；
- 若alignment良好，进入Q5/Q7。

---

# Q5：local_hysteresis imitation

## 目标

让actor先学会复现强deployable heuristic。

## 实现

新增teacher：

```text
local_hysteresis_teacher
```

Loss：

\[
L=-\sum_i\log\pi_i(S_i^H|o_i)
\]

## 测试

```text
test_hysteresis_teacher_uses_no_evaluator
test_actor_imitation_reduces_subset_nll
test_decoded_topology_matches_hysteresis_on_train
test_held_reproduction_rate_reported
```

## 评价

```text
proposal F1
decoded topology F1
held feasibility
switches/frame
return
```

## 退出条件

- actor能接近local_hysteresis；
- 如果不能，暂停RL，修actor/feature。

---

# Q6：Residual Action Space

## 目标

把策略从全拓扑生成改为anchor周围残差编辑。

## 实现

```text
base_topology = local_hysteresis(obs)
residual = add/remove/swap decisions
final_topology = base_topology xor residual
```

必须保持local mutual acceptance和部署去中心化。

## 测试

```text
test_residual_zero_equals_anchor
test_add_only_never_removes_anchor_edges
test_remove_only_never_adds_edges
test_swap_preserves_budget
test_residual_decoder_is_local
```

## 退出条件

- residual动作空间正确；
- action evaluator calls = 0。

---

# Q7：Add-only Repair

## 目标

只做可靠性修复，不做剪枝。

## 训练

从anchor失败或near-feasible场景开始，只允许add edges。

Reward/auxiliary：

\[
-D_{\text{quorum}}
\]

或PBRS版本。

## 指标

```text
anchor feasibility
add-repair feasibility
D_quorum reduction
C improvement
added edges
```

## 退出条件

- add-only repair在anchor失败场景上优于anchor；
- 若否，分析D_quorum alignment和edge features。

---

# Q8：Conservative Prune

## 目标

在可行anchor上降低成本但保持可靠性。

## 训练

只允许remove低risk边。

Safety target：

\[
risk_e=D(x\setminus e)-D(x)
\]

约束：

```text
remove only if predicted risk < margin
```

## 指标

```text
feasibility retention
energy reduction
latency reduction
removed edges
critical edge deletion rate
```

## 退出条件

- 可行保持率高；
- E/L有下降。

---

# Q9：Full Residual + PBRS

## 目标

把add/remove/swap与势函数塑形合并。

## Reward

\[
r'_t=r_t+\lambda_\Phi[\gamma\Phi_{t+1}-\Phi_t]
\]

\[
\Phi_t=-D_{\text{quorum}}(x_{t-1},g_t),\quad \Phi_T=0
\]

## 测试

```text
test_pbrs_telescopes_with_terminal_zero
test_pbrs_does_not_change_exact_small_mdp_optimum
test_shaping_only_training_not_eval
test_final_metrics_use_true_CEL
```

## 退出条件

- RL不破坏anchor；
- repair/prune均有效。

---

# Q10：Local Edge Handshake

## 目标

减少mutual acceptance中的两端错位。

## 实现

两轮邻居通信：

```text
endpoint score exchange
shared edge score
local budget selection
```

可选 correlated sampling：

```text
edge_hash_random_key
```

## 测试

```text
test_shared_edge_score_symmetric
test_handshake_uses_only_neighbor_messages
test_no_global_sort
test_mutual_acceptance_rate_improves_on_probe
test_control_message_cost_recorded
```

## 退出条件

- critical-edge disagreement下降；
- deployable边界保持。

---

# Q11：PNA in Residual Framework

## 目标

重新测试当前唯一正向线索PNA。

## 对比

```text
MLP residual
PNA residual
PNA residual + stale CSI
PNA residual + PBRS
```

## 指标

```text
seed collapse rate
mean feasibility
CI
gradient norm
runtime
parameter count
```

## 退出条件

- 判断PNA趋势是否真实。

---

# Q12：完整Campaign

## 数据

```text
urban 4RSU
random ablation
N = 8,12,16
optional N=24 if evaluator allows
frames >= 6
CSI modes = current, delay1, delay2, partial
seeds >= 5
```

## Arms

```text
local_hysteresis
anchor imitation
anchor + add repair
anchor + prune
anchor + full residual
anchor + full residual + PBRS
anchor + full residual + PBRS + PNA
central myopic reference
```

## 报告

```text
per-seed
CI
feasibility
return
energy
latency
switches
D_quorum
repair success
retention
mutual acceptance
CSI prediction error
evaluator calls
```

## 退出条件

- 明确是否超过deployable anchor；
- 明确与central reference gap；
- 明确时间机制是否有价值。

---

# Q13：文档收口

更新：

```text
README.md
CURRENT_HEAD_STATUS.md
CURRENT_DYNAMIC_REPAIR_STATUS.md
URBAN_V2X_RESEARCH_LOG.md
DYNAMIC_TASK_REPORT.md
AGENTS.md
```

必须写清：

```text
what changed
what was active
what was not active
scope
negative result boundaries
```

---

# 严禁事项

1. 不验证D_quorum alignment就接入reward。
2. 不开启stale CSI却声称时序机制已测试。
3. 让actor看到真实CSI却使用stale CSI名义。
4. PBRS用于T=1并声称有效。
5. 把proxy reward作为最终评估。
6. residual策略破坏anchor却只看平均return。
7. 不报告retention rate。
8. 不报告mutual mismatch。
9. 不比较local_hysteresis anchor。
10. 不分central reference和deployable baseline。
11. 一次同时上线hidden blockage、stale CSI、random intent和battery decay。
12. 将PNA trend写成显著胜利。
13. 用单seed判断机制。
14. 不报告seed collapse。
15. 跳过Q1-Q4直接训练完整模型。

---

# 每轮工件

每轮输出：

```text
experiment_plan.md
failing_tests.txt
changed_files.txt
mechanism_activation.json
training_history.json
validation_metrics.json
held_metrics.json
diagnostics.json
decision.md
```

关键diagnostics包括：

```text
D_quorum alignment
CSI prediction
anchor reproduction
repair success
retention
mutual acceptance
```
