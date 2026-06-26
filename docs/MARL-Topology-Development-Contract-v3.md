# MARL-Topology 开发契约：完整接入、实验严谨性与反退化规范

> 文档类型：开发契约 / Agent 工作边界  
> 项目：MARL-Topology  
> 目的：防止后续开发再次出现“组件已实现但正式实验未接入”“动态实验没有接入完整 Phase 8–11 架构”“测试用简化替代方案”“机制参数失效”“实验结论超出实际激活范围”等问题。  
> 适用对象：所有后续 agent / Codex / Claude / 人工开发者。  
> 生效范围：环境数学、动态任务、训练入口、actor/critic/decoder、SCQ、COMA、PNA、chance/CVaR、Pareto、评估脚本、结果报告、README/日志。

---

# 0. 核心铁律

项目中的任何机制都必须同时满足四层要求，才能被称为“已完成”：

```text
1. 组件已实现
2. 组件已接入正式训练 / 评估入口
3. 组件在本次 run 中实际激活
4. 组件在真实规模、正式配置、多 seed、正确 split 下验证
```

只满足第 1 层或第 2 层，不得写成“完成”。  
只在 smoke / unit test / diagnostic script 中运行，不得写成“正式测试”。  
只在单步 \(T=1\) 中验证，不得写成“动态任务验证”。  
只在默认 off 状态存在，不得写成“完整模型验证”。  
只对比中央 evaluator reference，不得写成“简单可部署基线胜出”。

---

# 1. 状态命名契约

每个机制必须标注为以下状态之一：

```text
NOT_IMPLEMENTED
IMPLEMENTED_ONLY
WIRED_BUT_INACTIVE
ACTIVE_IN_SMOKE
ACTIVE_IN_PILOT
ACTIVE_IN_RESEARCH
VALIDATED_POSITIVE
VALIDATED_NEGATIVE
NEGATIVE_BUT_SCOPE_LIMITED
FAILED_INTEGRATION
DEFERRED
RETIRED
```

禁止使用模糊状态：

```text
done
completed
works
full model
final result
verified
```

除非明确附带：

```text
activated_in_entrypoint
config
test_scale
seed_count
artifact_path
scope_limit
```

---

# 2. 组件完成定义

## 2.1 环境组件完成定义

一个环境机制只有满足以下条件才算完成：

```text
math primitive implemented
unit test with independent truth
integrated into Stage-21 / production evaluator
used by dataset generation
used by training reward
used by evaluation
logged in mechanism_activation or metrics
covered by real-shard smoke
```

例如，`PBFTMessagePlan` 文件存在不等于 phase-specific accounting 完成。只有当正式 `Stage21ObjectiveStackEvaluator.evaluate()` 不再复用同一组 all-pairs records，而是真正使用 pre-prepare / prepare / commit message plan 时，Phase-specific accounting 才能标记完成。

## 2.2 模型组件完成定义

一个模型组件只有满足以下条件才算完成：

```text
module exists
training entrypoint can instantiate it
runtime config can activate it
activation log records it
loss actually uses it
gradient reaches it if trainable
checkpoint saves/restores it
held evaluation used same semantics
ablation compares on same budget
```

例如，PNA actor 存在不等于“PNA 已测试”。必须说明：

```text
--actor pna
dataset
N
seeds
budget
comparison arm
result
```

## 2.3 算法机制完成定义

一个算法机制只有满足以下条件才算完成：

```text
formal objective defined
implementation matches objective
mechanism activation assertion
nonzero signal during run
diagnostic metrics recorded
negative/positive result reported with CI
```

例如，SCQ 不能只计算 counterfactual；必须进入 critic loss，并记录：

```text
scq_evaluator_calls
scq_loss
exact_delta_variance
critic_difference_error
held_q_fidelity
```

---

# 3. 动态任务接入契约

动态数据底座不等于动态强化学习任务。

## 3.1 动态任务必须具备

正式动态任务必须同时具有：

```text
episode_length > 1
state_t
action_t
topology persists or transition depends on action
reconfiguration cost
state_t+1
discounted or explicitly undiscounted return
validation over whole trajectory
held/test over disjoint trajectories
```

必须满足：

\[
s_{t+1}\sim P(s_{t+1}\mid s_t,a_t)
\]

或至少 reward/hidden/topology state 中存在动作对未来的影响。

若车辆只是按预定轨迹移动，而每一帧都独立重新优化且无切换代价，则它只是“相关帧 contextual bandit”，不能称为完整动态 RL。

## 3.2 `hold_interval` 契约

如果文档声称拓扑保持 \(H_{\mathrm{PBFT}}\) 个 PBFT 微轮，则 RL reward 必须体现：

\[
r_t
=
H_{\mathrm{PBFT}}\,r_{\mathrm{base},t}
-
c_{\mathrm{reconfig},t}
\]

或明确给出不同数学定义。

禁止：

```text
Temporal Value Test 使用 H * base_reward
RL reward 不使用 H
```

这会导致 Temporal Value Test 与训练任务不是同一个问题。

## 3.3 动态训练与评估目标一致

若训练使用 discounted return：

\[
G_t=r_t+\gamma G_{t+1}
\]

则 validation / checkpoint / held comparison 必须使用同一 discounted episode return，或明确说明为什么使用 undiscounted return。

禁止：

```text
训练优化 discounted return
checkpoint 使用 undiscounted return
报告混用两者
```

## 3.4 动态 split 契约

动态实验必须至少有三组轨迹：

```text
train
validation
held/test
```

禁止用训练轨迹做 keep-best，然后把 held drop 称为纯泛化差距。若算力限制只能 train/held，必须写成：

```text
pilot only
no independent validation
not headline eligible
```

## 3.5 动态观测 Markov 性契约

如果报告声称：

```text
current CSI + previous topology is approximately Markov
```

必须证明 actor 或 critic 可观测到足以预测下一帧的信息，例如：

```text
velocity
heading
relative velocity
distance derivative
CSI history
CSI age
```

若 actor看不到速度/航向，而车辆按隐藏速度运动，则不能声称 memoryless observation 是 Markov。

---

# 4. 数据生成契约

## 4.1 数据描述必须与代码一致

报告中描述的数据生成机制必须逐项对应源码调用。

若代码调用：

```text
_sample_scene()
```

且 `_node()` 只生成：

```text
rsu_0 + vehicles
```

则报告不能写成：

```text
4-RSU urban grid
```

除非确实调用了城市网格与多 RSU sampler。

## 4.2 动态数据 manifest 必须记录

```text
scene_sampler
rsu_count
road/grid/building model
node_count_distribution
trajectory_length_distribution
vehicle_speed_distribution
motion_model
candidate_graph_mode
channel_model
shadowing/nlosv flags
fault_model
relay mode
timeout latency flag
phase_accounting mode
content hash
```

## 4.3 场景族不等于可解性

若动态 hot path 没有每帧运行 witness search 或 upper-bound certificate，则：

```text
solvability_family = feasible_sparse / near_threshold / infeasible
```

只能是 intended family，不是 measured solvability。

报告必须区分：

```text
intended_family
finite_search_witness
certified_infeasible
unknown
policy_success
```

禁止将 `scenario_id` 后缀当成真实 solvability 标签。

---

# 5. 环境数学完整接入契约

## 5.1 PBFT quorum

每次正式训练和评估必须记录：

```text
validator_count
configured_f
effective_f
quorum_q
external_quorum
fault_model
fault_strategy
fault_set_count
enumeration_exact
is_certified
```

若使用 greedy fault strategy，报告必须写：

```text
optimistic approximation
not certified
```

## 5.2 Route / relay

必须证明：

```text
one_hop_relay=True
relay_hops=H
no double-counting
latency-aware relay
```

并保留 A-B-C 测试。

## 5.3 Phase-specific accounting

完整 accounting 必须证明正式 evaluator 使用：

```text
pre_prepare: primary -> backups
prepare: validator -> validator
commit: validator -> validator
```

禁止只实现 `pbft_message_plan.py` 而不接入生产 evaluator。

## 5.4 Timeout-aware latency

若报告说 timeout-aware latency 已启用，必须记录：

```text
timeout_aware_latency=True
failed topology pays timeout
latency source
phase budget
```

## 5.5 Energy/latency/reliability 同源

训练 reward、validation metrics、held metrics、baseline、SCQ exact deltas 必须使用同一个 evaluator 口径。

禁止：

```text
训练 reward 一个 evaluator
baseline reference 另一个 evaluator
SCQ 使用简化 evaluator
held 使用新 evaluator
```

---

# 6. Actor / Decoder 契约

## 6.1 动作分布

生产训练中的节点动作是无序子集：

\[
S_i\subseteq \mathcal E_i,\quad |S_i|\le b_i
\]

应使用 BCSP：

\[
\pi_i(S_i)
=
\frac{
\mathbf 1[|S_i|\le b_i]\exp(\sum_{e\in S_i}\theta_e)
}{Z_i(\theta)}
\]

禁止将同一 subset 的不同 permutation 当成不同动作。

## 6.2 per-agent ratio

PPO 必须使用：

\[
\rho_i=
\exp(\log\pi_i^{new}-\log\pi_i^{old})
\]

禁止 joint ratio 进入 actor loss：

\[
\rho_{\mathrm{joint}}=
\exp\sum_i(\log\pi_i^{new}-\log\pi_i^{old})
\]

## 6.3 Decoder 一致性

训练 rollout 和部署必须语义一致。

允许：

```text
train: stochastic BCSP proposal
eval/deploy: local_mutual_assemble MAP decoder
```

但必须明确：

```text
BCSP MAP == local mutual decoder up to tie set
```

禁止：

```text
train uses global decoder
eval/deploy uses local decoder
failure repair uses central solver
```

## 6.4 Communication graph 契约

若 candidate graph 是 complete graph，message passing 也是 all-to-all。

报告必须明确：

```text
candidate graph mode
communication graph mode
control communication cost
```

不能在 complete graph message passing 下声称已验证稀疏物理邻居通信。

---

# 7. Critic 契约

## 7.1 central critic 合法边界

训练阶段允许 centralized critic。部署阶段禁止 critic。

必须记录：

```text
critic_enabled
critic_type
critic_inputs
critic_parameter_delta
critic_loss
critic_explained_variance
checkpoint_saved
checkpoint_restored
```

## 7.2 no-grad 禁令

critic train forward 不得处于 `torch.no_grad()`。

必须有测试：

```text
critic formal helper has grad
optimizer step changes critic parameters
actor unchanged by critic step
```

## 7.3 dynamic critic 契约

若任务是 POMDP，critic至少应考虑：

```text
history
previous topology
velocity / true state if training-only allowed
action if Q critic
```

若 critic只是 per-frame scalar \(V(s_t)\)，报告必须写：

```text
per-frame scalar critic only
no action-conditioned Q
no recurrent critic
```

不能声称完整 graph-temporal vector critic 已验证。

---

# 8. Counterfactual / SCQ 契约

## 8.1 COMA / Counterfactual PPO

若报告说使用 per-agent counterfactual credit，必须记录：

```text
counterfactual_enabled
Q critic enabled
critic_sees_action=True
K_cf
per-agent advantage variance
baseline independence
eval calls unchanged
```

actor loss必须真的使用：

\[
A_i=Q(s,\mathbf S)-\mathbb E_{\tilde S_i}Q(s,\tilde S_i,\mathbf S_{-i})
\]

禁止仍使用 scene scalar advantage。

## 8.2 SCQ

SCQ必须记录：

```text
scq_enabled
scq_m
exact evaluator calls
unique counterfactuals
duplicate topology cache
scq_loss
critic_difference_error
held_q_fidelity
```

如果SCQ只是存在或单元测试通过，正式实验没启用，必须标成：

```text
implemented_only / opt_in
```

不能说完整模型已测试。

---

# 9. Chance / CVaR / Pareto 契约

## 9.1 chance

必须使用：

\[
g=\Pr(C<\tau)-\delta
\]

并记录：

```text
frac_below_tau
delta
residual
lambda
lambda_up_down
```

如果 residual 永远非负或 lambda 只能上升，不得称为正式 chance constraint。

## 9.2 CVaR

必须区分：

```text
CVaR primitive
CVaR metric
CVaR loss
CVaR checkpoint criterion
```

## 9.3 Pareto

若报告称 Pareto 已启用，必须有：

```text
multiple preferences
energy metric
latency metric
reliability risk
archive entries
non-dominated entries
hypervolume
selected checkpoint source
```

只按 raw feasibility 选 checkpoint，不得称为 Pareto optimization。

---

# 10. Baseline 契约

## 10.1 Deployable baseline 与 oracle/reference分离

必须区分：

```text
deployable local baseline
centralized diagnostic baseline
oracle/reference
teacher
```

若 baseline 每帧调用 evaluator搜索候选 topology，它是 centralized reference，不是 deployable simple baseline。

## 10.2 Budget 公平

不同方法至少要报告：

```text
environment steps
evaluator calls
critic forwards
SCQ evaluator calls
wall-clock
parameter count
teacher/warm-start data
```

不得只比较 update 数。

## 10.3 Warm-start 公平

若使用 warm-start，必须记录：

```text
teacher source
teacher uses evaluator? 
teacher uses held? 
warmstart epochs
warmstart lr
warmstart held metric measurement-only
post-RL metric
warmstart-alone episode return
```

若 RL与teacher目标不同，必须同时报告 feasibility 和 episode return。

---

# 11. 测试契约

每个机制必须有五类测试：

```text
math truth test
integration test
production-scale complexity test
real-shard smoke
multi-seed research run
```

## 11.1 禁止的伪测试

以下测试不能单独证明机制有效：

```text
wrapper A == wrapper B
shape correct
import succeeds
mock evaluator fixed reward
3-update smoke
triangle graph only
forward only no backward
same implementation recompute equality
```

## 11.2 必须优先写失败测试

每轮必须先写至少一个会在旧代码上失败的测试。

示例：

```text
dynamic reward uses hold_interval
train/eval objective consistency
phase-specific accounting in Stage21 evaluator
BCSP no permutation path
critic parameter delta
per-agent ratio not joint
SCQ loss nonzero
PNA activated in training entrypoint
```

---

# 12. 报告契约

每份结果报告必须包含：

```text
scope
what was active
what was default-off
what was not tested
dataset exact description
command
config
seeds
split
runtime artifacts
baseline classification
CI
failure modes
scope of conclusion
```

必须写出：

```text
This result covers only ...
This result does not prove ...
```

禁止标题党式写法：

```text
full model failed
dynamic MARL failed
SCQ failed
PNA failed
recurrence useless
```

除非这些机制在对应范围完整激活并通过公平实验验证。

---

# 13. 严禁的错误开发方法

严禁以下做法：

1. 写一个组件，但不接入正式训练入口，然后宣称完成。
2. 接入一个组件，但默认 off，然后用默认实验宣称它无效。
3. 单元测试通过后跳过 real-shard smoke。
4. 用 mock evaluator 结果代表真实 evaluator。
5. 用旧数据测试新环境。
6. 用新环境评估旧数据训练出的策略而不标注 distribution shift。
7. 实验报告描述的数据生成与源码不一致。
8. 训练与验证使用不同 reward / return 定义。
9. 训练用 discounted return，checkpoint 用 undiscounted return。
10. 只有 train/held，没有 validation，却声称严格泛化评估。
11. 将 central evaluator reference称为 deployable baseline。
12. 使用 hidden teacher / searcher / evaluator repair 进入部署路径。
13. 将 default-off 的 SCQ/Pareto/PNA 写成“完整模型已测试”。
14. dynamic branch 没接 Phase 8–11，却写成完整 Phase 0–13 模型。
15. 使用 smoke/pilot 参数作为headline。
16. 用3 seed大方差结果下强结论。
17. 不报告失败seed或坍缩seed。
18. 不报告 per-seed 原始值。
19. 不报告机制激活文件。
20. 不报告 evaluator call budget。
21. 不报告 negative result 的适用范围。
22. 把“没有收益”写成“理论无效”。
23. 因为结果不好而改弱测试。
24. 同时改多个机制后做单因素归因。
25. 不更新 README/日志导致文档与代码冲突。

---

# 14. 每轮开发必须输出的工件

每轮至少输出：

```text
experiment_plan.md
failing_tests.txt
changed_files.txt
mechanism_activation.json
run_manifest.json
stdout.txt
stderr.txt
unit_test_summary.txt
integration_test_summary.txt
real_shard_smoke_summary.txt
pilot_result.json
decision.md
```

正式实验还必须输出：

```text
per_seed_results.json
confidence_intervals.json
data_manifest.json
split_manifest.json
training_history.json
held_traces.json
critic_metrics.json
baseline_budget_report.json
scope_limits.md
```

---

# 15. 最终验收格式

每个阶段结束时必须给出：

```text
Phase:
Status:
Implemented:
Wired into entrypoint:
Active by default:
Active in this run:
Test scale:
Seed count:
Dataset:
Evaluator:
Decoder:
Critic:
Actor:
Mechanisms active:
Mechanisms not tested:
Positive findings:
Negative findings:
Conclusion scope:
Next action:
```

没有这个表，不得宣布 phase 完成。

---

# 16. 当前项目特别需要防止的复发问题

这些是本项目过去几轮已经出现过的问题，必须重点防止：

## 16.1 RLOO M=1

声明 RLOO 必须：

\[
M\ge2
\]

否则 fail-fast。

## 16.2 Ordered PL factorial动作

生产动作分布必须是 BCSP 无序 subset，不得回退到 permutation action。

## 16.3 Critic no-grad

critic训练forward不能被 `@torch.no_grad()` 包裹。

## 16.4 Joint PPO ratio

actor loss不得使用 joint ratio。

## 16.5 Unknown 被删除

unknown 场景不得被旧 `feasible_exists=False` 静默过滤。

## 16.6 Phase-specific accounting 未接入

`pbft_message_plan.py` 存在不等于 production evaluator 完成。

## 16.7 动态实验未接完整模型

如果 dynamic branch没有接入 COMA、SCQ、chance、Pareto、PNA、vector critic，报告必须明确写：

```text
tested dynamic recurrent-PPO only
not full Phase 8-11 model
```

## 16.8 动态数据描述夸大

若实际是单RSU随机平面，不得写成4-RSU urban grid。

## 16.9 hold_interval无效

若文档说拓扑保持 \(H\) 个PBFT轮，reward必须体现 \(H\)。

## 16.10 中央reference混称baseline

myopic evaluator-greedy 必须标为 centralized reference，不得标为deployable simple baseline。

---

# 17. 本契约的最终目标

让所有后续agent都遵守：

\[
\boxed{
\text{组件完成}
\neq
\text{实验激活}
\neq
\text{科研验证}
}
\]

并确保每个负面结果都能回答：

```text
到底测试了什么？
没有测试什么？
失败来自机制本身，还是来自环境、目标、数据、训练或评估口径？
```

只有这样，项目才能避免在复杂机制和复杂环境中反复出现“写了很多代码，但结论其实只来自简化路径”的问题。
