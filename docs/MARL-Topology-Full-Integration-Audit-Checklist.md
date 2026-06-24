# MARL-Topology：完整环境与完整模型接入审查清单

> 文档用途：用于审查一次 MARL-Topology 实验是否真的测试了“完整环境设计”和“完整模型组件”，而不是只测试了某个简化替代方案。  
> 适用场景：Phase 0–13 复核、单步/动态实验复核、Graph-MAPPO / Counterfactual PPO / SCQ / PNA / Pareto 机制激活审查。  
> 核心问题：**实现存在 ≠ 正式训练入口已接入 ≠ 本次实验实际激活 ≠ 科研结论有效。**

---

## 0. 审查总原则

一次实验要声称测试了“完整环境”或“完整模型”，必须同时证明四件事：

```text
1. 组件存在
2. 组件接入正式训练 / 评测入口
3. 组件在该次 run 中实际激活
4. 组件在足够规模、足够 seed、正确 split 下被科研验证
```

任何机制只要停留在单元测试、mock 环境、diagnostic 脚本或 default-off 状态，都不能被视为已经进入最终性能结论。

推荐把每个机制的状态标成：

```text
NOT_IMPLEMENTED
IMPLEMENTED_ONLY
WIRED_BUT_INACTIVE
ACTIVE_IN_SMOKE
ACTIVE_IN_PILOT
ACTIVE_IN_RESEARCH
VALIDATED
NEGATIVE_BUT_VALID
FAILED_INTEGRATION
```

---

# 1. 最小证据链

每个机制都应保留以下证据：

```text
source_file
training_entrypoint
config_flag
default_or_opt_in
runtime_activation_log
unit_test
integration_test
production_scale_test
real_shard_smoke
multi_seed_result
artifact_path
```

若缺少 `runtime_activation_log`，则不能确认该机制在本次实验中实际运行。若缺少 `real_shard_smoke`，则不能确认该机制能在真实数据规模下运行。若缺少 `multi_seed_result`，则不能形成科研结论。

---

# 2. 环境数学是否完整接入

正式实验必须证明训练和评估都使用修复后的环境数学，而不是旧 evaluator 或简化 surrogate。

## 2.1 数据与 evaluator 一致性

必须记录：

```text
dataset_path
dataset_sha256
dataset_generation_command
environment_math_version
evaluator_id
regime_id
fault_model
one_hop_relay
relay_hops
timeout_aware_latency
phase_accounting_mode
solvability_status_version
```

审查问题：

1. 数据集是否由修复后的 evaluator 重新生成？
2. 训练时加载的 shards 是否就是修复后的 shards？
3. 评估时是否使用同一个 evaluator？
4. baseline 与模型是否使用同一个 evaluator？
5. SCQ 反事实是否调用同一个 evaluator？
6. 是否仍有旧 `op/` 与新 `op_corrected/` 混用？
7. result JSON 中是否只保存 basename，导致无法区分旧数据和新数据？

必须避免：

```text
训练数据来自旧环境
评估 reward 使用新环境
SCQ 用简化 evaluator
baseline 用旧 evaluator
模型用新 evaluator
```

这种混合会使结论不可解释。

---

## 2.2 PBFT quorum 与故障语义

正式实验必须记录：

```text
validator_count
configured_fault_tolerance
effective_fault_tolerance
quorum_q
external_quorum
fault_strategy
fault_set_count
enumeration_exact
is_certified
worst_case_fault_set
```

审查问题：

1. \(n,f,q\) 是否满足：
   \[
   n\geq 3f+1
   \]
   \[
   2q-n>f
   \]
   \[
   q\leq n-f
   \]
2. fixed Byzantine set \(B\) 是否跨 pre-prepare / prepare / commit 保持一致？
3. 是否还在使用 per-phase remove-largest 作为主结果？
4. 若使用 greedy fault set，是否明确标记为 optimistic / non-certified？
5. N=24 等大规模结果是否基于 exact worst-case，还是 greedy approximation？
6. 若是 approximation，是否禁止将其作为最终 headline？

通过条件：

```text
每个场景的 fault_accounting 可追溯
exact / approximate 明确
approximate 不被称为 certified
```

---

## 2.3 Route / relay 是否完整接入

必须证明只有一层多跳语义：

```text
direct one-hop matrix + relay DP
```

或者：

```text
route evaluator with max_hops
```

二者不能叠加。

必须有 A-B-C 回归：

```text
A -- B -- C

relay_hops = 1:
P(A -> C) = 0

relay_hops = 2:
P(A -> C) > 0
```

审查问题：

1. production config 是否默认启用 corrected one-hop relay？
2. legacy double-count 是否只能显式 opt-in？
3. relay path 是否累计 latency？
4. 超过 phase deadline 的 relay path 是否被排除？
5. relay energy 是否只计一次？
6. vectorized evaluator 与 reference evaluator 是否一致？

---

## 2.4 Phase-specific PBFT accounting

完整 PBFT accounting 不应继续把同一组 all-pairs records 复制给三个 phase。

必须构造：

```text
pre_prepare: primary -> backups
prepare: validators -> validators
commit: validators -> validators
```

审查问题：

1. `pre_prepare` 是否只包含 primary-to-backups？
2. `prepare` / `commit` 是否是 validator vote message？
3. clients 是否不产生 PBFT vote？
4. clients 是否仍可作为 relay？
5. 能耗是否区分 protocol、relay、retransmission、MAC-control、policy-comm、reconfig、view-change？
6. latency 是否来自 quorum-completion，而不是 max-all-pairs deadline clipping？
7. 失败拓扑是否支付 timeout？

通过条件：

```text
phase-specific message plan 接入 production evaluator
energy/latency 对 topology 有辨识度
失败不会得到零时延优势
```

---

## 2.5 Tri-state solvability 是否接入训练

必须使用：

```text
witness_feasible
certified_infeasible
unknown
```

而不是旧二值 `feasible_exists`。

审查问题：

1. finite search miss 是否被标记为 unknown？
2. unknown 是否被默认删除？
3. unknown 是否被当成 certified violation？
4. witness memory 是否 train / val / held 隔离？
5. 策略在 unknown 上发现可行拓扑时，是否记录 U→W？
6. conditional metric 是否改名为 witness recall，而不是 provably solvable recall？

通过条件：

```text
训练主干不再依赖 feasible_exists 过滤
unknown 的训练策略明确：探索 / discovery，而不是简单负 dense penalty
held/test witness 不反馈训练
```

---

# 3. 动态任务是否完整接入

动态数据底座不等于动态强化学习任务。

完整动态任务必须具备：

```text
episode_length > 1
state_t
action_t
topology persists
reconfiguration cost
state_t+1
return over horizon
```

## 3.1 必须记录的动态参数

```text
episode_length
topology_hold_interval
PBFT_micro_rounds_per_macro_step
gamma
reconfiguration_energy_per_edge
reconfiguration_latency_per_edge
queue_enabled
battery_enabled
previous_topology_enabled
state_fork_enabled
```

## 3.2 必须通过的动态机制测试

```text
test_episode_length_gt_one
test_topology_persists_for_hold_interval
test_action_changes_future_cost
test_reconfiguration_cost_nonzero
test_state_fork_isolation
test_T1_matches_static_mode
```

## 3.3 Temporal Value Test

必须比较：

\[
J_{\mathrm{myopic}}
\]

与：

\[
J_{\mathrm{horizon}}
\]

并计算：

\[
\Delta_H = J_{\mathrm{myopic}} - J_{\mathrm{horizon}}.
\]

审查问题：

1. Temporal Value Test 是否在真实 trajectory / corrected evaluator 上执行？
2. 是否只在 synthetic 2-frame toy 上执行？
3. 若 \(\Delta_H\approx0\)，其结论是否只限于对应 \(H\)、reconfiguration cost 和数据分布？
4. 若正式性能测试仍是 \(T=1\)，是否明确标注“只验证静态子任务”？

通过条件：

```text
动态环境接入正式训练入口
或明确声明动态机制只是 opt-in extension，当前 headline 不覆盖动态最终任务
```

---

# 4. Actor 与动作路径是否完整接入

完整 actor 路径应为：

\[
o_i
\rightarrow
\theta_{i,e}
\rightarrow
S_i\sim \pi_i^{\mathrm{BCSP}}
\rightarrow
D_{\mathrm{mutual}}(\mathbf S)
\rightarrow
x
\rightarrow
(C,E,L)
\]

## 4.1 BCSP 是否替代旧有序动作

应使用预算条件化无序子集策略：

\[
\pi_i(S)
=
\frac{
\mathbf 1[|S|\leq b_i]\exp(\sum_{e\in S}\theta_e)
}{
Z_i(\theta)
}.
\]

审查问题：

1. 正式训练是否调用 BCSP sampler？
2. PPO log-prob 是否是无序 subset 的 log-prob？
3. 是否仍然调用 ordered Plackett–Luce 作为主 path？
4. 代码路径中是否仍有 production 使用 `itertools.permutations`？
5. \(b\geq m\) 时是否走 independent Bernoulli fast path？
6. entropy 是否是 subset entropy，而不是 order entropy？
7. entropy 是否按每个 agent 的合法 action count 归一化？

必须避免：

```text
同一 subset 的不同 order 被当成不同动作
k=m 时仍出现 log(m!) entropy
permutation threshold + first-step entropy surrogate 作为最终方案
```

## 4.2 Per-agent ratio 是否真实使用

PPO 应使用：

\[
\rho_i
=
\exp(
\log\pi_i^{\mathrm{new}}(S_i)
-
\log\pi_i^{\mathrm{old}}(S_i)
).
\]

禁止主 actor loss 使用 joint ratio：

\[
\rho_{\mathrm{joint}}
=
\exp(
\sum_i
[
\log\pi_i^{\mathrm{new}}
-
\log\pi_i^{\mathrm{old}}
]
).
\]

审查问题：

1. actor loss 中 flatten 的单位是 agent-action 还是 scene？
2. epoch 0 是否逐 agent ratio = 1？
3. KL / clip fraction 是否按 per-agent 统计？
4. joint logp 是否只用于诊断？

通过条件：

```text
PPO loss 以 per-agent ratio 为输入
joint ratio 不进入 actor loss
```

## 4.3 Local mutual acceptance 是否唯一 decoder

训练和部署都必须使用：

\[
x_{ij}
=
\mathbf1[
e_{ij}\in S_i
\land
e_{ij}\in S_j
].
\]

审查问题：

1. 主 rollout 是否调用 local mutual acceptance？
2. 评估是否调用同一 decoder？
3. 是否存在训练用 global decoder、部署用 local decoder？
4. 是否存在失败后 centralized repair？
5. global argsort 是否只作为 ablation？

---

# 5. Critic 是否完整接入

## 5.1 Graph-MAPPO value critic

critic 必须：

```text
训练时构造
训练 forward 有梯度
optimizer step 后参数变化
checkpoint 保存与恢复
部署 artifact 不包含 critic
```

审查问题：

1. critic forward 是否被 `torch.no_grad()` 包裹？
2. critic loss 是否 backward？
3. critic optimizer 是否 step？
4. actor 参数是否不会被 critic update 改变？
5. explained variance 是否被记录？
6. critic 输入是否含 teacher topology / oracle / held outcome？
7. critic 是否在正式 Graph-MAPPO run 中实际激活？

通过条件：

```text
critic_parameter_delta > 0
critic_history 非空
critic_metrics.json 存在
```

## 5.2 Action-conditioned Q critic

Counterfactual PPO 必须使用：

\[
Q_\phi(s,\mathbf S)
\]

而不是仅使用：

\[
V_\phi(s).
\]

审查问题：

1. `critic_sees_action=True` 是否激活？
2. Q 输入是否包含 realized active edge one-hot？
3. action one-hot 是否 detach？
4. \(Q(s,S)\) 是否随 action 改变？
5. Q critic 是否训练到 reward？
6. resume 时是否校验 V/Q critic architecture mismatch？

---

# 6. Counterfactual PPO 是否完整接入

完整 COMA-style 反事实为：

\[
A_i
=
Q(s,\mathbf S)
-
\mathbb E_{\tilde S_i\sim\pi_i}
Q(s,\tilde S_i,\mathbf S_{-i}).
\]

审查问题：

1. 每个 agent 是否有自己的 \(A_i\)？
2. per-agent advantages 是否不全相同？
3. \(	ilde S_i\) 是否独立于实际 \(S_i\)？
4. 反事实是否只改变 agent \(i\) 的 subset？
5. \(S_{-i}\) 是否固定？
6. 反事实是否重新经过 mutual decoder？
7. counterfactual Q forward 是否不消耗 evaluator calls？
8. actor loss 是否真的使用 per-agent advantage，而非 scene scalar？

通过条件：

```text
within-scene advantage variance > 0
baseline independence test passed
eval_calls_per_scene 未因 COMA 增加
```

---

# 7. SCQ 是否完整接入

SCQ 应调用真实 evaluator 计算 exact difference：

\[
\Delta R_i^{\mathrm{env}}
=
R(\mathbf S)
-
R(\tilde S_i,\mathbf S_{-i}).
\]

critic consistency：

\[
\mathcal L_{\mathrm{SCQ}}
=
[
Q(s,\mathbf S)
-
Q(s,\tilde S_i,\mathbf S_{-i})
-
\Delta R_i^{\mathrm{env}}
]^2.
\]

审查问题：

1. `--scq` 是否启用？
2. `--counterfactual` 是否同时启用？
3. SCQ 是否记录 extra evaluator calls？
4. SCQ loss 是否进入 critic loss？
5. SCQ 是否只影响 critic，不直接进入 actor gradient？
6. top-M 反事实是否有 unique subset？
7. duplicate topology 是否有 cache？
8. exact delta 是否来自真实 evaluator？
9. `scq_critic_difference_error` 是否下降？
10. held Q fidelity 是否提升？

通过条件：

```text
SCQ evaluator calls > 0
SCQ loss > 0
SCQ difference error 有日志
Q-difference 与 exact delta 的相关性提升
```

若 SCQ 只在单元测试中有效、正式训练中未提升 held Q fidelity，则应标记为：

```text
verified primitive
opt-in
not promoted
```

而不是“最终主方法有效”。

---

# 8. Chance / CVaR / Pareto 是否完整接入

## 8.1 Chance constraint

可靠性约束应使用：

\[
g(\theta)
=
\Pr(C<\tau)-\delta.
\]

审查问题：

1. `--chance` 是否启用？
2. residual 是否可正可负？
3. \(\lambda\) 是否可升可降？
4. reward 是否加入 chance penalty？
5. 默认 off 时是否 byte-identical？
6. 训练曲线是否记录：
   ```text
   frac_below_tau
   chance_residual
   lambda_chance
   ```

## 8.2 CVaR

审查问题：

1. CVaR 是否只是 primitive？
2. 是否接入训练 loss？
3. 是否接入 checkpoint selection？
4. 是否只作为 metric 报告？

不要把“CVaR primitive 已验证”写成“CVaR 训练已完成”。

## 8.3 Pareto archive

审查问题：

1. `--pareto-archive` 是否启用？
2. archive 是否保存每个 validation checkpoint？
3. 是否记录：
   ```text
   reliability_violation
   energy
   latency
   hypervolume
   selected_update
   ```
4. 最终 checkpoint 是否来自 archive selection，而非 raw feasibility？
5. 是否至少有两个非支配点？
6. 是否评估多个 preference \(\omega\)？

通过条件：

```text
checkpoint rule = reliability-risk → min violation → non-dominated E/L → hypervolume → stability
```

---

# 9. PNA / Recurrent actor 是否完整接入

## 9.1 PNA actor

审查问题：

1. `--actor pna` 是否启用？
2. PNA 是否接入正式 training entrypoint？
3. PNA 是否使用 BCSP + local mutual decoder？
4. PNA 是否在 corrected dataset 上多 seed 比较？
5. PNA 是否只是 smoke 测试？
6. 默认 actor 是否仍是 MLP？
7. PNA 是否在 dynamic \(T>1\) 场景中测试？

若只在单步 \(N\le16\) 测试，则只能得出：

```text
PNA 在静态小规模任务上没有超过 MLP
```

不能得出：

```text
PNA 对动态任务无效
PNA 对大规模泛化无效
```

## 9.2 Recurrent temporal state

审查问题：

1. actor 是否维护 \(h_{i,t}\)？
2. hidden state 是否跨 episode step 更新？
3. episode 是否 \(T>1\)？
4. reset 是否正确？
5. previous topology 是否输入 actor？
6. temporal hidden state 是否影响 action？

若 \(T=1\)，recurrent actor 实际退化成更大静态网络。

---

# 10. Preference-conditioned policy 是否完整接入

审查问题：

1. actor 是否输入 \(\omega=(\omega_E,\omega_L)\)？
2. 训练是否随机采样多个 \(\omega\)？
3. 评估是否 sweep 多个 \(\omega\)？
4. 不同 \(\omega\) 是否产生不同拓扑？
5. energy/latency 是否都进入 reward？
6. Pareto archive 是否使用多个 preference？

通过条件：

```text
preference-action mutual information > 0
不同 omega 下 E/L tradeoff 非平凡
```

若 \(\omega\) 固定为：

\[
(0.5,0.5)
\]

则不能声称测试了 preference-conditioned Pareto policy。

---

# 11. Run artifact 检查模板

每次正式实验至少应包含：

```text
config.yaml
manifest.json
git_revision.txt
environment_version.json
data_manifest.json
seed_manifest.json
mechanism_activation.json
train_history.jsonl
critic_metrics.json
dual_history.json
validation_metrics.json
held_metrics.json
pareto_archive.json
solvability_report.json
failure_cases.json
decision.md
stdout.txt
stderr.txt
```

`mechanism_activation.json` 应至少包含：

```json
{
  "dynamic_task": {
    "enabled": true,
    "episode_length": 8,
    "hold_interval": 4,
    "reconfiguration_cost_nonzero": true
  },
  "environment_math": {
    "fault_model": "fixed_set",
    "one_hop_relay": true,
    "timeout_aware_latency": true,
    "phase_specific_accounting": true
  },
  "action": {
    "distribution": "bcsp",
    "per_agent_ratio": true,
    "local_mutual_decoder": true
  },
  "critic": {
    "graph_mappo": true,
    "critic_parameter_delta": 0.123,
    "critic_checkpointed": true
  },
  "counterfactual": {
    "enabled": true,
    "per_agent_advantage": true,
    "advantage_variance_nonzero": true
  },
  "scq": {
    "enabled": true,
    "evaluator_calls": 128,
    "loss_nonzero": true
  },
  "constraint": {
    "chance_enabled": true,
    "dual_up_down": true
  },
  "pareto": {
    "enabled": true,
    "archive_entries": 12,
    "selected_by_archive": true
  }
}
```

---

# 12. 快速判定表

| 观察到的实验 | 可以说明 | 不能说明 |
|---|---|---|
| 单步 \(T=1\) Graph-MAPPO 不超 baseline | 静态子任务上无显著收益 | 动态最终架构失败 |
| PNA 在 \(N\le16\) 不超 MLP | 静态小规模 PNA 没优势 | recurrent / large-N PNA 无效 |
| SCQ Q fidelity 不提升 | 当前 SCQ 设置无收益 | SCQ 机制理论无效 |
| COMA 不超 shared advantage | 当前 Q 质量/尺度下无收益 | per-agent credit 不需要 |
| suite 全绿 | 代码局部正确 | 正式科研结论成立 |
| smoke 退出 0 | 入口可运行 | 模型有效 |
| corrected shards 被构建 | 数据存在 | 本次实验实际使用 |
| dynamic env 类存在 | 机制存在 | 动态任务已训练 |

---

# 13. 最终审查结论模板

建议每次审查用如下格式输出：

```text
实验名称：
git commit：
数据版本：
训练入口：
命令：
seed：
split：
是否 headline eligible：

环境接入：
[PASS/PARTIAL/FAIL] safe quorum
[PASS/PARTIAL/FAIL] fixed fault set
[PASS/PARTIAL/FAIL] one-hop relay
[PASS/PARTIAL/FAIL] phase accounting
[PASS/PARTIAL/FAIL] timeout latency
[PASS/PARTIAL/FAIL] tri-state

模型接入：
[PASS/PARTIAL/FAIL] BCSP
[PASS/PARTIAL/FAIL] local mutual decoder
[PASS/PARTIAL/FAIL] per-agent PPO ratio
[PASS/PARTIAL/FAIL] graph critic
[PASS/PARTIAL/FAIL] counterfactual Q
[PASS/PARTIAL/FAIL] SCQ
[PASS/PARTIAL/FAIL] chance/CVaR
[PASS/PARTIAL/FAIL] Pareto archive
[PASS/PARTIAL/FAIL] PNA
[PASS/PARTIAL/FAIL] dynamic task

机制激活：
列出每个机制的 runtime evidence。

结果：
列出 raw、witness recall、energy、latency、risk、hypervolume、per-N、CI。

结论：
只能对实际激活并验证过的范围下结论。
```

---

# 14. 本清单的核心用途

如果一个实验报告说：

> “完整 CTDE-SCQ 模型没有超过简单基线。”

必须用本清单逐项确认：

1. 是否真的是完整环境；
2. 是否真的是完整模型；
3. 是否动态任务真正启用；
4. 是否所有机制进入正式 loss；
5. 是否所有机制有 runtime evidence；
6. baseline 是否公平；
7. 结论是否只覆盖实际测试范围。

若最终实验仍是：

```text
T = 1
N <= 16
Graph-MAPPO shared advantage
MLP actor
SCQ / chance / Pareto / PNA default off or opt-in
```

则严谨结论只能是：

> 在修复后的静态小规模 contextual-bandit 子任务上，复杂机制没有超过简单 Graph-MAPPO / MLP baseline。

不能写成：

> 完整动态 CTDE-SCQ-MARL 架构失败。
