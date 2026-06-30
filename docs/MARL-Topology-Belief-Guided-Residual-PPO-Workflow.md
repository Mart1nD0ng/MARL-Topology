# MARL-Topology 工作流：Belief-Guided Evidence-Gated Residual PPO

> 文档类型：工程工作流  
> 目的：指导从当前 Q14 状态推进到真正测试 residual PPO、CSI belief、beneficial edit supervision 的下一阶段。

---

# 0. 阶段总览

```text
R0 冻结 Q14 结果与代码路径审计
R1 修复 stale/partial feature standardization 与 logit saturation 监控
R2 接入 CSI belief prediction auxiliary
R3 接入 residual PPO / CTDE value critic
R4 生成 beneficial oracle-edit dataset
R5 训练 repair/safety/edit heads
R6 evidence-gated residual action
R7 adaptive anchor KL / safety constraint
R8 full method pilot
R9 multi-seed research campaign
R10 文档与结论收口
```

R1–R5 是前置条件。未完成前不得再次宣布 residual learning失败。

---

# R0：冻结 Q14 与路径审计

## 目标

明确 Q14 的失败来自 residual trainer 路径，而不是 Graph-MAPPO 主路径。

## 工作项

1. 保存 Q14 raw result。
2. 路径审计：
   ```text
   residual trainer uses REINFORCE?
   residual trainer calls ppo_clip_actor_loss?
   residual trainer uses centralized critic?
   residual trainer logs KL?
   residual trainer logs entropy?
   ```
3. 写 `docs/current_residual_trainer_audit.md`。

## 必须测试

```text
test_residual_trainer_does_not_call_ppo_before_fix
test_graph_mappo_ppo_existing_but_not_residual_path
```

## 退出条件

- 明确“PPO exists”与“PPO active in residual path”的差异；
- Q14 结论不再被误解成 PPO residual failure。

---

# R1：标准化与logit饱和修复

## 目标

让时序信号能穿过 actor head。

## 工作项

1. feature standardization 改为使用所有 train frames。
2. csi_age / mask / probability 使用固定物理归一化。
3. 记录 saturation metrics：
   ```text
   raw_abs_mean
   raw_abs_p95
   frac_abs_raw_gt_30
   frac_abs_logit_gt_9_5
   recurrent_memoryless_logit_delta
   recurrent_memoryless_action_delta
   ```
4. 新增 raw-logit L2。
5. residual head 与 full topology head 分离，residual logits小范围输出。

## 测试

```text
test_standardization_uses_all_train_frames
test_stale_feature_scale_not_exploding
test_saturation_metrics_logged
test_raw_logit_l2_reduces_saturation_on_pilot
test_recurrent_changes_logits_under_stale_csi
```

## 退出条件

- saturation rate 大幅下降；
- recurrent 与 memoryless logits不再bit-identical；
- 若仍相同，暂停后续控制实验。

---

# R2：CSI belief prediction auxiliary

## 目标

显式训练GRU用历史恢复当前真实CSI。

## 实现

新增：

```text
belief_head
L_CSI
```

输入：stale / partial CSI history。  
目标：true current link psucc / delivery probability。

## 损失

\[
L_{\text{CSI}}
=
\sum_{t,e}
w_{e,t}
Huber(logit(\tilde p_{e,t})-logit(p_{e,t}^{true}))
\]

## 测试

```text
test_belief_head_training_target_is_true_current_csi
test_actor_input_does_not_include_true_current_csi
test_recurrent_belief_beats_memoryless_under_delay1
test_belief_loss_logged
test_belief_prediction_improves_over_epochs
```

## Pilot

```text
current
delay1
delay2
partial
```

比较：

```text
memoryless belief MSE
recurrent belief MSE
```

## 退出条件

- recurrent 在 stale/partial下显著优于memoryless；
- belief prediction不是只在diagnostic脚本中，而是进入policy训练loss。

---

# R3：Residual PPO / CTDE value critic

## 目标

用PPO clip/KL和value critic替代REINFORCE + scalar baseline。

## 实现

1. rollout保存：
   ```text
   residual_decision
   logp_old
   reward
   value_old
   ```
2. 多epoch训练重算：
   ```text
   logp_new
   value_new
   entropy
   ```
3. 调用：
   ```text
   ppo_clip_actor_loss
   approx_kl
   ```
4. centralized critic：
   ```text
   V(s_t)
   ```
   可读取training-only true current CSI、previous topology、observed CSI、belief features。
5. target KL early stop。

## 测试

```text
test_residual_ppo_calls_ppo_clip_actor_loss
test_residual_ppo_logs_approx_kl
test_residual_critic_parameter_delta_positive
test_residual_ppo_epoch0_ratio_one
test_target_kl_can_stop_inner_epochs
test_entropy_bonus_present
```

## Pilot

比较：

```text
REINFORCE residual
PPO residual
```

在同样 seed / data / anchor 下测试。

## 退出条件

- bimodal clamp/collapse显著减少；
- KL/clip fraction正常；
- critic EV非负或可解释；
- residual edit rate非零但unsafe rate可控。

---

# R4：Beneficial oracle-edit dataset

## 目标

提供“比anchor更好”的方向监督。

## 数据生成

对每个 frame：

```text
anchor topology x_H
candidate edits add/remove/swap
evaluate x_H and x_H⊕edit
compute ΔC, ΔD_quorum, ΔE, ΔL, ΔJ
label beneficial if ΔJ > margin and safety ok
```

不要模仿完整oracle topology，只模仿有正增益的局部edit。

## 测试

```text
test_oracle_edit_dataset_contains_anchor_and_delta
test_only_positive_gain_edits_supervised
test_teacher_does_not_use_held_for_training
test_edit_labels_include_C_D_E_L
test_dataset_artifact_hash_logged
```

## 指标

```text
positive_edit_rate
best_edit_gain_distribution
anchor_failure_repairable_rate
safe_prune_rate
```

## 退出条件

- 数据中存在非零正增益edit；
- 若正增益率极低，停止RL，说明anchor近似局部最优。

---

# R5：Repair / Safety / Edit heads

## 目标

让模型先监督学习“哪些edit有益”。

## 实现

```text
repair_head
safety_head
utility_head
edit_policy_head
```

## 训练

\[
L = L_{\text{CSI}} + L_{\text{repair}} + L_{\text{safety}} + L_{\text{edit}}
\]

## 测试

```text
test_repair_head_predicts_delta_D
test_safety_head_predicts_deletion_risk
test_edit_head_topk_hits_positive_edits
test_supervised_edit_improves_over_random
```

## 验收

```text
top-k beneficial edit hit rate > random baseline
safe remove precision high
repair recall reported
```

若监督失败，不进入PPO。

---

# R6：Evidence-gated residual action

## 目标

把动作空间从全边Bernoulli flip改为证据门控候选编辑。

## 实现

```text
candidate_adds = add edges passing repair gate
candidate_removes = anchor edges passing safety gate
candidate_swaps = pairs passing swap gate
```

Policy只在候选集上采样。

## 测试

```text
test_bad_edits_excluded_by_gate
test_zero_candidates_returns_anchor
test_gated_action_budget_safe
test_gated_action_deployable_no_evaluator
test_gate_metrics_logged
```

## 指标

```text
candidate_count
gated_out_count
safe_candidate_precision
policy_edit_rate
```

---

# R7：Adaptive anchor KL / Safety constraint

## 目标

替代固定flip penalty。

## 实现

1. PPO KL controls new-vs-old.
2. Anchor KL controls distance from anchor.
3. Safety constraint controls feasible-anchor retention.

Adaptive schedule：

```text
if retention < target: tighten anchor KL
if retention stable and val improves: loosen anchor KL
```

## 测试

```text
test_anchor_kl_not_equal_flip_count
test_retention_drop_tightens_anchor_kl
test_stable_improvement_relaxes_anchor_kl
test_unsafe_edit_penalty_uses_safety_not_flip_count
```

## 退出条件

- 不再出现retention=0 collapse；
- 也不再出现edit_rate=0 clamp。

---

# R8：Full method pilot

## Arms

```text
anchor
residual PPO only
belief auxiliary only
supervised edit only
belief + supervised edit
belief + supervised edit + residual PPO
```

## 数据

```text
urban delay1
urban current
random delay1
```

## 指标

```text
feasibility
return
C/E/L
retention
repair success
edit rate
unsafe edit rate
CSI MSE
KL
entropy
saturation
```

## 退出条件

- 至少一个 learned arm超过anchor或显著修复stale CSI drop；
- 如果没有，定位是belief / edit / PPO哪一层失败。

---

# R9：Multi-seed research campaign

## 要求

```text
>=5 seeds
train/val/held
urban 4RSU
random ablation
current/delay1/delay2/partial
N={8,12,16}
```

## 报告

```text
per-seed raw
CI
baseline groups
evaluator calls
artifact pointers
scope
```

---

# R10：文档收口

更新：

```text
README.md
CURRENT_HEAD_STATUS.md
URBAN_V2X_RESEARCH_LOG.md
CURRENT_RESIDUAL_PPO_STATUS.md
```

必须明确：

```text
what was active
what was not active
whether residual PPO improved anchor
whether CSI belief made recurrence useful
```

---

# 严禁事项

1. 只把PPO函数放在代码里但residual trainer不调用。
2. 用Graph-MAPPO已有PPO证明residual PPO已启用。
3. 不解决logit saturation就评价recurrence无效。
4. actor看到true current CSI。
5. 模仿完整central oracle topology。
6. 没有beneficial edit数据就做residual PPO。
7. fixed flip penalty作为唯一anchor保护。
8. 只报feasibility不报edit_rate/retention/saturation/CSI_MSE。
9. 用单seed判断。
10. 没有fail-first test就实施。

---

# 每轮工件

```text
experiment_plan.md
failing_tests.txt
mechanism_activation.json
train_history.json
critic_metrics.json
belief_metrics.json
edit_metrics.json
residual_metrics.json
held_metrics.json
decision.md
```
