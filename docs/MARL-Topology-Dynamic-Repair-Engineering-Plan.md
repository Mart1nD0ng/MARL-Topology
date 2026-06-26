# MARL-Topology 动态任务与完整模型修复工程计划书

> 文档类型：工程计划书  
> 当前目标：修复动态实验定义、补全信息结构与训练稳定性、重新测试动态任务与完整模型组件。  
> 适用分支：`decentralized-marl-trunk`  
> 核心原则：先修实验口径，再修训练，再做完整模型对比。不得直接把当前负面结果当作最终结论。

---

# 0. 当前状态概述

动态任务已经完成了重要工程接入：

- \(T>1\) episode rollout；
- moving vehicle frames；
- BCSP per-agent action；
- local mutual acceptance；
- recurrent actor；
- per-frame centralized critic；
- discounted return；
- warm-start实验；
- held traces和机制激活日志。

但当前实验仍有关键缺口：

1. 动态数据并不是4-RSU城市网格，而是单RSU随机几何。
2. `hold_interval`没有进入RL reward。
3. Temporal Value Test与RL训练目标不一致。
4. 训练使用discounted return，eval/keep-best使用undiscounted return。
5. 没有独立validation trajectories。
6. phase-specific PBFT accounting未接入production evaluator。
7. 动态分支没有接入COMA、SCQ、chance/CVaR、Pareto、PNA、vector critic。
8. actor没有速度/航向/相对速度等预测下一帧所需信息。
9. myopic-greedy是central evaluator reference，不是deployable baseline。
10. warm-start teacher与RL objective不一致，PPO会破坏warm-start。

---

# 1. 总体阶段图

```text
D0  冻结当前动态结果与复现实验包
D1  修复动态数据定义：4-RSU城市/道路/建筑/移动
D2  修复动态reward：hold_interval、discount、一致目标
D3  修复validation与reporting：train/val/held三分
D4  接入phase-specific PBFT accounting到production evaluator
D5  修复观测信息：velocity/heading/relative motion/CSI history
D6  修复warm-start与RL训练：decoder-aware imitation + KL/BC anchor
D7  建立公平deployable baselines与central references
D8  重测recurrent vs memoryless
D9  接入dynamic COMA/Q critic
D10 接入dynamic SCQ
D11 接入chance/CVaR/Pareto
D12 接入PNA / recurrent PNA动态对比
D13 多seed、多N、完整动态campaign
D14 文档、报告、README与契约收口
```

D0–D7完成前，不得宣称“完整动态模型失败或成功”。

---

# 2. D0：冻结当前动态结果

## 目标

保存当前负面结果，作为对照基线，防止后续混淆。

## 工作项

1. 保存当前 commit SHA。
2. 复制：
   ```text
   result_save/DYNAMIC_TASK_REPORT.md
   result_save/dynamic_headline.json
   result_save/dynamic_headline_warmstart.json
   result_save/dynamic_data_manifest.json
   result_save/dynamic_temporal_value.json
   result_save/_dyn_headline/*
   ```
3. 为每个run保存：
   ```text
   config
   stdout/stderr
   mechanism_activation
   training_history
   held_traces
   dynamic_result
   ```
4. 写：
   ```text
   result_save/dynamic_baseline_frozen/README.md
   ```

## 退出条件

- 当前 D4/D6 结果可完整复现；
- 当前结果明确标记为：
  ```text
  single-RSU random-geometry dynamic task
  dynamic recurrent-PPO only
  not full Phase 8-11
  ```

---

# 3. D1：修复动态数据定义

## 目标

让动态数据真正匹配目标V2X场景。

## 修改内容

1. 新增或修复 urban dynamic sampler：
   ```text
   sample_dynamic_urban_scenes()
   ```
2. 使用：
   - 4 RSU；
   - urban grid；
   - buildings / road blocks；
   - road-constrained vehicle motion；
   - lane direction / intersections；
   - optional turn probability。
3. 保持：
   - node ids across frames；
   - candidate edge ids stable；
   - frame 0可复现静态场景；
   - per-frame Stage-21 link records真实重建。
4. manifest 必须记录：
   ```text
   rsu_count=4
   urban_grid_config
   building_count
   road_graph
   mobility_model
   speed_distribution
   content_hash
   ```

## 新测试

```text
test_dynamic_urban_has_four_rsus
test_dynamic_urban_uses_grid_and_buildings
test_vehicle_motion_is_road_constrained
test_edge_ids_stable_across_frames
test_frame0_matches_static_urban_context
test_channel_changes_with_motion
```

## 对照实验

保持旧单RSU动态作为 ablation：

```text
dynamic_random_geometry
dynamic_urban_4rsu
```

## 退出条件

- 报告中的数据描述与代码完全一致；
- 4-RSU城市动态数据可重生成；
- 旧单RSU随机几何只作为diagnostic baseline保留。

---

# 4. D2：修复动态reward与Temporal Value Test一致性

## 目标

让训练、Temporal Value Test、validation和held使用同一个动态目标。

## 修改内容

定义：

\[
r_t
=
H_{\mathrm{PBFT}}\,r_{\mathrm{base},t}
-
c_{\mathrm{reconfig},t}
\]

其中：

\[
c_{\mathrm{reconfig},t}
=
\omega_E
\frac{E_{\mathrm{reconfig},t}}{E_{\mathrm{budget}}}
+
\omega_L
\frac{L_{\mathrm{reconfig},t}}{L_{\mathrm{SLA}}}.
\]

若暂时仍使用 scalar `reconfig_e`，也必须明确：

\[
r_t = H r_{\mathrm{base},t} - e_{\mathrm{edge}}|\Delta E_t|.
\]

## 统一return

训练、validation、held、myopic、horizon全部使用：

\[
G_0=\sum_{t=0}^{T-1}\gamma^t r_t.
\]

## 新测试

```text
test_dynamic_reward_multiplies_base_by_hold_interval
test_temporal_value_and_rl_use_same_reward
test_dynamic_eval_uses_discounted_return
test_keep_best_uses_validation_discounted_return
test_reconfig_cost_scale_logged
```

## 参数扫描

对：

```text
hold_interval ∈ {1,4,16,64}
reconfig_e ∈ {0,0.01,0.03,0.1}
```

计算 Temporal Value Test。

## 退出条件

- Temporal Value Test与RL训练是同一数学任务；
- 当前 `hold_interval` 不再只是manifest字段。

---

# 5. D3：修复train/validation/held三分

## 目标

防止用训练轨迹选择checkpoint。

## 修改内容

动态训练生成：

```text
train_scenes
val_scenes
held_scenes
```

seed公式例如：

```text
train: seed*1000 + 1
val:   seed*1000 + 333
held:  seed*1000 + 777
```

checkpoint选择：

```text
best by validation discounted episode return
secondary by validation feasibility / risk
held only final reporting
```

## 新测试

```text
test_dynamic_train_val_held_seeds_disjoint
test_checkpoint_uses_val_not_train
test_held_not_used_until_final
test_split_manifest_records_all_three
```

## 退出条件

- dynamic_result中明确列出train/val/held；
- held不参与任何选择。

---

# 6. D4：接入phase-specific PBFT accounting

## 目标

让动态reward中的energy/latency来自真实PBFT三阶段消息。

## 修改内容

1. `Stage21ObjectiveStackEvaluator.evaluate()`使用：
   ```text
   build_pbft_message_plan()
   pbft_protocol_energy()
   phase_completion_latency()
   ```
2. pre-prepare只primary-to-backups。
3. prepare/commit只validators vote。
4. clients只relay，不vote。
5. energy breakdown写入metrics。
6. timeout-aware latency使用phase plan。

## 新测试

```text
test_stage21_uses_pbft_message_plan_when_corrected
test_pre_prepare_not_all_pairs
test_clients_do_not_vote_in_stage21
test_energy_breakdown_protocol_relay_control
test_latency_uses_phase_completion_from_plan
```

## 退出条件

- `pbft_message_plan.py`不再只是primitive；
- production evaluator真正接入；
-旧all-pairs-x3路径只legacy opt-in。

---

# 7. D5：补全动态观测信息

## 目标

让 memoryless actor 有机会成为真正Markov，让 recurrent actor有真实历史预测信号。

## Actor可观测新增

节点：

```text
velocity_x
velocity_y
speed
heading_sin
heading_cos
```

边：

```text
relative_velocity_along_link
distance_delta_estimate
link_success_delta
latency_delta
energy_delta
CSI_age
```

如需保持局部性，使用邻居广播速度即可。

## Critic training-only新增

critic可读：

```text
global positions
global velocities
previous topology
time index
```

不进入部署actor。

## 新测试

```text
test_dynamic_actor_observation_contains_velocity
test_relative_velocity_feature_changes_sign
test_memoryless_with_velocity_can_distinguish_approach_vs_depart
test_critic_can_read_training_only_velocity
test_actor_forbidden_global_fields_still_absent
```

## 消融

```text
A current CSI only, memoryless
B current CSI + velocity, memoryless
C current CSI only, recurrent
D current CSI + velocity, recurrent
```

## 退出条件

- “Markov”结论必须基于B组结果，而不是当前CSI猜测；
- recurrent无收益需在A/B/C/D中重新验证。

---

# 8. D6：修复warm-start与RL训练

## 目标

避免PPO破坏可行warm-start。

## 修改内容

### 8.1 Decoder-aware imitation

替代逐边 BCE 或作为对照，使用：

\[
L_{\mathrm{BCSP}}
=
-\sum_i \log \pi_i(S_i^{teacher})
\]

其中 \(S_i^{teacher}\) 是能经local mutual decoder还原teacher topology的节点proposal集合。

### 8.2 Teacher-KL anchor

PPO loss加入：

\[
L
=
L_{\mathrm{PPO}}
+
\lambda_{\mathrm{BC}}L_{\mathrm{BCSP}}
+
\lambda_{\mathrm{KL}}D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{teacher}})
\]

退火：

\[
\lambda_{\mathrm{BC}},\lambda_{\mathrm{KL}}\downarrow 0.
\]

### 8.3 Critic warm-start

用teacher rollouts预训练critic：

\[
V(s_t)\rightarrow G_t^{teacher}.
\]

## 新测试

```text
test_bcsp_warmstart_increases_teacher_subset_logp
test_decoder_aware_teacher_reconstructs_topology
test_ppo_kl_anchor_limits_drift_from_teacher
test_critic_pretrain_tracks_teacher_return
```

## 对照实验

```text
warm-start only
warm-start + PPO
warm-start + PPO + BCSP-BC
warm-start + PPO + KL
warm-start + PPO + BCSP-BC + KL
```

必须报告：

```text
feasibility
discounted episode return
switches
energy
latency
teacher reconstruction
```

## 退出条件

- RL不再系统性降低warm-start held return；
- 若仍降低，必须标记PPO objective/critic问题。

---

# 9. D7：建立公平baseline体系

## 目标

把central oracle/reference和deployable baseline分开。

## Deployable baselines

1. local link threshold；
2. local top-budget positive logits；
3. local hysteresis keep/repair；
4. static R7 actor逐帧执行；
5. memoryless dynamic actor；
6. velocity-aware memoryless actor。

## Central references

1. myopic evaluator-greedy over named candidates；
2. horizon DP over named candidates；
3. strong SA/LNS witness search；
4. exact small-N oracle。

## 报告分组

```text
Deployable policies
Central references
Teachers
Oracles
```

禁止混称。

## 新测试

```text
test_reference_uses_evaluator_and_is_marked_central
test_deployable_baseline_does_not_call_evaluator_for_action
test_baseline_budget_report_records_eval_calls
```

## 退出条件

- learned actor低于central reference不再被解读为“不如简单可部署基线”；
- 至少有一个公平deployable non-learned baseline。

---

# 10. D8：重测recurrent vs memoryless

## 前置条件

D1–D7完成。

## 实验矩阵

```text
dynamic_random_geometry
dynamic_urban_4rsu
```

每组：

```text
memoryless current CSI
memoryless velocity
recurrent current CSI
recurrent velocity
```

配置：

```text
seeds >= 5
train >= 100 trajectories
val >= 50
held >= 50
frames >= 8
updates sufficient for convergence curve
```

## 指标

```text
held discounted return
per-frame feasibility
episode success
switches/frame
energy
latency
timeout rate
train/val/held gap
critic EV
actor KL
warm-start drift
```

## 退出条件

- 得出recurrence是否有价值；
- 明确结果适用数据与参数范围。

---

# 11. D9：接入dynamic COMA/Q critic

## 修改内容

1. Dynamic branch支持：
   ```text
   --counterfactual
   ```
2. critic使用：
   \[
   Q(H_t,\mathbf S_t)
   \]
3. per-agent counterfactual：
   \[
   A_{i,t}=Q(H_t,\mathbf S_t)-E_{\tilde S_i}Q(H_t,\tilde S_i,\mathbf S_{-i})
   \]
4. counterfactuals通过local mutual decoder重组topology。

## 测试

```text
test_dynamic_counterfactual_advantages_not_all_equal
test_dynamic_q_critic_sees_action
test_dynamic_counterfactual_fixes_s_minus_i
test_dynamic_counterfactual_budget_neutral
```

## 对照

```text
dynamic Graph-MAPPO
dynamic COMA
```

## 退出条件

- Q fidelity足够；
- per-agent credit在动态场景被真实测试。

---

# 12. D10：接入dynamic SCQ

## 修改内容

1. 对动态任务执行one-step fork：
   \[
   (s_t,\mathbf S_t)
   \quad vs \quad
   (s_t,\tilde S_i,\mathbf S_{-i})
   \]
2. target:
   \[
   \Delta y_i=
   r_t+\gamma V(s_{t+1})
   -
   (\tilde r_t+\gamma V(\tilde s_{t+1}))
   \]
3. 训练Q difference：
   \[
   \Delta Q_i\approx \Delta y_i
   \]

## 测试

```text
test_dynamic_scq_state_fork_isolation
test_dynamic_scq_exact_delta_nonzero
test_dynamic_scq_loss_enters_critic
test_dynamic_scq_cache_duplicate_topologies
```

## 退出条件

- 动态SCQ不是静态SCQ primitive；
- 其Q fidelity和policy表现均被报告。

---

# 13. D11：接入chance/CVaR/Pareto到动态任务

## 修改内容

1. chance constraint over episode:
   \[
   \Pr(\exists t:C_t<\tau)\le\delta
   \]
   或 frame-average:
   \[
   E_t[\mathbf1(C_t<\tau)]\le\delta.
   \]
2. CVaR over episode shortfall：
   \[
   \mathrm{CVaR}_\alpha\sum_t(\tau-C_t)_+
   \]
3. energy/latency objective：
   \[
   \omega_E\tilde E+\omega_L\tilde L.
   \]
4. dynamic Pareto archive以validation episode metrics选择checkpoint。

## 测试

```text
test_dynamic_chance_dual_up_down
test_dynamic_cvar_metric_matches_bruteforce
test_dynamic_pareto_archive_uses_val
test_dynamic_preference_changes_actions
```

## 退出条件

- 动态任务真正优化可靠性/能耗/时延；
- 不是只优化feasibility。

---

# 14. D12：PNA / recurrent PNA 动态对比

## 前置

D1–D11完成。

## 对比

```text
MLP memoryless
MLP recurrent
PNA memoryless
PNA recurrent
PNA recurrent + velocity
```

## 要求

- 同样数据；
- 同样warm-start；
- 同样critic；
- 同样decoder；
- 同样eval budget；
- 同样seed。

## 指标

除性能外必须报告：

```text
parameter count
runtime
memory
communication graph
effective message degree
OOD N
```

## 退出条件

- PNA是否适合动态和大N有明确结论。

---

# 15. D13：完整动态campaign

## 配置

```text
N train: 8,12,16
N held: 8,12,16,24,32
seeds >= 5, ideally 10
urban 4RSU
frames >= 8
multiple e_edge
multiple hold_interval
multiple mobility speeds
```

## 报告

```text
per-seed raw values
CI
per-N curves
per-frame curves
train/val/held
deployable baselines
central references
runtime/evaluator calls
failure cases
```

## headline必须写清

```text
tested mechanisms
default-off mechanisms
not tested mechanisms
scope
```

---

# 16. D14：文档和发布

必须更新：

```text
README.md
CURRENT_HEAD_STATUS.md
URBAN_V2X_RESEARCH_LOG.md
TRUNK_MAP.md
AGENTS.md
```

删除或更正：

```text
no central critic invariant
critic-free trunk identity
4RSU dynamic claim if not true
full model claim if mechanisms off
```

---

# 17. 最小可执行修复顺序

如果算力有限，优先：

```text
D0 freeze
D2 reward/hold/eval consistency
D3 train/val/held
D5 velocity features
D6 warm-start protection
D7 fair baselines
D8 retest
```

D1城市4RSU和D4 phase accounting可以并行推进，但进入最终headline前必须完成。

---

# 18. 每轮工作流

每轮只改一个变量。

## experiment_plan.md

```text
Hypothesis
Single change
Controlled variables
Required failing tests
Dataset
Seeds
Budget
Expected mechanism activation
Success criterion
Failure criterion
```

## 执行

```text
1. failing test
2. minimal implementation
3. unit tests
4. integration test
5. real-shard smoke
6. mechanism_activation check
7. pilot
8. multi-seed if pilot passes
```

## decision.md

```text
Result
Per-seed
CI
Runtime
Evaluator calls
Mechanism activation
Failure modes
Scope
Keep / Revise / Rollback
Next hypothesis
```

---

# 19. 严禁事项

1. 不修reward一致性就解释recurrence无用。
2. 不修数据生成就宣称urban动态结论。
3. 不接phase accounting就宣称完整能耗/时延优化。
4. 不加velocity就宣称当前CSI已Markov。
5. 不设validation就做headline。
6. 把myopic evaluator reference称为deployable simple baseline。
7. 只报feasibility不报episode return。
8. 只报mean不报per-seed。
9. 只用3 seeds得强结论。
10. dynamic branch没接Phase 8–11却说full model。
11. 用warm-start teacher但不报告teacher/source/leakage。
12. RL破坏warm-start却不测KL/BC保护。
13. 继续使用单RSU随机几何却写4RSU。
14. 继续忽略hold_interval。
15. 将negative result扩大到未测试机制。

---

# 20. 目标完成定义

本计划完成时，项目应能回答：

1. 在真实4RSU城市动态场景中，temporal value是否存在？
2. memoryless + velocity是否足够？
3. recurrent actor是否有额外收益？
4. RL是否能超过imitation warm-start？
5. dynamic COMA / SCQ是否提升credit和泛化？
6. chance/CVaR/Pareto是否在动态任务中真实生效？
7. learned deployable policy能否超过deployable local baselines？
8. 与central references的gap有多大？
9. 失败来自环境不可解、动作不可达、critic误差、训练不稳还是泛化不足？

只有这些问题被实验证据回答后，才能判断动态MARL架构是否真正有效。
