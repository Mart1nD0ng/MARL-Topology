# MARL-Topology Development Contract v4：Claim-Path 证据契约

> 文档类型：通用开发契约  
> 目的：在 v3 “完整接入、实验严谨性与反退化规范”基础上，进一步防止“机制存在但未进入关键路径”“A 路径有机制却用 B 路径结果断言”“结论超出实际激活范围”等问题。  
> 适用范围：MARL-Topology 后续所有开发任务。  
> 核心变化：v3 更像任务清单；v4 将契约抽象成**每条结论必须绑定具体代码路径、配置、loss、指标和artifact**的 Claim-Path 证据制度。

---

# 0. 从 v3 到 v4 的原因

v3 已经规定：

```text
组件实现 ≠ 正式接入 ≠ run中激活 ≠ 科研验证
```

并要求机制激活、训练/验证一致、baseline分组、五类测试、每轮工件等。

但后续仍出现了新问题：

```text
Q4 stale CSI 只在感知层验证，Q12未接入控制闭环却给出时间机制结论；
Graph-MAPPO中存在PPO clip/KL，但residual trainer未调用，却容易被误认为“项目已有PPO”；
残差策略被anchor-reg夹死，Q12引用“residual==anchor”后容易被误读为“残差学习无效”；
Q14显示GRU模块存在但被logit饱和消掉，普通forward测试无法发现“对最终动作没有影响”。
```

因此 v4 增加三类强制证据：

```text
Claim Card
Load-Bearing Path Test
Effect-on-Decision Test
```

---

# 1. Claim Card：每条结论必须有证据卡

任何报告、README、日志中的关键结论必须附 Claim Card：

```yaml
claim:
scope:
mechanism:
code_path:
entrypoint:
config_flags:
dataset:
seeds:
split:
loss_uses_mechanism:
eval_uses_mechanism:
runtime_activation_artifact:
load_bearing_tests:
effect_on_decision_metrics:
raw_result_artifact:
conclusion_boundary:
```

没有 Claim Card 的结论不得进入 headline。

## 示例

错误：

```text
stale CSI 对控制无效。
```

正确：

```yaml
claim: stale CSI residual policy did not recover anchor loss
scope: Q14 eval-only residual joint, N<=16, 5 seeds
mechanism: stale CSI observation + residual REINFORCE
entrypoint: scripts/diagnostics/stale_csi_residual_joint.py
config_flags: delay1/delay2/partial, residual_prior=-1, anchor_reg=0
loss_uses_mechanism: yes, through observed ef
eval_uses_mechanism: true C/E/L uses current channel
not_active: residual PPO, value critic, CSI auxiliary
conclusion_boundary: does not prove residual PPO or belief-augmented policy failure
```

---

# 2. Mechanism-Path Matrix

每个机制必须声明它在哪些路径中 active：

```text
mechanism
static_trunk
dynamic_trunk
residual_trainer
diagnostic_script
evaluation_script
deployment_path
```

状态取值：

```text
NOT_PRESENT
IMPLEMENTED_ONLY
CALLABLE
ACTIVE_IN_LOSS
ACTIVE_IN_EVAL
ACTIVE_IN_DEPLOY
```

示例：

```text
PPO clip
static graph_mappo: ACTIVE_IN_LOSS
dynamic graph_mappo: ACTIVE_IN_LOSS
residual_pbrs_train: NOT_PRESENT
stale_csi_residual_joint: NOT_PRESENT
```

禁止用某路径的 active 证明另一条路径 active。

---

# 3. Load-Bearing Path Test

机制不只要存在，还要证明关键函数在目标路径中被调用。

## 必须写的测试类型

```text
monkeypatch / spy key function called
activation artifact contains flag
loss changes when mechanism coefficient changes
ablation off gives byte-identical path
gradient reaches trainable module
metrics change when mechanism is disabled
```

## 示例

### residual PPO

必须测试：

```text
residual trainer calls ppo_clip_actor_loss
approx_kl logged
target_kl can stop inner epoch
old_logp != new_logp after update
critic_parameter_delta > 0
```

### stale CSI

必须测试：

```text
actor observed ef != true current ef
evaluator context still true current channel
delay1 changes anchor decisions
Q12/Q14 run records csi_mode
```

### CSI auxiliary

必须测试：

```text
belief_loss enters total loss
true current CSI used only as target
actor input does not contain true CSI
recurrent belief MSE < memoryless belief MSE under delay
```

---

# 4. Effect-on-Decision Test

很多模块“forward有变化”，但最终action不变。必须测试机制对决策的影响。

## 必须记录

```text
logit_delta
action_delta
topology_delta
retention
edit_rate
zero_edit_rate
all_edit_rate
saturation_rate
hidden_to_logit_sensitivity
```

## 示例

GRU有效性不能只测：

```text
hidden changes
raw changes
```

必须测：

```text
recurrent vs memoryless logits differ after head
recurrent vs memoryless topology differs
under stale CSI recurrent improves current-CSI prediction
```

如果 hidden变化但 topology bit-identical，结论应写：

```text
module active but behaviorally inert
```

---

# 5. Path-specific Negative Result

负面结果必须写清路径。

禁止：

```text
PPO无效
SCQ无效
时序无效
残差无效
```

正确写法：

```text
REINFORCE residual trainer without PPO/critic/entropy failed under delay1 urban.
```

或：

```text
PNA in residual frame with trust-region equal to anchor, no win at N<=16.
```

---

# 6. Default-Off 禁令升级

若机制 default-off，本轮实验必须显式记录：

```text
default_off
activated_in_this_run
activation_flag
activation_artifact
```

报告中不得把 default-off 机制归入“完整模型”。

---

# 7. No Substitution Rule

不得用替代测试证明目标机制。

示例：

```text
CSI prediction diagnostic 成功 ≠ policy 使用CSI auxiliary
Graph-MAPPO PPO存在 ≠ residual PPO已接入
PBRS telescoping 成功 ≠ PBRS改善policy
oracle repair有效 ≠ deployable repair head有效
```

必须目标路径验证。

---

# 8. No Silent Citation Rule

如果一个阶段引用另一个阶段结果，而不是重新运行，必须写：

```text
cited_from:
source_commit:
source_artifact:
source_seed_list:
why_retraining_not_required:
what_claim_is_supported:
```

禁止：

```text
在Q12报告中把Q9/Q11引用结果写成Q12重新训练结果。
```

---

# 9. Saturation / Degeneracy Contract

任何离散动作策略必须报告退化指标：

```text
entropy
logit_saturation_rate
zero_action_rate
all_action_rate
edit_rate
retention
KL_to_old
KL_to_anchor
clip_fraction
```

若策略与baseline拓扑完全相同，必须标记：

```text
behaviorally_equal_to_baseline
```

不得宣称“训练策略有效探索”。

---

# 10. Teacher / Oracle Contract

teacher只能作为训练辅助，不得成为部署依赖。

必须区分：

```text
central oracle topology
central beneficial edit
deployable heuristic anchor
supervised target
runtime policy
```

禁止：

```text
直接模仿完整central oracle topology并声称可部署策略超越oracle。
```

允许：

```text
只模仿相对anchor有正增益的local edits，并退火teacher loss。
```

必须记录：

```text
teacher_uses_evaluator
teacher_uses_held
teacher_label_type
teacher_loss_weight_schedule
teacher_loss_active_in_final
```

---

# 11. Optimization Mechanism Contract

若声称使用PPO、critic、entropy、KL，必须记录：

```text
old_logp
new_logp
ratio
clip_fraction
approx_kl
target_kl
value_loss
critic_parameter_delta
entropy
entropy_coef
```

若某路径只用REINFORCE，必须写：

```text
REINFORCE only
no PPO
no critic
no KL
```

不得省略。

---

# 12. Proxy Reward Contract

任何proxy、shaping或auxiliary必须区分：

```text
training reward
auxiliary loss
diagnostic metric
evaluation metric
```

必须记录：

```text
proxy_alignment_test
where_proxy_enters_loss
whether_proxy_enters_eval
```

最终评估不得用proxy替代真实PBFT C/E/L。

---

# 13. Computation Contract

所有涉及 evaluator 的机制必须记录：

```text
action_time_evaluator_calls
training_time_evaluator_calls
SCQ_calls
oracle_repair_calls
cache_hit_rate
wall_clock
num_workers
memory_peak
```

不得将调用 evaluator 的central reference称为deployable baseline。

---

# 14. Report Language Rules

报告必须使用范围限定语言：

```text
This result covers...
This result does not test...
Mechanisms active...
Mechanisms inactive...
```

禁止绝对化：

```text
MARL failed
temporal modeling useless
PPO failed
SCQ failed
oracle imitation works
```

除非有对应 Claim Card 和完整验证。

---

# 15. Definition of Done v4

一个阶段完成需要：

```text
Claim Card written
Mechanism-Path Matrix updated
Load-Bearing Path Tests passed
Effect-on-Decision metrics reported
Real-shard smoke passed
At least pilot result with activation artifact
Decision.md written
Scope boundaries explicit
```

研究结论还需要：

```text
>=5 seeds
train/val/held
CI
raw per-seed artifact
budget report
baseline grouping
```

---

# 16. 推荐目录

```text
docs/contracts/
  DEVELOPMENT_CONTRACT_V4.md

docs/<stage>/
  experiment_plan.md
  claim_cards.yaml
  mechanism_path_matrix.md
  load_bearing_tests.md
  decision.md

result_save/<stage>/
  mechanism_activation.json
  raw_results.json
  metrics.json
  stdout.txt
  stderr.txt
```

---

# 17. v4 的最终目标

让每个开发结论都回答：

```text
这个机制是否存在？
它在哪条路径激活？
它是否进入loss？
它是否改变最终action？
它是否改变评估指标？
它的失败是否只属于该路径？
```

如果不能回答，就不得形成结论。
