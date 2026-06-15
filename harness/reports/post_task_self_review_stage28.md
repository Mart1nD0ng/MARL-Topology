# Post-Task Self-Review - Stage 28 Repaired-Critic MAPPO Rerun

## Completed Task

Stage 28 reran the fixed Stage 25 small-scale MAPPO pilot with the Stage 27
selected repaired graph value critic. The work reused the Stage 25 base
protocol, kept the active actor/sampler/assembler/evaluator/reward stack fixed,
generated manifest-approved training and visualization artifacts, compared the
result against the supervised GNN baseline and the Stage 25 old-critic pilot,
and updated project state for an owner-gated Stage 29 decision review.

## Cybernetic Control Model

- Controlled object: small-scale policy-gradient training loop with repaired
  centralized value baseline and fixed active stack.
- Desired state: the repaired critic is exercised in the value/GAE path under
  the fixed Stage 25 protocol, at least two seeds complete, eval reliability
  stays within the Stage 25 bound, human-readable diagnostics are generated,
  and no scale-up or forbidden architecture is introduced.
- State variables: protocol config id, train/eval split, seed completion,
  supervised baseline metrics, final eval metrics, Stage 25 comparison deltas,
  critic explained variance, value-return correlation, normalized value loss,
  entropy, KL, clip fraction, projection rejection, reward weights, sampler id,
  manifest validity, artifact paths.
- Sensors: Stage 28 training report JSON/CSV/plots, Stage 28 docs, source
  scans, unit and contract tests, run manifest validation, harness validation,
  and PROJECT_STATE synchronization.
- Actuators: Stage 28 repaired-critic pilot module, script entry point,
  reusable Stage 27 graph-critic bundle, report writer, tests, harness task,
  and PROJECT_STATE closeout.
- Disturbances: small train/eval scenario count, reward/objective tension from
  Stage 25, projection friction, stochastic seed differences, slow graph-critic
  evaluation, and risk of accidentally converting a small rerun into scale-up.
- Coupling: fixed actor scores feed the Plackett-Luce sampler and physical-link
  assembler; selected topologies feed Stage 3/4 evaluation and frozen Stage 5
  reward; repaired graph critic estimates values for raw-scale GAE after
  denormalization; report artifacts feed the owner decision loop.
- Feedback loop: Stage 27 critic repair -> Stage 28 fixed rerun -> metrics,
  plots, and comparisons -> PASS/FAIL gate -> Stage 29 recommendation or block.

## Intended Desired State

Stage 28 should determine whether the repaired critic removes the Stage 25
critic failure mode in the same small-scale protocol, without treating this as
scale-up or changing the reward, sampler, actor, evaluator, tau, or
architecture stack.

## Actual Achieved State

Stage 28 passed. All three seeds completed all `20` updates. The repaired
critic remained healthy during the rerun, with mean update explained variance
`0.946565`, mean value-return correlation `0.973309`, and mean normalized value
loss `0.069902`.

Against the supervised baseline, the result was mixed. Eval latency improved by
`0.0000468781` and energy improved by `0.0001375`, while tau-feasible rate
decreased by `0.0234375`, violation rate increased by `0.0234375`, surrogate
reward worsened by `1.865734`, and top proposal rejection worsened by
`0.009549`. The reliability degradation stayed within the allowed `0.05`
absolute bound.

Against the Stage 25 old-critic pilot, Stage 28 improved completed seeds
(`3/3` versus `2/3`), tau-feasible rate by `0.0078125`, violation rate by
`-0.0078125`, and surrogate reward by `0.600415`. Stage 28 was slightly worse
than Stage 25 on latency, energy, and top proposal rejection.

## Evidence

- Stage 28 verdict: `stage28_pass_repaired_critic_small_scale_rerun_complete`
- Completed seeds: `3 / 3`
- Updates per seed: `20`
- Active critic: `centralized_message_passing_graph_value_critic_v1`
- Stage 25 config reuse: `stage25_pilot_base_config`
- Mean normalized value loss: `0.069902`
- Mean update explained variance: `0.946565`
- Mean value-return correlation: `0.973309`
- Supervised eval tau-feasible delta: `-0.0234375`
- Supervised eval violation delta: `0.0234375`
- Supervised eval latency delta: `-0.0000468781`
- Supervised eval energy delta: `-0.0001375`
- Supervised eval surrogate reward delta: `-1.865734`
- Stage 25 comparison tau-feasible delta: `0.0078125`
- Stage 25 comparison violation delta: `-0.0078125`
- Stage 25 comparison surrogate reward delta: `0.600415`
- Manifest artifact root:
  `result_save/stage28_repaired_critic_mappo_rerun/stage28_repaired_critic_stage25_base_protocol`

## Tests And Gates

Passed gates:
- fixed Stage 25 base protocol gate
- repaired critic value path gate
- train/eval split gate
- multi-seed completion gate
- supervised baseline comparison gate
- Stage 25 old-critic comparison gate
- reliability bound gate
- critic health gate
- visualization/report artifact gate
- no checkpoint or scale-up gate
- no reward tuning, sampler switch, COMA, Transformer, GRU/LSTM, or recurrent
  PPO gate

Deferred gates:
- scale-up readiness
- final tau selection
- reward/objective repair decision
- projection/sampler repair decision
- recurrent actor/LSTM readiness

## Regression Check

Protected non-target behavior:
- actor input remains local and actor-safe;
- active sampler remains `physical_plackett_luce_top_k_sampler`;
- Stage 5 reward/surrogate weights remain frozen;
- no checkpoint path or model persistence was introduced;
- v5 remains read-only and unmodified;
- COMA, Transformer, GRU/LSTM, recurrent PPO, final tau selection, and scale-up
  remain blocked.

## Residual Risk

The repaired critic fixed the value-baseline health issue but did not make the
small-scale policy update clearly improve all objective dimensions. The
surrogate reward and projection rejection remain concerns, and the latency/energy
gains came with a small reliability decrease. Stage 29 should decide whether to
repair reward/objective alignment, projection/sampler coupling, data coverage,
or hold further training.

## Required Stage 28 Answers

1. Did Stage 28 use the fixed Stage 25 base protocol?
   Yes. The run reused `stage25_pilot_base_config` and did not tune the base
   hyperparameters.

2. Did the repaired critic get used?
   Yes. Stage 28 used `centralized_message_passing_graph_value_critic_v1` in
   the value/GAE path. Values were denormalized for GAE, while value loss used
   normalized returns.

3. Did all seeds complete?
   Yes. All `3` seeds completed all `20` updates.

4. Did MAPPO improve the supervised GNN on eval?
   Partially. Latency and energy improved, and reliability stayed within the
   allowed degradation bound. Tau-feasible rate, surrogate reward, and
   projection rejection did not improve.

5. Did the repaired critic improve over the Stage 25 old-critic result?
   Yes on seed completion, tau-feasible rate, violation rate, and surrogate
   reward versus Stage 25. No on latency, energy, and top proposal rejection.

6. Was critic health fixed during the rerun?
   Yes. Mean explained variance was `0.946565` and mean value-return
   correlation was `0.973309`.

7. Were reward weights unchanged?
   Yes. Stage 5 reward/surrogate weights remained frozen.

8. Was the sampler unchanged?
   Yes. The active sampler remained
   `physical_plackett_luce_top_k_sampler`.

9. Is scale-up allowed?
   No. Stage 28 is small-scale repaired-critic evidence only. Scale-up remains
   blocked pending owner decision.

10. Is LSTM still blocked?
    Yes. Stage 28 did not identify temporal memory as the primary blocker, and
    recurrent PPO/LSTM remains outside the approved scope.

11. What is the exact recommended Stage 29?
    `stage_29_pre_scale_decision_review`.

## Candidate Next Tasks

- `stage_29_pre_scale_decision_review`
- reward/objective diagnostic repair if Stage 29 prioritizes surrogate mismatch
- assembler/projection repair if Stage 29 prioritizes projection rejection
- data expansion if Stage 29 prioritizes small-scenario uncertainty

## Recommended Next Task

`stage_29_pre_scale_decision_review`

Owner decision required: `true`.
