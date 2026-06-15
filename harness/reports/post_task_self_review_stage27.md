# Post-Task Self-Review - Stage 27 Critic Baseline Repair

## Completed Task

Stage 27 repaired the centralized critic baseline before more MAPPO training.
The work separated value-critic and diagnostic-critic semantics, added enriched
pre-action centralized critic features, added train-only return normalization,
collected a larger frozen-policy critic-only dataset, trained enriched MLP and
centralized message-passing graph critic candidates, selected one active future
value critic, generated manifest-validated reports, and updated project state.

## Cybernetic Control Model

- Controlled object: training-only critic subsystem and its coupling to the
  future advantage path.
- Desired state: exactly one usable action-independent centralized value
  critic selected, with raw-scale value predictions after denormalization and
  no actor or policy-gradient update.
- State variables: value feature schema, diagnostic feature schema, train/eval
  critic dataset size, return normalizer state, value loss, explained variance,
  value-return correlation, raw value bias, value scale ratio, advantage
  variance reduction, actor checksum, sampler id, reward/surrogate id, manifest
  validation, registry active critic flag.
- Sensors: Stage 27 critic repair report, generated CSV/plot artifacts, unit
  and contract tests, source scans, run manifest validation, harness task
  validation.
- Actuators: critic feature module, return normalizer, critic dataset builder,
  enriched MLP critic, centralized graph critic, critic-only trainer, model
  registry active flag, report script, tests, harness task, PROJECT_STATE.
- Disturbances: small repeated scenario contexts, noisy returns, reward/objective
  tension from Stage 25, projection/sampler friction, possible critic-only
  leakage into actor input, old weak five-feature critic path.
- Coupling: frozen actor/sampler/assembler/evaluator/reward stack produces
  critic-only rollouts; pre-action state feeds value critic; post-action metrics
  feed diagnostic records only; normalized value loss trains critics; future GAE
  must denormalize selected critic values to raw return scale.
- Feedback loop: Stage 26 critic FAIL -> Stage 27 repair actuators -> critic
  train/eval metrics and boundary tests -> selected active critic or blocked
  closeout.

## Intended Desired State

Critic baseline repair should pass only if at least one critic candidate becomes
usable as a MAPPO value baseline, value scale is aligned with returns, actor
weights remain unchanged, reward weights remain unchanged, and future MAPPO is
owner-gated rather than executed automatically.

## Actual Achieved State

Stage 27 passed. The selected future active critic is
`centralized_message_passing_graph_value_critic_v1`. The old weak
`centralized_mlp_critic_baseline` remains historical and is not active for the
future value baseline. The enriched MLP critic also passed the minimum gate but
was not selected because the graph critic had materially stronger eval quality.

## Evidence

- Train transitions: `2048`
- Eval transitions: `1024`
- Return variance: `87062.5012`
- Return normalizer train mean/std: `-449.6700` / `323.4741`
- Selected graph critic eval explained variance: `0.597247`
- Selected graph critic eval value-return correlation: `0.775182`
- Selected graph critic eval raw value bias: `-6.202177`
- Selected graph critic bias reduction from Stage 26 baseline: `0.981886`
- Selected graph critic value scale ratio: `1.022855`
- Selected graph critic advantage variance reduction: `0.597247`
- Enriched MLP eval explained variance: `0.244510`
- Enriched MLP eval value-return correlation: `0.494499`
- Manifest validation: valid
- Actor update performed: `False`
- Policy-gradient update performed: `False`
- Reward weights changed: `False`
- Sampler switched: `False`
- Checkpoint created: `False`

## Tests And Gates

Passed gates:
- critic semantic split gate
- enriched critic feature gate
- return normalization gate
- critic-only dataset gate
- enriched MLP critic gate
- centralized graph critic gate
- critic pretraining gate
- critic selection gate
- no actor update gate
- no policy-gradient update gate
- no forbidden architecture gate
- manifest report artifact gate

Deferred gates:
- Stage 28 repaired-critic MAPPO rerun evidence
- scale-up readiness
- recurrent actor/LSTM readiness

## Regression Check

Protected non-target behavior:
- deployment actor input remains local and actor-safe;
- active sampler remains `physical_plackett_luce_top_k_sampler`;
- Stage 5 reward/surrogate weights remain frozen;
- no checkpoint path or model persistence was introduced;
- v5 remains read-only and unmodified;
- COMA, Transformer, GRU/LSTM, recurrent PPO, final tau selection, and scale-up
  remain blocked.

## Residual Risk

The critic repair was trained on frozen-policy critic-only data with repeated
source contexts. It demonstrates value-scale and baseline usability, not that
actor training will improve the objective. Stage 28 must test whether the
repaired critic improves the fixed small-scale MAPPO pilot without reliability
or projection regressions.

## Required Stage 27 Answers

1. Did value predictions reach the same scale as returns?
   Yes. The selected graph critic eval value scale ratio was `1.022855`.

2. Did explained variance improve?
   Yes. The selected graph critic reached eval explained variance `0.597247`
   versus the Stage 26 near-zero failed critic.

3. Did value-return correlation become positive and meaningful?
   Yes. The selected graph critic eval value-return correlation was `0.775182`.

4. Did advantage variance improve?
   Yes. The selected graph critic reduced advantage variance by `0.597247`
   versus the batch-mean baseline.

5. Which critic was selected and why?
   `centralized_message_passing_graph_value_critic_v1` was selected because it
   passed the minimum and preferred gates and materially outperformed the
   enriched MLP on eval explained variance, value-return correlation, and
   advantage variance reduction.

6. Was graph critic implemented or honestly blocked?
   Implemented and tested. It uses centralized message passing, graph pooling,
   no Transformer, and remains training-only.

7. Is MAPPO training still blocked or allowed for Stage 28?
   Further training is blocked until owner approval. The recommended Stage 28
   is a fixed small-scale MAPPO rerun with the repaired critic, not scale-up.

8. Were actor weights unchanged?
   Yes. Stage 27 collected frozen-policy data and trained critics only; actor
   update flags are false.

9. Were reward weights unchanged?
   Yes. Stage 5 reward/surrogate weights remain frozen.

10. What is the exact recommended Stage 28?
    `stage_28_rerun_small_scale_mappo_with_repaired_critic`.

## Candidate Next Tasks

- `stage_28_rerun_small_scale_mappo_with_repaired_critic`
- reward/objective diagnostic repair only if Stage 28 shows reward-objective
  regression despite critic repair
- data expansion only if Stage 28 shows overfit or unstable seed behavior

## Recommended Next Task

`stage_28_rerun_small_scale_mappo_with_repaired_critic`

Owner decision required: `true`.
