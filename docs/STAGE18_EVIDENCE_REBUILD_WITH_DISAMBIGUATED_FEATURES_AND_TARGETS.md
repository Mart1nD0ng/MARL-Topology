# Stage 18 Evidence Rebuild With Disambiguated Features And Targets

## Controlled Object

The controlled object is learning evidence alignment: actor-safe local features
must match actor target semantics, while global counterfactual effects remain
critic-only. Stage 18 is implementation-bearing evidence-generation work, not
training.

## Rebuilt Evidence Views

Each Stage 18 row contains four separated views:

- `actor_safe_view`: original local actor fields plus Stage 18 local
  topology/history/resource/projection/message/link-estimate features.
- `actor_target_view`: soft utility targets, pairwise ranking targets, and
  hard-label diagnostics with training-allowed flags.
- `critic_target_view`: global edge-delta, feasibility, consensus, latency,
  energy, surrogate diagnostic, and oracle membership targets marked
  `critic_only`.
- `diagnostics_view`: scenario, topology, fixture, target source, ambiguity
  reasons, contradiction flags, and forbidden field scan results.

## Before / After Result

Before rebuild, Stage 18 reproduces the Stage 17 contradiction result:

- raw sample count: `850`
- signature count: `142`
- contradiction cluster count: `142`
- hard-label contradiction rate: `1.0`

After rebuild:

- rebuilt sample count: `850`
- actor-safe signature count: `548`
- hard-label allowed subset contradiction rate: `0.0`
- samples converted to soft utility target: `850`
- ranking pairs generated: `412`
- samples moved to critic-only: `850`
- high-ambiguity samples downweighted: `850`
- remaining hard-label contradiction clusters: `0`

The improvement comes from removing contradictory hard labels as the default
actor objective, adding local topology/resource context, and making global
counterfactual effects critic-only or soft-distillation-only.

## Remaining Ambiguity

Remaining ambiguity is explicit, not hidden. Dominant reasons include:

- `missing_local_resource_context`
- `missing_previous_topology_or_history`
- `target_depends_on_global_context`
- `hard_label_should_be_soft_or_ranked`
- `oracle_label_not_actor_observable`
- `true_dec_pomdp_ambiguity`

These reasons lower target confidence and block hard-label actor supervision.

## Completion Gate

Passed when:

- Stage 16 raw contradictions are reproduced before rebuild.
- Stage 18 evidence generates actor-safe features.
- Soft utility and ranking targets are available.
- Hard labels are no longer the primary actor target.
- Global counterfactual targets are critic-only.
- Actor-safe view and actor target view pass forbidden-field checks.

## Next-Stage Readiness Gate

Stage 19 supervised actor rerun is allowed only with owner approval and only on
the rebuilt Stage 18 evidence. PPO/MAPPO, COMA, Transformer, scale-up training,
reward-weight tuning, and final tau selection remain blocked.

Recommended Stage 19 model order:

1. MLP
2. GNN
3. GRU on real multi-step subset
4. LSTM only after GRU sanity

## Implementation

Durable logic lives in:

- `src/marl_topology/data/actor_feature_rebuild.py`
- `src/marl_topology/data/disambiguated_targets.py`
- `src/marl_topology/data/learning_evidence_stage18.py`
- `scripts/replay/stage18_evidence_rebuild_report.py`

## Cybernetic Control Summary

This is a software-engineering control summary derived by analogy from the
project workflow; it is not direct control-theory doctrine.

Controlled object identified: Stage 18 controls the evidence rebuild boundary
between actor-safe local inputs, actor targets, critic-only targets, and
diagnostics.

Desired state defined: the target state is measurable by before/after
contradiction metrics, actor-safe forbidden-field checks, soft target coverage,
ranking pair count, and critic-only target separation.

State variables defined: state variables include actor-safe feature fields,
actor soft target confidence, hard-label allowed flags, ranking pairs,
critic-only edge-delta rows, and diagnostics outputs.

Sensors defined: sensors are unit tests, contract tests, the Stage 18 report
script, static forbidden-field checks, full pytest, harness validation, and
rubric scoring.

Actuators defined: safe actuators are durable data modules, report-only replay
script, documentation, harness task, PROJECT_STATE, and tests. Unsafe
actuators remain training execution, checkpoint writes, v5 writes, and model
implementation.

Feedback loop present: baseline Stage 16/17 contradiction evidence is
observed before rebuild, Stage 18 features and targets are changed, and
after-rebuild contradiction and leakage checks are observed.

Verification plan present: verification commands are `python -m pytest -q`,
`python harness\scripts\validate_tasks.py`, and
`python scripts\replay\stage18_evidence_rebuild_report.py`.

Stability risk checked: the main regression risk is leakage of critic-only
global targets into actor views; regression tests protect that boundary.

Observability gap checked: true Dec-POMDP ambiguity remains visible through
ambiguity reasons and confidence weights instead of hidden hard labels.

Controllability checked: Stage 18 controls evidence representation only; it
does not control model parameters or training execution.

Disturbance checked: deterministic fixture limitations and modest real
multi-step sequence volume remain constraints.

Delay or async risk checked: no asynchronous service or CI queue is part of
the evidence builder; reports are rebuilt in memory from current source.

Noise or flakiness checked: no random training signal is used; remaining
noise comes from deterministic fixture coverage limits.

Decoupling checked: actor-safe view, actor target view, critic target view,
and diagnostics view are separated and tested independently.

Reliability or error control checked: negative tests reject forbidden actor
fields and report scripts write no artifacts.

Persistent learning update suggested: Stage 19 should consume Stage 18
evidence through the harness-registered owner gate instead of reviving raw hard
labels.
