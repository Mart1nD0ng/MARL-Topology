# Reward Surrogate Contract

## Stage 5.0 Surrogate Freeze

This contract defines requirements for a future training surrogate only. Stage
5.0 does not implement reward code, does not choose reward weights, and does not
start training.

Boundary shorthand: Stage 5.0 does not implement reward code.

## Relationship To Objective

The objective contract defines evaluation semantics:

```text
consensus_success_probability >= tau_consensus
minimize latency
minimize energy
```

A future reward surrogate is only a training signal that should help policies
learn the objective. It must not replace registered evaluation metrics.

Every future experiment must report objective metrics separately from any
surrogate reward value.

## Future Surrogate Shape

A future surrogate may include:

- smooth constraint-violation penalty for
  `consensus_success_probability < tau_consensus`;
- latency penalty;
- energy penalty;
- optional diagnostic tie-breaker only when explicitly approved;
- scale normalization terms declared before training.

The future surrogate must include a reliability plateau:

```text
if consensus_success_probability >= tau_consensus:
    no additional reliability bonus is granted by default
```

This prevents redundant topology pressure after the reliability constraint is
satisfied.

## Normalization Requirements

Before implementation, a reward proposal must declare:

- reliability penalty scale and clipping behavior;
- latency normalization reference and units;
- energy normalization reference and units;
- how missing, failed, or infeasible communication records are handled;
- whether normalization is scenario-local, batch-local, or fixed by a
  calibration set;
- tests showing that scaling does not hide constraint violations.

Stage 4.8 is a boundary audit, not a calibration set. Its diagnostic threshold
must not be reused as `tau_consensus` or as a reward scale.

Stage 5.0a defines a tau-consensus calibration plan. That plan is still not a
reward implementation and still does not choose reward scales.

Stage 5.0c defines the tau calibration report design. The report design is
still not a reward implementation and must not use shaped reward values as tau
selection evidence.

Stage 5.0d implements a report-only tau feasibility sensor from that design.
It requires explicit candidate tau inputs and still does not choose
`tau_consensus`, implement reward, calibrate reward weights, or train models.

Stage 5.0e defines the fixture-family coverage needed before a calibration run.
It is still not a reward implementation and does not make fixture labels,
topology diagnostics, or oracle candidates into reward terms.

Stage 5.0f implements a minimal alpha fixture suite. The suite is still
evaluation evidence only; fixture labels, topology diagnostics, and
per-primary reliability diagnostics are not reward terms.

Stage 5.0m reviews whether the objective side is ready for Stage 5.1. Its
decision is plan-only: Stage 5.1 may design a reward implementation plan
without code. It does not permit reward implementation, reward weights,
training, actor/critic/model work, final tau selection, or v5 migration.

Stage 5.1 defines the future reward-surrogate implementation plan in
`docs/STAGE5_1_REWARD_IMPLEMENTATION_PLAN.md`. The plan keeps reward as a
training surrogate, requires reliability plateau above tau, requires explicit
latency and energy normalization references before code, and still does not
implement reward, select weights, add replay reward columns, or run training.

Stage 5.2 implements only the minimal pure interface skeleton described in
`docs/STAGE5_2_REWARD_SURROGATE_INTERFACE.md` and
`src/marl_topology/objectives/surrogate_signal.py`. The interface consumes only
registered evaluation concepts plus declared diagnostics, returns decomposed
training-only surrogate diagnostics, and admits those diagnostics only through
the replay-column boundary. Stage 5.2 does not train, calibrate weights, select
final `tau_consensus`, add actor/critic/model code, or migrate v5 code.

Stage 5.3 Reward Normalization Reference Selection selects fixed latency and
energy normalization references in
`docs/STAGE5_3_REWARD_NORMALIZATION_REFERENCE_SELECTION.md` and
`src/marl_topology/objectives/normalization.py`. The default source is the
Stage 5.0l Stage 3-backed range review, and the deterministic policy is
`feasible_positive_max_v1`. Stage 5.3 does not calibrate reward weights, train,
select final `tau_consensus`, add model code, or migrate v5 code.

Stage 5.4 Reward Report Integration Without Training attaches Stage 5.2
surrogate decomposition and Stage 5.3 normalization references to a report in
`docs/STAGE5_4_REWARD_REPORT_INTEGRATION.md` and
`src/marl_topology/evaluation/surrogate_diagnostics_report.py`. Its policy is
`component_only_no_scalar_reward_v1`: component diagnostics are reported, but
no scalar surrogate is reported as an evaluation metric. Stage 5.4 does not
calibrate reward weights, train, select final `tau_consensus`, add model code,
or migrate v5 code.

## Forbidden Surrogate Components

The future surrogate must not introduce:

- standalone timeout reward;
- standalone quorum reward;
- standalone deadline reward;
- standalone density reward;
- standalone edge-count reward;
- reliability bonus above `tau_consensus`;
- full-mask or full-graph bonus;
- oracle-candidate labels as training targets unless a future imitation task is
  separately contracted;
- unregistered metrics;
- legacy v5 reward formulas, weights, aliases, or phase scripts.

Timeout, quorum, and deadline are protocol or communication conditions. They
may affect registered reliability, latency, or energy through Stage 3/4 records,
but they are not standalone reward components.

## Implementation Gate

Reward implementation remains blocked beyond the Stage 5.2 pure interface
skeleton until a future owner-approved task provides or executes:

- formal `tau_consensus` value or selection process;
- scenario calibration evidence or owner decision;
- reward scale and normalization contract;
- reward-hacking tests;
- Dec-POMDP leakage checks for reward, oracle, and metric fields;
- replay dataset column rules for reward/return fields;
- baseline/oracle comparison evidence under the chosen scenario regime.

Stage 5.1 satisfies the plan requirement only. Stage 5.2 satisfies the pure
interface-skeleton requirement only. Neither stage satisfies reward weight
calibration, report integration, return/advantage/value-target contracts, or
training readiness. Stage 5.3 satisfies fixed normalization-reference selection
only and still does not satisfy weight calibration or training readiness. Stage
5.4 satisfies report integration only and still does not satisfy training
readiness.

Stage 5.5 Training Preflight Review Without Training is recorded in
`docs/STAGE5_5_TRAINING_PREFLIGHT_REVIEW.md` and
`src/marl_topology/evaluation/training_preflight.py`. Its verdict is
`not_ready_for_training_execution`. It permits only a future owner-approved
training design contract as the next recommended task. It does not calibrate
reward weights, train, select final `tau_consensus`, add model code, or migrate
v5 code.

Stage 5.5 keeps these items blocked before training execution:

- surrogate scalarization and weight policy;
- return, advantage, and value-target contract;
- actor and critic architecture contract;
- artifact, seed, and run-manifest policy;
- multi-seed stochastic evidence protocol;
- owner approval for training execution.

Stage 5.6 Training Design Contract Without Execution freezes the design-only
scalarization policy in `docs/STAGE5_6_TRAINING_DESIGN_CONTRACT.md` and
`src/marl_topology/training/design_contract.py`.

Stage 5.6 policy:

```text
constraint_first_component_policy_v1
```

Stage 5.6 still selects no weights and still runs no training. Future
scalarization may use reliability violation, normalized latency, and normalized
energy components only after a later owner-approved task supplies explicit
weights and tests. The reliability plateau above tau remains mandatory.

## Acceptance

- Stage 5.0 adds no reward module and no training loop.
- Future reward is labeled a training surrogate, not the evaluation objective.
- Reliability plateaus above `tau_consensus` unless a later registered safety
  metric justifies margin optimization.
- Latency and energy penalties must be normalized before implementation.

## Stage 31 activation: feasibility-first barrier structure (owner-approved)

Stages 26-30 found that the original `flat_weighted_sum_v1` surrogate cannot
represent the lexicographic objective when most samples are infeasible: reward
could worsen while latency and energy improved (the B1 blocker). Under the
Stage 31 owner decision (`docs/STAGE31_OWNER_DECISION_AND_UNFREEZE.md`) the
surrogate gains a second, owner-approved scalarization structure:

```text
feasibility_first_barrier_v2
```

implemented in `src/marl_topology/objectives/surrogate_signal.py` and
recalibrated in `src/marl_topology/data/stage31_surrogate_recalibration.py`.

Definition (tau fixed at 0.9):

- Feasible (`psucc >= tau`): the reliability term plateaus to zero; cost is
  `latency_weight * norm_latency + energy_weight * norm_energy` only.
- Infeasible (`psucc < tau`): cost is `feasibility_barrier +
  reliability_weight * (tau - psucc)^power`, with no latency/energy term, so the
  only way to improve reward while infeasible is to raise `psucc` toward tau.
- `feasibility_barrier = feasibility_margin + (latency_weight + energy_weight) *
  clip_max`, which guarantees every feasible signal strictly exceeds every
  infeasible signal (a hard feasibility gate).
- The decomposition invariant `training_signal_value == -(reliability +
  latency + energy penalties)` is preserved.

Properties (tested in `tests/unit/test_stage31_feasibility_first_surrogate.py`):
feasibility-first ordering, reliability plateau above tau, within-feasible
latency-then-energy preference, within-infeasible psucc-only gradient, and a
reduced objective/reward inversion rate on the real Stage 31 gradient data
(from ~0.11 for the flat sum to <0.03). References are recalibrated on the
feasible Stage 31 scenarios via `feasible_positive_max_v1`. tau is not lowered;
the flat structure remains the default for back-compatibility.
