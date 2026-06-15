# Reward Contract

## Stage 2.8 Reward Contract Review

Stage 2.8 is a review gate only. The project still has no active reward
function, no reward implementation module, no reward weights, no training
surrogate, and no reward column admitted into replay datasets.

In short: no training surrogate is active in Stage 2.8.

The current contract exists to protect future reward work from semantic drift:

- Reliability remains a constraint expressed through registered consensus
  metrics.
- Latency and energy remain the objectives once reliability is feasible.
- Topology diagnostics remain diagnostics unless a future task explicitly
  declares a deterministic tie-breaker after metric objectives are unchanged.
- Any future scalar reward is an implementation proposal, not a metric, and
  must report registered metrics separately.
- No project-wide reliability threshold `tau` is fixed by this review. Future
  reward proposals must declare their threshold and evaluation setting.

This review introduces no new metric names. In particular, no `P_eff`, soft
evaluation, hard evaluation, tail-risk, geometric, arithmetic, or legacy v5
metric default is active.

## Objective Structure

Consensus reliability is a constraint. Latency and energy are objectives.

Default target:

- Satisfy the registered consensus reliability metric, currently `consensus_success` or `consensus_success_probability`, under the evaluation contract.
- Within the reliable region, minimize latency and energy.

## Reliability Constraint

A registered consensus reliability metric is a constraint gate, not a linear reward to maximize indefinitely.

After the consensus reliability threshold is satisfied, reliability enters a plateau unless the task explicitly studies safety margins or tail risk through a registered metric. This prevents wasting reward pressure on already sufficient reliability while latency and energy remain objectives.

## Optimization Terms

Allowed objective terms:

- Latency, with declared aggregation and units.
- Energy, with declared aggregation and units.
- Reliability penalties for failing the registered consensus threshold, when tied to metric-governance fields.

Diagnostic / tie-breaker terms:

- `edge_count`
- `edge_density`
- topology sparsity
- mask retention ratio

These may guide reporting or deterministic tie-breaking only after reliability,
latency, and energy semantics are unchanged. They are not default reward
objectives and must not be used as the main success signal.

## Prohibited Standalone Rewards

- Timeout as standalone reward.
- Quorum as standalone reward.
- `edge_count` as primary reward.
- Undefined `rel` or link-reliability proxy shaping.
- Old `v5` reward copied as the new reward.
- Full graph, full mask, or dense topology as an implicit reward target.
- Oracle labels, future outcomes, or evaluation metrics as deployment actor
  observations.

Timeout and quorum may affect protocol success, latency, or a registered consensus metric through the protocol and metric contracts.

## Pre-Implementation Gates

Reward implementation remains blocked until all of these are available:

- Metric registry mapping for every metric used in evaluation or reporting.
- Protocol mapping that keeps quorum, deadline, timeout, consensus event, and
  consensus probability separate.
- Link-model regime declaration for the evaluated scenario.
- Oracle and non-learning baseline evidence showing that policy failure is not
  mistaken for infeasibility.
- Dec-POMDP leakage tests proving reward, oracle labels, and evaluation metrics
  do not enter deployment actor observations.
- Replay column contract admitting any future reward or return columns as
  training-only, never deployment actor input.

## Required Reward-Hacking Tests

Before a future reward implementation can be accepted, tests must cover:

- Redundant topology pressure: once reliability is feasible, extra edges are
  not rewarded unless latency or energy objectives justify them.
- Constraint violation: unreliable topologies fail the constraint even if they
  have low latency or energy.
- Timeout and quorum aliasing: timeout and quorum do not appear as standalone
  reward terms.
- Diagnostic promotion: edge count, density, sparsity, and full-mask flags do
  not become primary objectives.
- Actor leakage: reward values, oracle labels, future outcomes, and evaluation
  metrics do not appear in `ActorObservation` or deployment actor columns.
- Metric registration: all reported evaluation quantities use registered metric
  names.
- Regime scope: reward evidence states the active physics regime and does not
  claim cross-regime policy quality from the deterministic Stage 2 link model.

## Required Reward Proposal Fields

Every reward change must state:

- Task and controlled object.
- Registered consensus reliability metric definition.
- Any training surrogate definition if one is introduced through metric registration.
- Evaluation metric definition.
- Reliability threshold `tau`.
- Latency objective and aggregation.
- Energy objective and aggregation.
- Excluded proxies.
- Failure modes and reward-hacking tests.
- Verification commands.

## Acceptance

- No training task starts without metric, reward, and evaluation contracts.
- Reward logs separate constraint penalties, latency terms, energy terms, and diagnostics.
- Improvements in shaped reward are not reported as task improvement without registered metric evidence.

## Stage 5.0 Reward / Objective Contract Freeze

Stage 5.0 freezes objective and future reward-surrogate semantics. It still
does not implement a reward function, does not add reward weights, does not
train models, and does not migrate v5 code.

Authoritative Stage 5.0 documents:

- `docs/OBJECTIVE_CONTRACT.md`
- `docs/REWARD_SURROGATE_CONTRACT.md`
- `docs/STAGE5_REWARD_OBJECTIVE_FREEZE.md`

Frozen objective:

```text
feasible if consensus_success_probability >= tau_consensus
then minimize latency and energy among feasible topologies
```

`tau_consensus` is the formal objective-contract parameter. It remains unset
until scenario calibration or owner decision.

Stage 4.8 used `reliability_threshold = 0.2` only as an audit diagnostic
threshold. That value is not the formal `tau_consensus`.

Objective and reward-surrogate boundary:

- The objective is the evaluation target expressed through registered metrics.
- A future reward surrogate is a training signal only.
- Future shaped reward must report registered metrics separately.
- Reliability is a constraint and plateaus after `tau_consensus` by default.
- Latency and energy are the primary optimization objectives.
- `topology_diagnostics` is diagnostic or a declared tie-breaker only.

Forbidden active reward components:

- standalone timeout reward;
- standalone quorum reward;
- standalone deadline reward;
- standalone density reward;
- standalone edge-count reward;
- reliability bonus above `tau_consensus`;
- unregistered metrics;
- old v5 reward formulas, weights, aliases, or phase scripts.

Timeout, quorum, and deadline are already represented by Stage 3/4
communication and protocol contracts. They are not standalone reward terms.

Stage 5.0 leaves reward implementation, reward weight calibration, replay
reward columns, and training blocked.

## Stage 5.0a Tau Consensus Calibration Plan

Stage 5.0a defines the evidence and owner-decision process for selecting
`tau_consensus`.

Authoritative plan:

- `docs/TAU_CONSENSUS_CALIBRATION_PLAN.md`

Stage 5.0a does not set a final `tau_consensus`, does not implement reward,
does not calibrate reward weights, and does not train models.

The Stage 4.8 `reliability_threshold = 0.2` remains a diagnostic threshold only
and must not be copied as `tau_consensus`.

Future tau selection must state:

- scenario set and scenario families;
- active physics regime;
- active PBFT reliability model;
- protocol committee and fault-tolerance settings;
- feasible topology counts for each candidate tau;
- latency and energy trade-offs within the feasible region;
- owner decision status.

## Stage 5.0c Tau Calibration Report Design

Stage 5.0c designs the future calibration report schema.

Authoritative design:

- `docs/TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md`

Stage 5.0c does not run calibration, does not choose `tau_consensus`, does not
implement reward, does not calibrate reward weights, and does not train models.

The future report must keep reward and objective evidence separate: shaped
reward values are not allowed as tau-selection evidence.

## Stage 5.0d Tau Calibration Report Implementation

Stage 5.0d implements the report-only sensor described by Stage 5.0c.

Authoritative implementation note:

- `docs/STAGE5_0D_TAU_CONSENSUS_CALIBRATION_REPORT.md`

Stage 5.0d requires owner-declared candidate tau values and does not choose
`tau_consensus`. It still does not implement reward, does not calibrate reward
weights, does not train models, and does not migrate v5 code. Stage 4.8 rows
remain smoke-test evidence only.

## Stage 5.0e Tau Fixture Family Design

Stage 5.0e defines calibration fixture-family coverage in:

- `docs/STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md`

It does not implement reward, does not calibrate reward weights, does not train,
does not select `tau_consensus`, and does not make topology diagnostics into
default reward terms.

## Stage 5.0f Tau Fixture Suite Implementation

Stage 5.0f implements the minimal alpha fixture suite in:

- `docs/STAGE5_0F_TAU_CONSENSUS_FIXTURE_IMPLEMENTATION.md`

It still does not implement reward, does not calibrate reward weights, does not
train, does not select `tau_consensus`, and does not convert fixture labels or
topology diagnostics into reward terms.

## Stage 5.1 Reward Implementation Plan Without Code

Stage 5.1 defines a future reward-surrogate implementation plan in:

- `docs/STAGE5_1_REWARD_IMPLEMENTATION_PLAN.md`

It does not implement reward, does not select reward weights, does not add
training loops, does not add actor/critic/model code, does not select final
`tau_consensus`, and does not migrate v5 code.

The planned surrogate remains training-only:

```text
reward_surrogate = constraint violation penalty
                 + normalized latency penalty
                 + normalized energy penalty
```

The sign, weights, clipping, and normalization references remain future config
items. Reliability receives a default plateau above tau; no reliability bonus
above tau is allowed unless a future registered safety-margin metric explicitly
permits it.

Stage 5.1 also keeps reward, return, advantage, and value-target fields out of
deployment actor inputs. Future replay columns for reward diagnostics require a
replay-column contract update before use.

## Stage 5.2 Reward Surrogate Interface Skeleton

Stage 5.2 implements the first narrow interface skeleton for the future
training surrogate:

- `docs/STAGE5_2_REWARD_SURROGATE_INTERFACE.md`
- `src/marl_topology/objectives/surrogate_signal.py`

The interface is a pure adapter over registered evaluation concepts:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

It returns a scalar training-side surrogate signal and decomposed diagnostics.
These outputs are not registered evaluation metrics. They are training-only
replay columns and are forbidden deployment actor inputs.

Stage 5.2 preserves the reliability plateau:

```text
if consensus_success_probability >= tau:
    reliability_penalty = 0
    no reliability bonus is added
```

Stage 5.2 does not calibrate reward weights, does not select final
`tau_consensus`, does not implement training, does not add actor/critic/model
code, and does not migrate v5 code.

## Stage 5.3 Reward Normalization Reference Selection

Stage 5.3 selects fixed normalization references for the Stage 5.2 interface:

- `docs/STAGE5_3_REWARD_NORMALIZATION_REFERENCE_SELECTION.md`
- `src/marl_topology/objectives/normalization.py`

Default source evidence:

- Stage 5.0l Stage 3-backed range review.

Selection policy:

```text
feasible_positive_max_v1
```

Selected references:

```text
latency_reference_s = 0.0022698175688954207
energy_reference_j = 0.004134917967719052
```

These references are fixed training-side config diagnostics. They are not
reward weights, not evaluation metrics, not deployment actor inputs, and not
deployment calibration.

Stage 5.3 does not calibrate reward weights, does not select final
`tau_consensus`, does not train, does not add actor/critic/model code, and does
not migrate v5 code.

## Stage 5.4 Reward Report Integration Without Training

Stage 5.4 integrates surrogate decomposition into a report:

- `docs/STAGE5_4_REWARD_REPORT_INTEGRATION.md`
- `src/marl_topology/evaluation/surrogate_diagnostics_report.py`

Policy:

```text
component_only_no_scalar_reward_v1
```

The report includes reliability violation, reliability penalty, normalized
latency, normalized energy, latency penalty, and energy penalty as training-only
diagnostics.

It does not report scalar surrogate, does not treat surrogate components as
evaluation metrics, does not calibrate reward weights, does not train, does not
add actor/critic/model code, and does not migrate v5 code.

## Stage 5.5 Training Preflight Review Without Training

Stage 5.5 reviews readiness for future training execution:

- `docs/STAGE5_5_TRAINING_PREFLIGHT_REVIEW.md`
- `src/marl_topology/evaluation/training_preflight.py`

The verdict is:

```text
not_ready_for_training_execution
```

Training execution remains blocked. Stage 5.5 may recommend only a future
training design contract:

```text
stage_5_6_training_design_contract_without_execution
```

Blocked before training execution:

- surrogate scalarization and reward weight policy;
- return, advantage, and value-target contract;
- actor and critic architecture contract;
- artifact, seed, and run-manifest policy;
- multi-seed stochastic evidence protocol;
- explicit owner approval for training execution.

Stage 5.5 does not calibrate reward weights, does not implement training, does
not add actor/critic/COMA/GNN/LSTM code, does not select final
`tau_consensus`, and does not migrate v5 code.

## Stage 5.6 Training Design Contract Without Execution

Stage 5.6 freezes a design-only future training contract:

- `docs/STAGE5_6_TRAINING_DESIGN_CONTRACT.md`
- `src/marl_topology/training/design_contract.py`

The verdict is:

```text
design_contract_frozen_training_execution_blocked
```

The scalarization policy is:

```text
constraint_first_component_policy_v1
```

Allowed future components remain reliability violation penalty, normalized
latency penalty, and normalized energy penalty. Reliability still plateaus
above tau; no reliability bonus above tau is allowed.

Stage 5.6 selects no reward weights. Future weights require explicit
owner-approved config, contract tests, and report evidence. They must not be
copied from v5.

Stage 5.6 does not run training, does not add actor/critic/COMA/GNN/LSTM code,
does not write checkpoints, does not select final `tau_consensus`, and does
not migrate v5 code.
