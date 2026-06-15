# Metric Governance

This document is a governance contract, not a catalog of final metrics. At Stage 0.2 the project only keeps the minimum concepts needed to express the goal. New metrics must be registered before they appear in code, CSV output, summaries, plots, or reward definitions.

## Current Minimal Concepts

`consensus_success`

- Binary result for one consensus attempt under the current protocol contract.
- Range: `0` or `1`.
- Level: round or topology evaluation.
- Used for: reliability constraint checks and oracle evaluation.

`consensus_success_probability`

- Estimated or analytic probability that consensus succeeds under a declared topology, link model, and protocol setting.
- Range: `[0, 1]`.
- Level: link-set, round, scenario, or batch as declared by the metric registration.
- Used for: reliability constraint design, evaluation, and later training signals if explicitly registered.

`latency`

- Time cost for the evaluated communication or consensus process.
- Unit: seconds unless a registration declares another unit.
- Level: link, round, episode, or batch as declared.
- Used for: optimization objective after reliability is feasible.

`energy`

- Energy cost for the evaluated communication or consensus process.
- Unit: joules unless a registration declares another unit.
- Level: link, node, round, episode, or batch as declared.
- Used for: optimization objective after reliability is feasible.

`topology_diagnostics`

- Non-objective descriptors of a topology, such as edge count, degree distribution, connectivity, sparsity, or feasibility tags.
- Range/unit: field-specific.
- Level: graph, node, edge, scenario, or batch as declared.
- Used for: debugging, oracle explanation, tie-breaking only when explicitly allowed, and failure analysis.

## Metric Registration Rule

Any new metric must be registered before use with:

| Field | Required content |
|---|---|
| `name` | One canonical snake_case name |
| `definition` | Exact semantic meaning |
| `range/unit` | Numeric range, categorical domain, or physical unit |
| `level` | link, node, graph, round, episode, scenario, batch, or another declared level |
| `used_for` | objective, constraint, diagnostic, logging, oracle, training signal, or evaluation |
| `formula source` | contract, paper, implementation note, or derivation source |
| `dependencies` | required upstream quantities and assumptions |
| `tests` | unit, contract, regression, or oracle checks required before use |

The registration belongs in this document or in a dedicated metric registry if this document becomes too large.

## Prohibited Metric Practices

- Same metric with multiple names.
- Same name with multiple meanings.
- Reward and metric sharing a name when they are not the same quantity.
- Unregistered metrics entering CSV, summary output, plots, dashboards, or reports.
- Diagnostic metrics presented as the primary objective.
- Core metrics privately reimplemented in scripts instead of a shared module.
- Legacy metric names imported from `v5` without registration and tests.

## CSV And Summary Rule

Every CSV or summary field that represents a metric must map to a registered metric name. Diagnostic fields must be labeled as diagnostics and must not be used as success evidence unless a task contract promotes them.

Minimum recommended identifiers for future outputs:

- `scenario_id`
- `topology_id`
- `round_id` or `episode_id` when applicable
- `metric_name`
- `metric_level`
- `metric_value`
- `used_for`

Wide CSV formats are allowed only when every metric column is registered.

## Current Acceptance

- The project can express the core goal using only `consensus_success`, `consensus_success_probability`, `latency`, `energy`, and `topology_diagnostics`.
- More complex metrics are deferred until a module or experiment genuinely needs them.
- Any future `P_eff`, soft surrogate, hard evaluation, tail-risk, geometric, or aggregate metric must be registered before use.

## Stage 2 Code Registry

The Stage 2 skeleton mirrors the current minimal concepts in `src/marl_topology/metrics/registry.py`.

Registered code-level metric names:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Topology evaluator outputs must pass the code registry before metric rows are exported. `topology_diagnostics` remains diagnostic and must not be treated as the primary objective.

## Stage 2.6 Replay Column Rule

Replay or dataset rows may include registered metric names only as evaluation-only columns. They must not enter deployment actor input projections.

The active replay column contract is `docs/REPLAY_DATASET_CONTRACT.md` and the code boundary is `src/marl_topology/data/replay_schema.py`.

## Stage 4.0 PBFT Planning Note

Stage 4.0 does not add a new metric name. The planned PBFT/application
reliability formula maps to the existing concept:

- `consensus_success_probability`

Before implementation, the formula must register these details in the task or
implementation note:

- named protocol variant;
- round or topology-evaluation level;
- analytic formula source;
- Stage 3 message-delivery matrix dependencies;
- `n`, `f`, quorum, and phase deadline assumptions;
- tests for heterogeneous probability monotonicity, quorum boundaries,
  deadline gating, and no sampling or subset enumeration.

Stage 4 must not export phase message delivery, quorum count, timeout state, or
network delivery as standalone project metrics unless they are registered first.

## Stage 4.2 PBFT Reliability Mapping

## Stage 4.2 / Stage 4.4 PBFT Reliability Mapping

Stage 4.2 implements primary-specific analytic PBFT reliability. Stage 4.4
implements topology-level expected initiator reliability. Both use the existing
registered concept:

- `consensus_success_probability`

The implementation does not add metric names. It does not export phase
readiness, prepared probability, committed probability, quorum count, fault
filter mode, latency, energy, or timeout as project metrics. Those fields are
record diagnostics unless a future metric registration promotes them.

The Stage 4.4 value is the uniform average of per-primary reliability values.
The per-primary values and primary distribution are diagnostics unless a future
task registers them as metrics.

## Stage 4.3 Message-Matrix Adapter Boundary

Stage 4.3 does not add metric names.

`network_delivery_probability` remains a Stage 3 communication record field.
The Stage 4.3 adapter may use it to populate PBFT phase message matrices, but it
must not export it as `consensus_success_probability`.

Only the Stage 4 PBFT reliability evaluators may produce the existing
`consensus_success_probability` concept after consuming declared matrices.

## Stage 4.5 Baseline And Oracle-Candidate Review

Stage 4.5 does not add metric names.

Its review rows use only registered concepts:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

The oracle-candidate status, full-graph-baseline flag, primary distribution,
phase-budget diagnostics, and search counts are diagnostics, not new metrics.
They must not enter CSV or summaries as metric columns unless registered by a
future task.

## Stage 4.6 Protocol Latency And Energy Accounting

Stage 4.6 does not add metric names.

The protocol accounting module maps its outputs to existing registered
concepts only:

- `latency`: sum of PBFT phase-max latencies clipped to declared phase budgets.
- `energy`: sum of scheduled Stage 3 network attempt energy across PBFT phases.
- `topology_diagnostics`: accounting model id, phase counts, deadline-filtered
  counts, zero-delivery counts, and accounting modes.

The accounting record does not export `consensus_success_probability`.
Reliability remains the responsibility of the Stage 4 PBFT reliability
evaluator. Accounting diagnostics must not be reported as reward components or
new metrics unless a future registration explicitly promotes them.

## Stage 4.7 PBFT Application Evaluation Report

Stage 4.7 does not add metric names.

The application evaluation report flattens registered metric rows from Stage
4.5 and reports only:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

The lowest-latency and lowest-energy feasible topology selections are report
views over existing `latency` and `energy` rows. They are not new metrics,
reward components, or training targets.

## Stage 4.8 Boundary Audit

Stage 4.8 does not add metric names.

The communication/consensus boundary audit reports only the registered
concepts:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

`network_scheduled_latency_s`, `network_successful_delivery_latency_s`,
deadline delivery probability, per-primary reliability, inverse-reliability cap
flags, and full-graph-baseline labels are diagnostics unless a future task
registers them as metrics. They must not be used as reward terms or primary
objectives by name.

## Stage 5.0 Objective / Reward Contract Freeze

Stage 5.0 does not add metric names.

Boundary shorthand: Stage 5.0 adds no new metric names.

The frozen objective uses existing registered concepts only:

- `consensus_success_probability` as the feasibility constraint quantity;
- `latency` as an optimization objective after reliability is feasible;
- `energy` as an optimization objective after reliability is feasible;
- `topology_diagnostics` as diagnostics or explicitly declared tie-breakers.

`protocol_latency` and `protocol_energy` are source quantities mapped to
registered `latency` and `energy`; they are not new metric names in Stage 5.0.

`tau_consensus` is an objective-contract parameter, not a metric name. Stage
4.8's `reliability_threshold = 0.2` remains an audit diagnostic threshold and is
not the formal `tau_consensus`.

## Stage 5.0a Tau Calibration Plan

Stage 5.0a does not add metric names.

The tau calibration plan may produce calibration table fields, but metric-valued
fields must still map to the existing registered concepts:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

`tau_candidate`, `scenario_family`, `owner_decision_status`, and feasibility
flags are calibration diagnostics or identifiers, not new metric names.

## Stage 5.0c Tau Calibration Report Design

Stage 5.0c does not add metric names.

The report design uses registered metric-valued fields only:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

`tau_candidate`, `recommended_tau_candidate`, `owner_selected_tau`,
`full_graph_only_feasible`, `pareto_feasible_count`, and owner decision fields
are report diagnostics or identifiers, not metrics.

## Stage 5.0d Tau Calibration Report Implementation

Stage 5.0d does not add metric names.

The executable report maps metric-valued output fields to:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Candidate tau rows, feasibility booleans, full-graph-only flags, Pareto counts,
and owner-decision packet fields remain diagnostics or identifiers. They must
not enter CSV or summary outputs as unregistered metric names.

## Stage 5.0e Tau Fixture Family Design

Stage 5.0e does not add metric names.

Fixture-family ids, coverage axes, topology variant ids, expected diagnostic
flags, and manifest fields are identifiers or diagnostics. Future fixture
outputs must still map metric-valued quantities to:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

## Stage 5.0f Tau Fixture Suite Implementation

Stage 5.0f does not add metric names.

The executable alpha fixture suite emits metric-valued row fields using the same
registered concepts:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Coverage axes, fixture ids, topology variant ids, per-primary reliability
spread, and diagnostic flags remain diagnostics or identifiers.

## Stage 5.1 Reward Implementation Plan

Stage 5.1 does not add metric names.

The planned future reward surrogate is a training signal, not an evaluation
metric. Future reward fields such as surrogate value, reliability penalty,
latency penalty, energy penalty, or reward config id must not enter metric
outputs unless a later registry explicitly promotes them. They must not enter
deployment actor inputs.

## Stage 5.2 Reward Surrogate Interface Skeleton

Stage 5.2 does not add metric names.

The Stage 5.2 interface consumes only existing registered concepts:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

The emitted reward-surrogate fields are training-only diagnostics, not metrics:

- `reward_surrogate`
- `reward_reliability_penalty`
- `reward_latency_penalty`
- `reward_energy_penalty`
- `reward_config_id`

They must not enter metric CSV rows, summary metric tables, or deployment actor
inputs unless a later owner-approved contract changes their boundary.

## Stage 5.3 Reward Normalization Reference Selection

Stage 5.3 does not add metric names.

The selected reference fields are configuration diagnostics, not metrics:

- `latency_reference_s`
- `energy_reference_j`
- `selection_policy`
- `eligible_row_count`
- `excluded_row_count`

The selector reads metric-valued source fields that already map to registered
concepts:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

No reward-surrogate field, normalization-reference field, or source-stage label
is promoted to an evaluation metric in Stage 5.3.

## Stage 5.4 Reward Report Integration

Stage 5.4 does not add metric names.

The Stage 5.4 report reads existing registered metric-valued inputs:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

The following report fields are training-only diagnostics, not metrics:

- `reliability_violation`
- `reliability_penalty`
- `normalized_latency`
- `normalized_energy`
- `latency_penalty`
- `energy_penalty`
- `component_policy`
- `scalar_surrogate_reported`

They must not enter metric CSV rows, summary metric tables, or deployment actor
inputs unless a later owner-approved metric registration changes that boundary.

## Stage 5.5 Training Preflight Review

Stage 5.5 does not add metric names.

The preflight review consumes the Stage 5.4 component-only diagnostic report as
evidence and reports gate statuses. Gate ids, verdict labels, blocked-task
labels, and owner-decision markers are review diagnostics, not evaluation
metrics.

Metric-valued evidence remains limited to the existing registered concepts:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Stage 5.5 must not promote surrogate component fields, gate ids, pass counts,
blocked counts, deferred counts, return fields, advantage fields, value-target
fields, or training status labels into metric CSV rows or summary metric
tables.

## Stage 5.6 Training Design Contract

Stage 5.6 does not add metric names.

The design contract names future training-only diagnostics such as surrogate
components, constraint violation rate, policy entropy, action sparsity, and
seed variance. These are not project evaluation metrics in Stage 5.6 and must
not enter metric CSV rows or summary metric tables unless a future metric
registration explicitly promotes them.

Registered evaluation evidence remains:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Learning-target fields such as `return`, `advantage`, and `value_target` remain
future replay/training fields, not evaluation metrics.

## Stage 5.7 Policy Architecture Contract

Stage 5.7 does not add metric names.

Architecture option ids, critic-boundary ids, leakage-test ids, serialization
checks, and credit-calibration labels are design diagnostics. They are not
evaluation metrics.

Registered evaluation evidence remains limited to:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

## Stage 5.8 Learning Target Replay Contract

Stage 5.8 does not add metric names.

Planned learning-target columns such as `return`, `advantage`, `value_target`,
`discount_factor`, and `bootstrap_value` are future training fields, not
evaluation metrics. Planned diagnostics such as `policy_entropy`,
`action_sparsity`, and `constraint_violation` are training diagnostics, not
registered metrics.

Metric-valued success evidence remains limited to the existing registered
concepts:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

## Stage 5.9 Run Manifest Artifact Contract

Stage 5.9 does not add metric names.

Run manifest ids, artifact policy ids, seed ids, config ids, code version
markers, contract ids, retention labels, and reproducibility checks are
metadata. They are not evaluation metrics and must not be reported as objective
evidence.

Future artifact reports must keep registered metric values separate from
manifest metadata.

## Stage 5.10 Run Manifest Validator Exit Gate

Stage 5.10 does not add metric names.

Validator issue codes, checked field names, artifact-root labels, stage-closure
flags, and owner-decision markers are control diagnostics, not evaluation
metrics. They must not enter metric CSV rows or summary metric tables.

## Stage 6.0 Minimal Training Stack Guard

Stage 6.0 does not add metric names.

Stack readiness flags, requested-operation labels, blocked-operation labels,
contract marker checks, and manifest issue codes are control diagnostics, not
evaluation metrics. They must not enter metric CSV rows or objective summaries.

## Stage 6.1 Actor-Safe Batch Builder

Stage 6.1 does not add metric names.

Batch ids, row counts, field names, actor ids, time steps, projection sources,
and stage-closure flags are data-boundary diagnostics, not evaluation metrics.
They must not enter metric CSV rows or objective summaries.

## Stage 7.0 Learning Evidence Dataset

Stage 7.0 does not add metric names.

Learning evidence rows may carry the registered evaluation metrics
`consensus_success_probability`, `latency`, `energy`, and
`topology_diagnostics` as row-level evidence and learning-target source data.
Those values must not enter `actor_safe_view`.

Dataset ids, evidence view names, edge-delta target ids, artifact-scope labels,
and writer guard verdicts are data-boundary diagnostics, not registered
evaluation metrics.

Stage 7.1 quality fields such as row counts, target counts, warning counts,
gate names, and readiness flags are control diagnostics. They do not add metric
names and must not be mixed with registered evaluation metrics.

## Stage 8.0 Actor Policy Interface

Stage 8.0 does not add metric names.

Actor policy input fields, output field names, schema ids, interface verdicts,
and leakage-check flags are boundary diagnostics, not registered evaluation
metrics. Objective metrics remain outside deployment actor inputs and may only
be used by future training-only or evaluation-only views after explicit
contract approval.
