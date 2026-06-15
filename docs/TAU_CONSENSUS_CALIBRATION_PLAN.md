# Tau Consensus Calibration Plan

## Stage 5.0a Scope

Stage 5.0a defines the calibration plan for `tau_consensus`. It does not choose
the final threshold, does not implement reward, does not tune reward weights,
does not train models, and does not migrate v5 code.

Boundary shorthand: Stage 5.0a does not choose the final threshold.

## Controlled Object

The controlled object is the decision process that will later turn the formal
objective parameter:

```text
tau_consensus
```

into an owner-approved reliability threshold for:

```text
consensus_success_probability >= tau_consensus
```

This is a calibration and decision workflow, not a metric, not a reward, and
not a protocol implementation.

## Desired State

The project should be able to select `tau_consensus` from evidence without
semantic drift:

- reliability remains a constraint;
- latency and energy remain objectives after feasibility;
- Stage 4.8's diagnostic `reliability_threshold = 0.2` remains non-authoritative;
- PBFT reliability remains analytic expected-initiator reliability;
- full graph remains a baseline, not an oracle;
- threshold choice records scenario regime, protocol configuration, and owner
  decision.

## Calibration Inputs

Required inputs for a future calibration run:

- registered `consensus_success_probability`;
- registered `latency`;
- registered `energy`;
- `topology_diagnostics` including edge count, per-primary reliability,
  retransmission attempts, full-graph-baseline flag, and oracle-candidate flag;
- Stage 3 communication regime id, including `urlcc_finite_blocklength_v1`;
- Stage 4 PBFT model id, including `pbft_expected_initiator_mean_field_v1`;
- protocol committee size, fault tolerance, phase budgets, and accounting mode;
- scenario identifiers and deterministic seed policy when randomness is used.

No unregistered metric may enter the tau selection table.

## Required Scenario Coverage

A future calibration report must cover scenario families, not only the Stage
4.8 boundary audit:

- clear free-space communication;
- near-threshold communication;
- blocked or NLoS urban communication;
- high-RSU or geometry-improved communication;
- same-resource interference;
- orthogonal-resource mitigation;
- sparse topology candidates;
- dense/full graph baseline candidates;
- oracle-candidate review rows that remain non-deployment diagnostics;
- failed scheduled-message cases with visible latency and energy.

Stage 4.8 can be used as a smoke-test sensor only. It is not a calibration set.

## Candidate Tau Sweep

A future calibration report should evaluate a declared candidate grid or owner
candidate list such as:

```text
tau_candidates = owner_declared_values
```

Stage 5.0a intentionally does not provide default numeric tau candidates.

For each candidate `tau`, the report must show:

- feasible topology count;
- infeasible topology count;
- feasible non-full topology count;
- whether full graph is the only feasible baseline;
- lowest feasible latency;
- lowest feasible energy;
- Pareto frontier summary for feasible topologies;
- per-primary weak-link diagnostics;
- scenario-family pass/fail summary;
- cases where reliability feasibility conflicts with latency or energy.

## Selection Criteria

`tau_consensus` may be selected only after the calibration report and owner
decision state:

- acceptable reliability risk class;
- active scenario family set;
- active physics and protocol regimes;
- whether feasibility must hold in every scenario family or by declared
  aggregate rule;
- whether full graph alone being feasible is acceptable;
- whether sparse feasible alternatives exist;
- latency/energy trade-off policy after feasibility.

The selected tau must be a scalar in `[0, 1]`.

## Forbidden Shortcuts

The calibration process must not:

- copy Stage 4.8 `reliability_threshold = 0.2` as `tau_consensus`;
- choose tau from v5 reward weights, old aliases, or phase scripts;
- use `P_eff`, hard/soft legacy naming, or unregistered effective-success
  metrics;
- label full graph as oracle;
- use policy failure as infeasibility proof;
- introduce reward implementation;
- run training;
- implement actor, critic, COMA, GNN, or LSTM code;
- use reward improvement as evidence of threshold validity.

## Required Calibration Report Fields

A future report should include:

- `scenario_set_id`
- `scenario_family`
- `physics_regime`
- `protocol_model_id`
- `accounting_model_id`
- `tau_candidate`
- `topology_name`
- `is_full_graph_baseline`
- `is_oracle_candidate`
- `consensus_success_probability`
- `latency`
- `energy`
- `reliability_feasible`
- `per_primary_reliability`
- `diagnostic_flags`
- `owner_decision_status`

All metric columns must map to the metric governance contract.

## Acceptance For Stage 5.0a

- The plan exists and does not select a final tau.
- Stage 4.8 threshold is explicitly non-authoritative.
- Calibration inputs use registered metric concepts only.
- Future calibration requires scenario coverage and owner decision.
- Reward implementation, weight calibration, training, actor/critic code, and
  v5 migration remain blocked.

## Stage 5.0c Report Design

Stage 5.0c is documented in:

`docs/TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md`

It designs the future calibration report schema, fixture families, output
tables, diagnostics, and validation gates. It still does not run calibration,
does not provide default numeric tau candidates, and does not select the final
`tau_consensus`.

## Stage 5.0d Report Implementation

Stage 5.0d is documented in:

`docs/STAGE5_0D_TAU_CONSENSUS_CALIBRATION_REPORT.md`

It implements a print-only report sensor that requires owner-declared candidate
tau values and uses Stage 4.8 rows as smoke-test input only. It does not select
`tau_consensus`, does not implement reward, does not train, and does not migrate
v5 code. Final threshold selection still requires scenario-family calibration
coverage and owner decision.

## Stage 5.0e Fixture Family Design

Stage 5.0e is documented in:

`docs/STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md`

It defines the fixture-family coverage required before a calibration run:
geometry/visibility, URLLC link regimes, deadline/retransmission, network
resource interference, sparse/dense topology trade-offs, and PBFT primary
asymmetry. It does not implement fixtures, run calibration, provide default tau
candidates, or select `tau_consensus`.

## Stage 5.0f Fixture Suite Implementation

Stage 5.0f is documented in:

`docs/STAGE5_0F_TAU_CONSENSUS_FIXTURE_IMPLEMENTATION.md`

It implements a minimal executable alpha fixture suite and extends the Stage
5.0d report builder so the report can consume Stage 5.0f fixture rows. It still
does not run a tau-selection decision, does not provide default tau candidates,
does not implement reward, and does not train models.

## Residual Risks

- Calibration scenarios now have a minimal alpha suite, but not yet a large
  calibrated city distribution.
- The project still needs an owner decision about acceptable risk class.
- The expected-initiator PBFT model remains a mean-field approximation.
- Latency/energy scalarization remains deferred until after tau selection.
