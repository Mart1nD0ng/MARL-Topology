# Tau Consensus Calibration Report Design

## Stage 5.0c Scope

Stage 5.0c designs the future tau-consensus calibration report. It does not run
calibration, does not choose `tau_consensus`, does not implement reward, does
not tune reward weights, does not train models, and does not migrate v5 code.

Boundary shorthand: Stage 5.0c does not select the final threshold.
Boundary shorthand: Stage 5.0c does not run calibration.

## Controlled Object

The controlled object is the future report interface that will summarize
scenario evidence for selecting:

```text
tau_consensus
```

The report will consume registered Stage 3/4 evaluation outputs and scenario
diagnostics. It is a decision-support sensor, not a reward implementation, not
a training artifact, and not an actor input source.

Boundary shorthand: not a training artifact.

## Desired State

The report design should make threshold selection auditable:

- every metric-valued field maps to metric governance;
- every candidate tau is owner-declared or explicitly configured;
- Stage 4.8 `reliability_threshold = 0.2` remains diagnostic only;
- full graph remains a baseline, not an oracle;
- oracle-candidate labels remain review-only diagnostics;
- feasibility under tau is separated from latency and energy optimization;
- owner decision remains required before final tau selection.

## Report Inputs

The future report requires these input collections:

### Scenario Set Manifest

Required fields:

- `scenario_set_id`
- `scenario_family`
- `scenario_id`
- `fixture_id`
- `physics_regime`
- `channel_regime`
- `link_transmission_regime`
- `network_regime`
- `pbft_model_id`
- `accounting_model_id`
- `deterministic_seed_policy`
- `owner_scope_note`

### Topology Evaluation Rows

Required fields:

- `scenario_set_id`
- `scenario_family`
- `scenario_id`
- `topology_name`
- `topology_family`
- `selected_edge_count`
- `is_full_graph_baseline`
- `is_oracle_candidate`
- `is_deployment_actor_input`
- `consensus_success_probability`
- `latency`
- `energy`
- `per_primary_reliability`
- `diagnostic_flags`

### Candidate Tau Rows

Required fields:

- `tau_candidate`
- `tau_source`
- `tau_owner_note`
- `is_default`
- `is_selected`

Stage 5.0c does not provide default numeric tau candidates. A future executable
report must receive candidates from owner configuration or a declared
calibration request.

### Protocol And Regime Rows

Required fields:

- `node_count`
- `fault_tolerance`
- `phase_budgets_s`
- `primary_distribution`
- `fault_filter_mode`
- `view_change_mode`
- `mean_field_assumption`
- `urlcc_finite_blocklength_v1` presence flag
- `pbft_expected_initiator_mean_field_v1` presence flag

## Report Output Tables

The future report should emit the following tables.

### Scenario Summary

One row per scenario family:

- `scenario_family`
- `scenario_count`
- `topology_count`
- `candidate_tau_count`
- `physics_regime_set`
- `protocol_model_set`
- `has_failed_scheduled_message_case`
- `has_full_graph_baseline`
- `has_sparse_candidate`
- `has_oracle_candidate_diagnostic`

### Tau Feasibility Summary

One row per `tau_candidate` and scenario family:

- `tau_candidate`
- `scenario_family`
- `feasible_topology_count`
- `infeasible_topology_count`
- `feasible_non_full_topology_count`
- `full_graph_feasible`
- `full_graph_only_feasible`
- `oracle_candidate_feasible`
- `lowest_feasible_latency`
- `lowest_feasible_energy`
- `pareto_feasible_count`
- `weak_primary_case_count`
- `diagnostic_flags`

### Topology By Tau Detail

One row per candidate tau, scenario, and topology:

- `tau_candidate`
- `scenario_id`
- `scenario_family`
- `topology_name`
- `consensus_success_probability`
- `latency`
- `energy`
- `reliability_feasible`
- `is_full_graph_baseline`
- `is_oracle_candidate`
- `is_deployment_actor_input`
- `per_primary_reliability`
- `diagnostic_flags`

### Owner Decision Packet

One row per report:

- `report_id`
- `scenario_set_id`
- `candidate_tau_count`
- `recommended_tau_candidate`
- `recommendation_basis`
- `owner_decision_status`
- `owner_selected_tau`
- `owner_decision_note`
- `blocked_reason`

Stage 5.0c does not populate `recommended_tau_candidate`, `owner_selected_tau`,
or a final decision. The fields exist so a future owner-approved calibration
report can carry a decision packet without changing schema.

## Derived Diagnostics

The report may compute diagnostics from registered fields:

- feasibility boolean for each tau:
  `consensus_success_probability >= tau_candidate`;
- full-graph-only feasibility flag;
- feasible non-full topology count;
- lowest feasible latency and energy views;
- Pareto feasible count;
- weak-primary count from per-primary reliability spread;
- scenario-family pass/fail flags.

These are diagnostics and report views, not new metric names.

## Required Fixture Families

The future report design expects fixture coverage for:

- `free_space_clear`
- `near_threshold_link`
- `blocked_or_nlos`
- `high_rsu_recovery`
- `same_resource_interference`
- `orthogonal_resource_mitigation`
- `sparse_candidate`
- `dense_full_graph_baseline`
- `oracle_candidate_review_only`
- `failed_scheduled_message`

Stage 4.8 can seed some of these cases but remains a boundary audit, not a
complete calibration set.

## Validation Gates

The future executable report must fail if:

- no candidate tau values are supplied by owner configuration;
- any candidate tau is outside `[0, 1]`;
- Stage 4.8 `reliability_threshold = 0.2` appears as an implicit default;
- every scenario family lacks a sparse candidate;
- full graph is labeled oracle;
- oracle-candidate labels are marked as deployment actor input;
- metric-valued fields are not registered;
- `network_delivery_probability` is renamed as
  `consensus_success_probability`;
- reward, training, actor, critic, COMA, GNN, LSTM, or v5 code is introduced.

## Metric Governance

Stage 5.0c adds no metric names.

Metric-valued fields must map to existing registered concepts:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

`tau_candidate`, feasibility booleans, full-graph-only flags, Pareto counts,
recommendation fields, and owner-decision fields are report diagnostics or
identifiers.

## Stage 5.0d Implementation Boundary

Stage 5.0d implements a print-only executable report from this schema:

- `src/marl_topology/evaluation/tau_consensus_calibration_report.py`
- `scripts/replay/tau_consensus_calibration_report.py`
- `docs/STAGE5_0D_TAU_CONSENSUS_CALIBRATION_REPORT.md`

The implementation requires owner-declared candidate tau values and still does
not select the final `tau_consensus`. It uses Stage 4.8 rows as a smoke-test
source only, not as a final calibration set.

Reward implementation remains blocked after Stage 5.0d.

## Acceptance For Stage 5.0c

- This report design document exists.
- The schema separates inputs, output tables, diagnostics, and owner decision.
- No final tau is selected.
- No default numeric tau candidate is introduced.
- Stage 4.8 threshold remains diagnostic only.
- No reward implementation, weight calibration, training, actor/critic code, or
  v5 migration is introduced.

## Residual Risks

- The executable report is not implemented in Stage 5.0c.
- The Stage 5.0d executable report is only a smoke-test sensor until larger
  scenario families exist.
- Fixture families still need concrete scenario builders.
- Owner risk tolerance and decision policy remain unspecified.
- The report design may need revision after larger scenario fixtures exist.
