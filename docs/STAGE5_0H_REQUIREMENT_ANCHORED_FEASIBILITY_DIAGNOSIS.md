# Stage 5.0h Requirement-Anchored Feasibility Diagnosis

## Scope

Stage 5.0h treats `tau = 0.9` as a reality-anchored requirement baseline, not a
threshold fitted from the current alpha fixture suite.

It does not choose any lower final tau, does not implement reward, does not
train models, does not add actor/critic/COMA/GNN/LSTM code, does not migrate v5
code, and does not change simulation parameters to force feasibility.

Boundary shorthand: does not migrate v5 code.

## Terminology

```text
tau_requirement_min = 0.9
tau_stress_candidates = [0.95, 0.99]
tau_diagnostic_values = [0.5, 0.75]
```

Rules:

- `tau_requirement_min` is the minimum acceptable reliability requirement.
- `tau_stress_candidates` are stricter-than-requirement stress checks only.
- `tau_diagnostic_values` are lower diagnostic cuts only; they must not become
  final objective thresholds.
- `tau_candidate` language in earlier reports should be read as report input,
  not automatic selection.
- Stage 4.8 `reliability_threshold = 0.2` remains an audit diagnostic threshold.
- Stage 5.0g `tau_candidate = 0.9` is reinterpreted here as
  `tau_requirement_min = 0.9`.

The objective feasibility condition remains:

```text
consensus_success_probability >= tau_requirement_min
```

## Controlled Object

The controlled object is the Stage 5 feasibility diagnosis loop over the Stage
5.0f alpha fixture suite and Stage 4 expected-initiator PBFT reliability rows.

## Desired State

The project should identify which scenarios and topologies satisfy
`tau_requirement_min = 0.9`, classify infeasible rows by likely cause, and
recommend environment/protocol/topology diagnostics before any reward or
training work.

## Executable Sensor

The executable report builder is:

```text
src/marl_topology/evaluation/requirement_feasibility_diagnosis.py
```

Replay command:

```powershell
python scripts\replay\stage5_0h_requirement_feasibility_diagnosis.py
```

## Requirement Feasibility Summary

At `tau_requirement_min = 0.9`:

```text
total_topology_rows: 24
feasible_topology_rows: 6
infeasible_topology_rows: 18
families_with_any_feasible_topology: 4
families_with_no_feasible_topology: 4
final_tau_selected: False
final_tau_below_requirement_selected: False
```

| Scenario family | Feasible / total | Dominant failure reason | Interpretation |
| --- | ---: | --- | --- |
| `blocked_or_nlos_urban` | 0 / 3 | `link_budget_failure` | NLoS/blocking link budget is below requirement. |
| `clear_free_space_reference` | 2 / 3 | `topology_candidate_failure` | Sparse and full graph pass; weak baseline fails. |
| `deadline_tight_retransmission` | 0 / 3 | `deadline_failure` | Deadline and retry budget need diagnosis. |
| `near_threshold_link_budget` | 0 / 3 | `link_budget_failure` | Link budget remains below requirement. |
| `same_resource_interference` | 1 / 3 | `interference_failure` | Sparse passes; full graph is penalized by interference/resource sharing. |
| `sparse_vs_dense_tradeoff` | 2 / 3 | `topology_candidate_failure` | Sparse and full graph pass; sparse is cheaper. |
| `unreachable_reliability_target` | 1 / 3 | `resource_budget_failure` | Sparse passes; capped dense/weak rows fail. |
| `weak_primary_distribution` | 0 / 3 | `primary_distribution_failure` | Uniform initiator distribution exposes weak primary and PBFT quorum amplification. |

## Infeasible Row Classification

| Scenario family | Topology | Failure reason | Diagnostic note |
| --- | --- | --- | --- |
| `clear_free_space_reference` | `clear_free_space_reference/weak_baseline` | `topology_candidate_failure` | Weak/disconnected topology cannot satisfy the requirement. |
| `near_threshold_link_budget` | `near_threshold_link_budget/weak_baseline` | `link_budget_failure` | Audit path loss, LoS/NLoS, bandwidth, power, and distance. |
| `near_threshold_link_budget` | `near_threshold_link_budget/sparse_candidate` | `link_budget_failure` | Sparse candidate still below requirement. |
| `near_threshold_link_budget` | `near_threshold_link_budget/dense_full_graph_baseline` | `link_budget_failure` | Dense topology does not compensate for weak link budget. |
| `blocked_or_nlos_urban` | `blocked_or_nlos_urban/weak_baseline` | `link_budget_failure` | Blocking/NLoS link budget below requirement. |
| `blocked_or_nlos_urban` | `blocked_or_nlos_urban/sparse_candidate` | `link_budget_failure` | Geometry and link budget need audit before lowering tau. |
| `blocked_or_nlos_urban` | `blocked_or_nlos_urban/dense_full_graph_baseline` | `link_budget_failure` | Dense/full topology is not enough under NLoS penalty. |
| `same_resource_interference` | `same_resource_interference/weak_baseline` | `topology_candidate_failure` | Weak topology candidate fails independently of sparse feasible row. |
| `same_resource_interference` | `same_resource_interference/dense_full_graph_baseline` | `interference_failure` | Full graph is hurt by shared-resource interference. |
| `deadline_tight_retransmission` | `deadline_tight_retransmission/weak_baseline` | `deadline_failure` | Scheduled communication exists but deadline budget is too tight. |
| `deadline_tight_retransmission` | `deadline_tight_retransmission/sparse_candidate` | `retransmission_insufficient` | Candidate improves reliability but retry budget is insufficient. |
| `deadline_tight_retransmission` | `deadline_tight_retransmission/dense_full_graph_baseline` | `deadline_failure` | Dense graph still fails under tight deadline. |
| `unreachable_reliability_target` | `unreachable_reliability_target/weak_baseline` | `topology_candidate_failure` | Weak/disconnected topology cannot satisfy requirement. |
| `unreachable_reliability_target` | `unreachable_reliability_target/dense_full_graph_baseline` | `resource_budget_failure` | Finite resource budget appears capped before requirement. |
| `sparse_vs_dense_tradeoff` | `sparse_vs_dense_tradeoff/weak_baseline` | `topology_candidate_failure` | Weak topology fails while sparse and dense candidates pass. |
| `weak_primary_distribution` | `weak_primary_distribution/weak_primary_baseline` | `primary_distribution_failure` | Weak primary drives expected initiator reliability below requirement. |
| `weak_primary_distribution` | `weak_primary_distribution/sparse_candidate` | `primary_distribution_failure` | Improved sparse row still has weak-primary spread. |
| `weak_primary_distribution` | `weak_primary_distribution/dense_full_graph_baseline` | `pbft_quorum_failure` | PBFT three-phase quorum amplification remains below requirement. |

Supported failure categories:

- `link_budget_failure`
- `deadline_failure`
- `retransmission_insufficient`
- `topology_candidate_failure`
- `interference_failure`
- `pbft_quorum_failure`
- `primary_distribution_failure`
- `resource_budget_failure`
- `modeling_suspicious`
- `unknown`

## Parameter Sanity Table

| Parameter | Current representation | Status | Diagnostic note |
| --- | --- | --- | --- |
| `tx_power` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Audit Stage 3.6 link records before changing. |
| `bandwidth` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Needed for finite-blocklength feasibility envelope. |
| `noise` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Needed to separate SINR failure from topology failure. |
| `carrier_frequency` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Needed for path-loss realism. |
| `path_loss_los_nlos_penalty` | represented only by family diagnostic flags | `unknown_needs_reference` | Blocked/NLoS family suggests sensitivity but not calibrated realism. |
| `interference_model` | same-resource family uses declared alpha probabilities | `unknown_needs_reference` | Requires Stage 3 resource-group audit. |
| `payload_bits` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Needed for finite-blocklength and deadline diagnosis. |
| `deadline` | deadline-tight retransmission is a stress fixture | `too_strict` | Stress row intentionally diagnoses deadline pressure. |
| `attempt_duration` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Needed to diagnose retransmission feasibility. |
| `max_retransmissions` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Needed to separate retry-budget failure from link-budget failure. |
| `rsu_vehicle_distances` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Needed for geometry-backed calibration. |
| `candidate_edge_radius` | not exposed in Stage 5.0f alpha rows | `unknown_needs_reference` | Needed to diagnose topology candidate expansion. |
| `pbft_n_f_quorum` | `n=4`, `f=1`, quorum `3` minimal PBFT fixture | `reasonable` | Valid minimal committee; larger `n` should be swept later. |
| `fault_filter_mode` | `none` in Stage 5.0f alpha suite | `too_loose` | Optimistic relative to conservative `remove_largest` comparison. |

This table is intentionally diagnostic. It does not change parameters to make
`tau_requirement_min = 0.9` pass.

## Stress And Diagnostic Values

Stress checks:

| Tau value | Feasible rows | Infeasible rows | Meaning |
| ---: | ---: | ---: | --- |
| 0.95 | 6 | 18 | stricter-than-requirement stress only |
| 0.99 | 5 | 19 | stricter-than-requirement stress only |

Lower diagnostic values:

| Tau value | Feasible rows | Infeasible rows | Meaning |
| ---: | ---: | ---: | --- |
| 0.5 | 10 | 14 | diagnostic only |
| 0.75 | 8 | 16 | diagnostic only |

Lower diagnostic values are not candidate final thresholds.

## Feasibility Envelope Plan

Do not run a large sweep yet. The next actuator should be a controlled envelope
plan that holds `tau_requirement_min = 0.9` fixed and varies one family of
controls at a time:

| Sweep | Control | Diagnosis target |
| --- | --- | --- |
| `bandwidth_sweep` | `bandwidth_hz` | Separate finite-blocklength capacity from topology failure. |
| `tx_power_sweep` | `tx_power_w` | Separate link-budget failure from topology candidate failure. |
| `deadline_sweep` | `deadline_s` | Separate deadline failure from intrinsic link failure. |
| `payload_sweep` | `payload_bits` | Separate message-size pressure from topology failure. |
| `rsu_height_placement_sweep` | RSU height and position | Separate NLoS geometry from channel budget. |
| `resource_orthogonalization_sweep` | channel/resource assignment | Separate full-graph interference from dense-topology infeasibility. |
| `fault_filter_mode_comparison` | `none` vs `remove_largest` | Compare optimistic and conservative PBFT reliability. |
| `topology_candidate_expansion` | candidate edge radius and candidate edges | Separate missing candidates from environment infeasibility. |

Blocked during these sweeps:

- do not lower `tau_requirement_min`;
- do not implement reward;
- do not train models;
- do not migrate v5 code.

## Answers To Owner Questions

1. Feasible fixture families at `tau = 0.9`: `clear_free_space_reference`,
   `same_resource_interference`, `sparse_vs_dense_tradeoff`, and
   `unreachable_reliability_target` each have at least one feasible topology.
2. Fully infeasible fixture families: `blocked_or_nlos_urban`,
   `deadline_tight_retransmission`, `near_threshold_link_budget`, and
   `weak_primary_distribution`.
3. Main causes: link budget, deadline/retransmission budget, topology candidate
   weakness, interference, resource caps, PBFT quorum amplification, and
   primary distribution weakness.
4. The current failures are mixed: communication resources dominate
   `near_threshold` and `blocked/NLoS`; topology candidates affect weak
   baselines; protocol/PBFT effects dominate weak-primary rows; simulation
   parameter realism remains an observability gap because Stage 5.0f rows are
   alpha fixtures rather than full Stage 3 parameterized scenarios.
5. Yes: `tau_requirement_min = 0.9` can be written into the objective contract
   as a requirement baseline. It is not a final calibrated optimum and not a
   reward threshold implementation.
6. Reward implementation should remain blocked. The next missing evidence is
   feasibility envelope sensitivity, not reward shaping.
7. Yes: the next technical task should be a feasibility envelope sweep design
   and then a minimal executable sweep, with `tau_requirement_min = 0.9` held
   fixed.

## Regression Check

Protected behavior:

- no final tau below `0.9`;
- no final tau selection in this stage;
- no reward implementation;
- no training;
- no actor/critic/model code;
- no v5 code migration;
- no simulation parameter change to force feasibility;
- no full-graph-as-oracle claim.

## Residual Risks

- Stage 5.0f is alpha-scale and not yet a realistic city distribution.
- Most physical parameters are not directly exposed in the alpha rows; they
  require a Stage 3-backed feasibility envelope sensor.
- `fault_filter_mode = none` is optimistic; a conservative comparison may lower
  feasible counts.
- Requirement baseline `0.9` is now documented, but final deployment acceptance
  still needs owner approval after envelope evidence.
