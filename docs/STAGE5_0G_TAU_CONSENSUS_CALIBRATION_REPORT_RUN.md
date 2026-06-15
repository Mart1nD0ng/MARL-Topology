# Stage 5.0g Tau-Consensus Calibration Report Run

## Scope

Stage 5.0g runs the existing tau-consensus calibration report over the Stage
5.0f alpha fixture suite using owner-supplied candidate tau values.

It does not select final `tau_consensus`, does not implement reward, does not
tune reward weights, does not train models, does not add actor/critic/COMA/GNN/
LSTM code, and does not migrate v5 code.

## Owner-Supplied Candidate

Owner input:

```text
tau_consensus <= 0.9
```

For this report-only run, the input is interpreted as the candidate threshold:

```text
tau_candidate = 0.9
```

The objective feasibility inequality remains:

```text
consensus_success_probability >= tau_consensus
```

This document records the candidate-run evidence only. It does not freeze or
recommend final `tau_consensus`.

## Controlled Object

The controlled object is the Stage 5 tau calibration evidence loop: Stage 5.0f
fixture rows consumed by the Stage 5.0d calibration report builder.

## Desired State

The project should have an executable report run showing how the alpha fixture
suite behaves under the owner-supplied candidate `tau = 0.9`, while preserving
metric governance, source labeling, no-selection semantics, and reward/training
blocks.

## Sensors

- `build_tau_consensus_calibration_report(...)` over Stage 5.0f fixture rows.
- Candidate tau rows.
- Feasibility summary rows by scenario family.
- Report checks for metric registration, source kind, no default tau, no final
  selection, full-graph boundary, oracle leakage, reward, training, and v5.
- Contract tests and harness validation.

## Actuators

- Correct Stage 5.0f feasibility-summary source labeling.
- Record the Stage 5.0g report-run summary.
- Add a harness task and contract test to prevent future source-label and
  tau-selection regressions.
- Update project state and post-task self-review.

## Report Command

```powershell
python scripts\replay\tau_consensus_calibration_report.py --source stage5_0f --tau 0.9 --tau-source owner_supplied_stage5_0g --tau-owner-note "owner supplied tau_consensus <= 0.9; interpreted for Stage 5.0g as candidate tau=0.9, not final selection"
```

The same builder was also queried directly to extract the summary below.

## Report Header

```text
source_kind: stage5_0f_alpha_fixture_suite
calibration_ready: True
tau_selected: False
final_tau_consensus: None
candidate_tau: 0.9
tau_source: owner_supplied_stage5_0g
```

## Gate Checks

```text
candidate_tau_values_supplied: True
candidate_tau_values_in_range: True
no_default_tau_candidates: True
no_final_tau_selected: True
stage4_8_threshold_not_used_as_default: True
source_is_smoke_test_only: False
source_is_alpha_fixture_suite: True
metric_valued_fields_registered: True
non_saturated_consensus_present: True
sparse_better_than_full_graph_for_some_family: True
full_graph_not_oracle: True
oracle_labels_not_actor_inputs: True
reward_implemented: False
training_run: False
v5_code_migrated: False
```

## Feasibility Summary At Tau 0.9

| Scenario family | Feasible / total | Non-full feasible | Full graph feasible | Lowest feasible latency | Lowest feasible energy | Notes |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `blocked_or_nlos_urban` | 0 / 3 | 0 | False | n/a | n/a | no feasible topology at tau |
| `clear_free_space_reference` | 2 / 3 | 1 | True | 0.0006 | 0.00011 | sparse feasible and cheaper |
| `deadline_tight_retransmission` | 0 / 3 | 0 | False | n/a | n/a | tight deadline remains below tau |
| `near_threshold_link_budget` | 0 / 3 | 0 | False | n/a | n/a | near-threshold rows remain below tau |
| `same_resource_interference` | 1 / 3 | 1 | False | 0.00065 | 0.00012 | sparse feasible; full graph penalized |
| `sparse_vs_dense_tradeoff` | 2 / 3 | 1 | True | 0.0007 | 0.00013 | sparse and full feasible; sparse cheaper |
| `unreachable_reliability_target` | 1 / 3 | 1 | False | 0.0011 | 0.00026 | sparse candidate feasible in alpha rows |
| `weak_primary_distribution` | 0 / 3 | 0 | False | n/a | n/a | weak-primary spread visible but below tau |

Aggregate:

```text
total_topology_rows: 24
feasible_topology_rows_at_tau_0_9: 6
infeasible_topology_rows_at_tau_0_9: 18
families_with_any_feasible_topology: 4
families_without_feasible_topology: 4
feasible_non_full_topology_rows: 4
```

## Feasible Detail Rows

| Scenario family | Topology | Family | Consensus success probability | Latency | Energy | Full graph baseline |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `clear_free_space_reference` | `clear_free_space_reference/sparse_candidate` | `sparse_candidate` | 0.9995130274056901 | 0.0006 | 0.00011 | False |
| `clear_free_space_reference` | `clear_free_space_reference/dense_full_graph_baseline` | `dense_full_graph_baseline` | 0.9971441583935992 | 0.0014 | 0.00046 | True |
| `same_resource_interference` | `same_resource_interference/sparse_candidate` | `sparse_candidate` | 0.9995130274056901 | 0.00065 | 0.00012 | False |
| `unreachable_reliability_target` | `unreachable_reliability_target/sparse_candidate` | `sparse_candidate` | 0.9610448520494868 | 0.0011 | 0.00026 | False |
| `sparse_vs_dense_tradeoff` | `sparse_vs_dense_tradeoff/sparse_candidate` | `sparse_candidate` | 0.9995130274056901 | 0.0007 | 0.00013 | False |
| `sparse_vs_dense_tradeoff` | `sparse_vs_dense_tradeoff/dense_full_graph_baseline` | `dense_full_graph_baseline` | 0.9995130274056901 | 0.0022 | 0.00082 | True |

## Interpretation

Candidate `tau = 0.9` is stringent but not vacuous on the alpha suite.

It separates free-space, sparse resource-efficient, and interference-sensitive
cases from blocked, near-threshold, deadline-tight, and weak-primary cases.
This is useful as a calibration signal, but it is still not enough to select a
final threshold because the Stage 5.0f suite is deterministic and alpha-scale.

The run also confirms that sparse feasible candidates can be cheaper than full
graph baselines, and that full graph remains a baseline rather than an oracle.

## Regression Check

Protected behavior:

- Stage 5.0f source rows are labeled as `stage5_0f_alpha_fixture_suite`, not
  Stage 4.8 smoke-only rows.
- Candidate tau remains owner-supplied and non-default.
- `final_tau_consensus`, `recommended_tau_candidate`, and `owner_selected_tau`
  remain unset.
- No reward, training, actor/critic model, or v5 code is introduced.

## Residual Risks

- Stage 5.0f is an alpha fixture suite, not a full city-scale calibration
  distribution.
- `tau = 0.9` may be too strict for weak-primary and NLoS-heavy regimes, but
  that is evidence for owner review rather than a model failure.
- The final threshold still needs owner decision and likely a broader
  scenario-calibration run before reward implementation.

## Owner Decision Required

Yes. This run does not select final `tau_consensus`.
