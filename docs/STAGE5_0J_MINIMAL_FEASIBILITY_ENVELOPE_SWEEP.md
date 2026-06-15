# Stage 5.0j Minimal Executable Feasibility Envelope Sweep

## Scope

Stage 5.0j executes a small deterministic feasibility envelope sweep using the
Stage 5.0i manifest. It is an alpha evidence sensor, not a final calibration
run.

It does not select final `tau_consensus`, does not lower
`tau_requirement_min = 0.9`, does not implement reward, does not train models,
does not add actor/critic/COMA/GNN/LSTM code, does not migrate v5 code, and
does not treat full graph as an oracle.

## Controlled Object

The controlled object is the minimal Stage 5 feasibility envelope around
selected Stage 5.0h infeasible rows.

## Desired State

The project should produce deterministic sweep rows that show whether specific
control families can restore feasibility under:

```text
consensus_success_probability >= tau_requirement_min
tau_requirement_min = 0.9
```

Reliability, latency, and energy remain separate registered metric concepts.

## Executable Sensor

Module:

```text
src/marl_topology/evaluation/feasibility_envelope_sweep.py
```

Replay:

```powershell
python scripts\replay\stage5_0j_minimal_feasibility_envelope_sweep.py
```

The sweep uses Stage 4.4 expected-initiator PBFT closed-form reliability over
declared alpha message matrices. It does not use Monte Carlo, sampling, subset
enumeration, or v5 code.

## Executed Sweep Subset

Executed:

- `bandwidth_sweep`
- `deadline_sweep`
- `resource_orthogonalization_sweep`
- `fault_filter_mode_comparison`
- `topology_candidate_expansion`

Deferred:

- `tx_power_sweep`
- `payload_sweep`
- `rsu_height_placement_sweep`

The deferred sweeps still matter, but they need more Stage 3-backed parameter
records before they should be interpreted.

## Sweep Summary

| Sweep | Baseline feasible | Intervention feasible | Max delta consensus probability | Interpretation |
| --- | --- | --- | ---: | --- |
| `bandwidth_sweep` | False | True | 0.6207294976131281 | Increased bandwidth-like alpha control can recover near-threshold link budget. |
| `deadline_sweep` | False | True | 0.2367227095276806 | Relaxed retry/deadline budget can recover deadline-tight sparse candidate. |
| `resource_orthogonalization_sweep` | False | True | 0.9743962698980282 | Orthogonal resources can recover dense full-graph interference case. |
| `fault_filter_mode_comparison` | False | False | 0.0 | Conservative `remove_largest` does not improve reliability and remains below requirement. |
| `topology_candidate_expansion` | False | True | 0.16334243816555805 | Candidate expansion can recover weak-primary sparse case, with latency/energy cost. |

## Key Rows

| Sweep | Topology | Control | Probability | Requirement met | Latency | Energy |
| --- | --- | --- | ---: | --- | ---: | ---: |
| `bandwidth_sweep` | `near_threshold_link_budget/sparse_candidate` | baseline | 0.35910052088506106 | False | 0.001 | 0.00020 |
| `bandwidth_sweep` | `near_threshold_link_budget/sparse_candidate` | `bandwidth_x2_alpha` | 0.9798300184981892 | True | 0.00078 | 0.00022 |
| `deadline_sweep` | `deadline_tight_retransmission/sparse_candidate` | baseline | 0.7243221425218062 | False | 0.00085 | 0.00023 |
| `deadline_sweep` | `deadline_tight_retransmission/sparse_candidate` | `deadline_relaxed_retry_budget_alpha` | 0.9610448520494868 | True | 0.00120 | 0.00030 |
| `resource_orthogonalization_sweep` | `same_resource_interference/dense_full_graph_baseline` | baseline | 0.022747888495571016 | False | 0.00260 | 0.00074 |
| `resource_orthogonalization_sweep` | `same_resource_interference/dense_full_graph_baseline` | `orthogonal_resources_alpha` | 0.9971441583935992 | True | 0.00160 | 0.00055 |
| `fault_filter_mode_comparison` | `weak_primary_distribution/dense_full_graph_baseline` | `none_baseline` | 0.8474859430426385 | False | 0.00200 | 0.00078 |
| `fault_filter_mode_comparison` | `weak_primary_distribution/dense_full_graph_baseline` | `remove_largest_conservative` | 0.014229157765296013 | False | 0.00200 | 0.00078 |
| `topology_candidate_expansion` | `weak_primary_distribution/sparse_candidate` | baseline | 0.7555748270758926 | False | 0.00100 | 0.00024 |
| `topology_candidate_expansion` | `weak_primary_distribution/sparse_candidate` | `expanded_candidate_edges_alpha` | 0.9189172652414507 | True | 0.00135 | 0.00036 |

## Interpretation

The alpha sweep suggests that four control families can restore feasibility in
the chosen deterministic rows:

- communication resource increase for near-threshold link budget;
- deadline/retry budget relaxation for deadline-tight rows;
- resource orthogonalization for dense interference rows;
- topology candidate expansion for weak-primary sparse rows.

The conservative fault-filter comparison is intentionally negative: it lowers
or preserves reliability and should not be interpreted as a strict Byzantine
adversary model. It confirms that the previous `fault_filter_mode = none` alpha
suite is optimistic.

## Metric Governance

Stage 5.0j adds no new metric names.

Metric-valued fields remain:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Fields such as `sweep_id`, `failure_reason_before`, `failure_reason_after`,
`parameter_source`, and `diagnostic_flags` are diagnostics or identifiers.

## Negative Controls

The executable sweep preserves:

- `tau_requirement_min = 0.9` fixed;
- no final tau selection;
- no lower diagnostic tau as threshold;
- no reward implementation;
- no training;
- no actor/critic/model implementation;
- no v5 code migration;
- no oracle labels or sweep labels in actor inputs;
- no full graph as oracle.

## Residual Risks

- This is still an alpha sweep over declared Stage 5 fixture controls, not a
  full Stage 3 geometry/channel-backed parameter sweep.
- Numeric control labels such as `bandwidth_x2_alpha` are deterministic
  evidence probes and need future physical-unit grounding.
- `tx_power_sweep`, `payload_sweep`, and `rsu_height_placement_sweep` remain
  deferred.
- Combined controls remain deferred until single-axis behavior is backed by
  richer Stage 3 records.

## Recommended Next Step

`Stage 5.0k - Stage 3-backed feasibility envelope sweep hardening`.

The next step should attach sweep rows to actual Stage 3 finite-blocklength,
geometry, and network resource records for at least bandwidth, deadline, and
resource orthogonalization.
