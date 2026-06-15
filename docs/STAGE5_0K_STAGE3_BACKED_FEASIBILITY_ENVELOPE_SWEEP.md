# Stage 5.0k Stage 3-Backed Feasibility Envelope Sweep Hardening

## Scope

Stage 5.0k hardens the Stage 5.0j alpha sweep by grounding a minimal subset of
controls in Stage 3 communication records.

It does not select final `tau_consensus`, does not lower
`tau_requirement_min = 0.9`, does not implement reward, does not train models,
does not add actor/critic/COMA/GNN/LSTM code, does not migrate v5 code, and
does not treat full graph as an oracle.

## Controlled Object

The controlled object is the feasibility envelope around three Stage 3-backed
communication controls:

- `bandwidth_sweep`
- `deadline_sweep`
- `resource_orthogonalization_sweep`

The row objective sensor remains:

```text
consensus_success_probability >= tau_requirement_min
tau_requirement_min = 0.9
```

## Stage 3 Backing

The executable report builder is:

```text
src/marl_topology/evaluation/feasibility_envelope_sweep_stage3_backed.py
```

Replay command:

```powershell
python scripts\replay\stage5_0k_stage3_backed_feasibility_envelope_sweep.py
```

Each row uses:

- Stage 3 channel records from `stage3_channel_v1_fspl_sinr`;
- Stage 3.6 finite-blocklength link records from `urlcc_finite_blocklength_v1`;
- Stage 3 network communication records from `stage3_network_communication_v1`;
- Stage 4.3 message-matrix adapter `stage4_stage3_network_to_pbft_matrix_v1`;
- Stage 4.6 protocol latency/energy accounting;
- Stage 4.4 expected-initiator PBFT reliability.

Fault filter mode is `unknown_faults_remove_largest` as a conservative
engineering lower-bound approximation. It is not a strict Byzantine adversary
model.

## Executed Sweep Subset

Executed with Stage 3 backing:

- `bandwidth_sweep`
- `deadline_sweep`
- `resource_orthogonalization_sweep`

Deferred:

- `tx_power_sweep`
- `payload_sweep`
- `rsu_height_placement_sweep`
- `fault_filter_mode_comparison`
- `topology_candidate_expansion`

The deferred sweeps need either wider physical-unit bounds or a different
non-alpha fixture design before their results should be interpreted.

## Sweep Summary

| Sweep | Baseline feasible | Intervention feasible | Max delta consensus probability | Stage 3-backed interpretation |
| --- | --- | --- | ---: | --- |
| `bandwidth_sweep` | False | True | 0.9768154531910614 | Increasing bandwidth from 15 MHz to 20 MHz restored reliability under finite-blocklength link records. |
| `deadline_sweep` | False | True | 0.8900812193082798 | Increasing link deadline from 1 ms to 3 ms restored reliability but increased latency and energy. |
| `resource_orthogonalization_sweep` | False | True | 1.0 | Orthogonal resources removed same-resource interference groups and restored reliability. |

## Key Rows

| Sweep | Control | Probability | Requirement met | Latency | Energy | Stage 3 diagnostic |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `bandwidth_sweep` | `bandwidth_15mhz_stage3` | 0.0 | False | 0.00720281399225744 | 0.005944279211619645 | finite-blocklength, full graph, no interference |
| `bandwidth_sweep` | `bandwidth_20mhz_stage3` | 0.9768154531910614 | True | 0.0022698175688954207 | 0.004134917967719052 | finite-blocklength, full graph, no interference |
| `deadline_sweep` | `deadline_1ms_stage3` | 0.08673423388278172 | False | 0.00180070349806436 | 0.00396285280774643 | deadline-limited finite-blocklength rows |
| `deadline_sweep` | `deadline_3ms_stage3` | 0.9768154531910614 | True | 0.0022698175688954207 | 0.004134917967719052 | relaxed retransmission window |
| `resource_orthogonalization_sweep` | `shared_resource_stage3` | 0.0 | False | 0.00720281399225744 | 0.014871345143343598 | interference groups present |
| `resource_orthogonalization_sweep` | `orthogonal_resources_stage3` | 1.0 | True | 0.00180070349806436 | 0.003965692099788303 | interference groups removed |

## Interpretation

The Stage 3-backed sweep confirms that the Stage 5.0j alpha directions were not
purely hand-written probability artifacts for the three hardened axes:

- bandwidth affects finite-blocklength delivery probability and protocol
  reliability;
- deadline affects retransmission opportunity, scheduled latency, and energy;
- resource orthogonalization affects interference and dense/full topology cost.

This is still not a final deployment feasibility envelope. It is a deterministic
reference sensor over one small four-node scene.

## Metric Governance

Stage 5.0k adds no new metric names.

Metric-valued fields remain:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Additional row fields such as `stage3_backed`, `interference_group_ids`,
`phase_budgets_s`, `fault_filter_mode`, and `parameter_source` are diagnostics
or identifiers, not new primary objectives.

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

- The scene is a deterministic four-node reference, not a large city scenario
  distribution.
- `tx_power_sweep`, `payload_sweep`, and `rsu_height_placement_sweep` still need
  Stage 3-backed execution.
- Resource orthogonalization currently assumes enough orthogonal resources for
  the full graph; future work must add explicit resource-budget limits.
- The conservative `remove_largest` filter remains an engineering lower-bound
  approximation, not a strict Byzantine adversary model.

## Recommended Next Step

`Stage 5.0l - Stage 3-backed sweep range expansion and realism review`.

The next useful actuator is to add bounded `tx_power`, `payload`, and
RSU-height/placement sweeps, plus explicit resource-budget limits, before
authorizing any reward implementation or training work.
