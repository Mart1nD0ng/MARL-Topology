# Stage 5.0l Stage 3-Backed Sweep Range Expansion And Realism Review

## Scope

Stage 5.0l expands the Stage 3-backed feasibility envelope beyond Stage 5.0k.
It adds bounded reference probes for `tx_power`, `payload`, RSU height/placement,
and explicit resource-budget limits.

It does not select final `tau_consensus`, does not lower
`tau_requirement_min = 0.9`, does not implement reward, does not train models,
does not add actor/critic/COMA/GNN/LSTM code, does not migrate v5 code, and
does not treat full graph as an oracle.

## Controlled Object

The controlled object is the Stage 3-backed feasibility envelope under:

```text
consensus_success_probability >= tau_requirement_min
tau_requirement_min = 0.9
```

The task expands coverage for previously deferred controls:

- `tx_power_sweep`
- `payload_sweep`
- `rsu_height_placement_sweep`
- `resource_budget_limit_sweep`

## Executable Sensor

Module:

```text
src/marl_topology/evaluation/feasibility_envelope_sweep_range_review.py
```

Replay:

```powershell
python scripts\replay\stage5_0l_stage3_backed_sweep_range_review.py
```

Every row uses Stage 3 network records, Stage 3.6 finite-blocklength links, the
Stage 4.3 message-matrix adapter, Stage 4.6 protocol accounting, and Stage 4.4
expected-initiator PBFT reliability.

## Executed Sweeps

| Sweep | Baseline | Intervention | Result |
| --- | --- | --- | --- |
| `tx_power_sweep` | `tx_power_minus_10dbm_stage3` | `tx_power_minus_8dbm_stage3` | reliability recovers from `0.0` to `0.9768154531910614` |
| `payload_sweep` | `payload_18kbits_stage3` | `payload_12kbits_stage3` | reliability recovers from `0.0` to `0.9768154531910614` |
| `rsu_height_placement_sweep` | `rsu_height_8m_blocked_stage3` | `rsu_height_25m_los_stage3` | reliability recovers from `0.0` to `1.0` |
| `resource_budget_limit_sweep` | `resource_budget_2_stage3` | `resource_budget_6_stage3` | reliability recovers from `0.0` to `1.0` |

## Key Design Notes

- `tx_power_sweep` is a bounded diagnostic probe. The values are not a deployment
  power policy.
- `payload_sweep` checks finite-blocklength message-size pressure. It is not a
  reward term.
- `rsu_height_placement_sweep` uses a deterministic blocked-geometry scene and
  demonstrates that RSU height can change LoS feasibility.
- `resource_budget_limit_sweep` prevents a misleading conclusion from Stage
  5.0k: full-graph orthogonalization is feasible only when enough resources are
  available. With two resources, interference groups remain and reliability
  stays below requirement; with six resources, interference groups disappear.

## Realism Review

| Parameter | Current representation | Status | Follow-up |
| --- | --- | --- | --- |
| `tx_power_dbm` | bounded diagnostic values -10 dBm to -8 dBm | `unknown_needs_reference` | Need regulatory and hardware reference. |
| `payload_bits` | 12 kbit and 18 kbit finite-blocklength probes | `unknown_needs_reference` | Need application message-size audit. |
| `rsu_height_m` | 8 m blocked and 25 m high-RSU cases | `unknown_needs_reference` | Need city installation constraints and placement policy. |
| `resource_budget_count` | 2-resource and 6-resource full-graph comparisons | `unknown_needs_reference` | Need explicit spectrum/resource budget. |
| `fault_filter_mode` | `unknown_faults_remove_largest` | `conservative_diagnostic` | Engineering lower-bound approximation only. |

## Metric Governance

Stage 5.0l adds no new metric names.

Metric-valued fields remain:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Fields such as `resource_budget_count`, `realism_status`, `parameter_source`,
and `diagnostic_flags` are diagnostics or identifiers, not primary objectives.

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

- This is still a deterministic four-node reference, not a calibrated city
  distribution.
- Numeric ranges are bounded diagnostic probes and need external realism
  references.
- The 25 m RSU case demonstrates controllability, but deployment placement
  constraints remain open.
- The resource budget limit is simple modulo assignment, not a final scheduler.

## Recommended Next Step

`Stage 5.0m - objective readiness review before reward implementation`.

The next task should decide whether the current feasibility evidence is enough
to permit a reward implementation plan, or whether more scenario calibration is
required first. Reward implementation itself should remain blocked until owner
approval.
