# Stage 5.0i Feasibility Envelope Sweep Design

## Scope

Stage 5.0i designs the feasibility envelope sweep that should follow Stage
5.0h. It is a design and gate-freeze step, not a sweep execution.

It does not select final `tau_consensus`, does not lower
`tau_requirement_min`, does not change simulation parameters to force
feasibility, does not implement reward, does not train models, does not add
actor/critic/COMA/GNN/LSTM code, and does not migrate v5 code.

## Controlled Object

The controlled object is the feasibility envelope around
`tau_requirement_min = 0.9`, spanning:

- Stage 3 communication resources and physical/channel controls;
- Stage 3 network resource assignment;
- Stage 4 expected-initiator PBFT controls;
- topology candidate generation.

## Desired State

The next executable sweep should show which control families can move
infeasible Stage 5.0h rows toward `consensus_success_probability >= 0.9`, while
reporting latency and energy costs separately.

The design must preserve:

- `tau_requirement_min = 0.9` fixed in every sweep row;
- single-axis changes before any combined sweep;
- metric governance;
- full graph as baseline only;
- no oracle labels or sweep labels in deployment actor inputs;
- reward and training blocks.

## Executable Design Manifest

The machine-readable design manifest is:

```text
src/marl_topology/evaluation/feasibility_envelope_sweep_design.py
```

Replay command:

```powershell
python scripts\replay\stage5_0i_feasibility_envelope_sweep_design.py
```

The replay prints a JSON design manifest. It does not run a sweep.

## Tau Policy

```text
tau_requirement_min = 0.9
fixed_during_sweeps = true
lower_tau_diagnostic_values_allowed_in_sweep = false
final_tau_selected = false
final_tau_below_requirement_selected = false
```

The primary feasibility test remains:

```text
consensus_success_probability >= tau_requirement_min
```

Lower diagnostic tau values from Stage 5.0h are not sweep thresholds.

## Sweep Axes

| Sweep id | Controlled parameter | Unit | Main target failures | Primary signal |
| --- | --- | --- | --- | --- |
| `bandwidth_sweep` | `bandwidth_hz` | Hz | `link_budget_failure`, `retransmission_insufficient` | Higher bandwidth should improve reliability or reduce required duration. |
| `tx_power_sweep` | `tx_power_w` | W | `link_budget_failure` | Higher power should improve SINR-limited reliability. |
| `deadline_sweep` | `deadline_s` | s | `deadline_failure`, `retransmission_insufficient` | Longer deadline should increase allowed attempts and deadline delivery probability. |
| `payload_sweep` | `payload_bits` | bits | `link_budget_failure`, `deadline_failure`, `resource_budget_failure` | Larger payload should reduce feasibility; smaller payload may recover it. |
| `rsu_height_placement_sweep` | `rsu_height_and_position` | m | `link_budget_failure`, `topology_candidate_failure` | Better placement/height should recover LoS where geometry permits. |
| `resource_orthogonalization_sweep` | `channel_resource_assignment` | resource label | `interference_failure`, `resource_budget_failure` | Orthogonal resources should improve dense/full rows hurt by interference. |
| `fault_filter_mode_comparison` | `fault_filter_mode` | mode | `pbft_quorum_failure`, `primary_distribution_failure` | `remove_largest` should lower or preserve reliability relative to `none`. |
| `topology_candidate_expansion` | `candidate_edge_radius_and_candidate_edges` | m or edge count | `topology_candidate_failure`, `primary_distribution_failure` | More candidates can improve feasibility but may raise latency and energy. |

## Execution Order

1. `single_axis_physics_and_link`
   `bandwidth_sweep`, `tx_power_sweep`, `payload_sweep`.
   Resolve link-budget and finite-blocklength observability gaps first.

2. `single_axis_timing_and_resource`
   `deadline_sweep`, `resource_orthogonalization_sweep`.
   Separate deadline and interference failures from topology failures.

3. `single_axis_geometry_and_topology`
   `rsu_height_placement_sweep`, `topology_candidate_expansion`.
   Test whether geometry and candidate-set actuators can restore feasibility.

4. `protocol_conservatism_check`
   `fault_filter_mode_comparison`.
   Measure optimistic vs conservative PBFT reliability sensitivity.

Combined sweeps are deferred until single-axis behavior is observable.

## Required Future Sweep Row Schema

Future executable sweep rows must include:

```text
sweep_id
sweep_run_id
scenario_family
fixture_id
topology_name
controlled_parameter
control_value_label
control_unit
tau_requirement_min
requirement_met
consensus_success_probability
latency
energy
failure_reason_before
failure_reason_after
selected_edge_count
is_full_graph_baseline
is_oracle_candidate
is_deployment_actor_input
parameter_source
diagnostic_flags
```

Metric-valued fields are only:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

All other fields are identifiers or diagnostics.

## Comparison Policy

Baseline:

```text
unswept Stage 5.0h row or Stage 3-backed equivalent
```

Interpretation order:

1. Did the controlled parameter move the intended failure reason?
2. Did reliability reach `tau_requirement_min`?
3. What latency and energy cost changed?
4. Did `failure_reason_after` change to a more specific cause?
5. Is the effect consistent with the declared monotonicity expectation?

Latency and energy are compared only after feasibility status is known.

## Acceptance Sensors

Future Stage 5.0i executable sweep must show:

- all required sweep ids are present;
- every sweep row holds `tau_requirement_min = 0.9`;
- one control family changes at a time;
- registered metric-valued fields only;
- monotonic sanity checks pass or emit anomaly flags;
- full graph is not treated as oracle;
- oracle and sweep labels are not actor inputs;
- reward and training remain absent.

## Negative Controls

Forbidden:

- selecting final `tau_consensus`;
- lowering `tau_requirement_min` below `0.9`;
- using lower diagnostic tau values as final thresholds;
- changing simulation parameters merely to force feasibility;
- implementing reward or reward weights;
- running training or adding model code;
- migrating v5 code or phase scripts;
- treating full graph as oracle.

## Coupling Map

| Controlled axis | Coupled behavior | Regression risk |
| --- | --- | --- |
| Bandwidth and transmit power | finite-blocklength reliability, latency, energy | hiding topology failure as link failure |
| Deadline and retransmission | delivery probability, scheduled latency, energy | counting failed scheduled messages as no-cost |
| Payload | link reliability and latency/energy | making unrealistic small payloads pass requirement |
| RSU placement | LoS/NLoS and candidate graph geometry | using geometry edits to force feasibility without documenting scenario change |
| Resource assignment | interference and dense topology cost | claiming full graph is optimal after adding resources |
| Fault filter | PBFT reliability conservatism | treating `remove_largest` as strict Byzantine model |
| Candidate expansion | topology feasibility and resource cost | leaking oracle candidate labels to actors |

## Residual Risks

- Stage 5.0i does not yet run the sweep; it freezes the future sweep contract.
- Some numeric ranges still need owner or literature-backed bounds before the
  executable sweep.
- Combined control interactions remain deferred until single-axis sweeps are
  observed.

## Recommended Next Step

`Stage 5.0j - minimal executable feasibility envelope sweep`.

The first executable sweep should use a small deterministic subset:

- `bandwidth_sweep`;
- `deadline_sweep`;
- `resource_orthogonalization_sweep`;
- `fault_filter_mode_comparison`;
- `topology_candidate_expansion`.

It should still keep `tau_requirement_min = 0.9` fixed and should not implement
reward or training.
