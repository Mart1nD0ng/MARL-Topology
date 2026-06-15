# Stage 4.8 Communication/Consensus Boundary Audit

## Controlled Object

The controlled object is the boundary between Stage 3 communication records and
Stage 4 PBFT/application consensus evaluation:

- `src/marl_topology/link/transmission.py`
- `src/marl_topology/network/communication.py`
- `src/marl_topology/protocol/message_matrix_adapter.py`
- `src/marl_topology/protocol/pbft_accounting.py`
- `src/marl_topology/evaluation/stage4_boundary_audit.py`

## Desired State

Before Stage 5 reward/objective contract freeze, the project must show that
the communication and consensus layers behave correctly in non-saturated,
boundary, failure, and resource trade-off cases.

Desired properties:

- Stage 3.6 still uses `urlcc_finite_blocklength_v1`.
- The old active SINR-only packet success surrogate is not active.
- Link reliability, latency, and energy remain coupled by the same
  finite-blocklength transmission attempt model.
- Inverse reliability caps are explicit diagnostics, not silent success.
- Failed scheduled messages keep scheduled latency and energy visible.
- Stage 4.4 expected-initiator PBFT distinguishes weak and central primary
  cases.
- Full graph remains a baseline, not an oracle.
- Boundary shorthand: full graph remains a baseline.
- No reward, training, actor, critic, COMA, GNN, LSTM, or v5 migration is
  introduced.

## Implemented Interfaces

Active audit id:

`stage_4_8_communication_consensus_boundary_audit`

Public interfaces:

- `STAGE4_8_BOUNDARY_AUDIT_STAGE_ID`
- `STAGE4_8_REQUIRED_CASES`
- `Stage48BoundaryAuditConfig`
- `Stage48BoundaryAuditRow`
- `build_stage4_8_boundary_audit_report`

Replay helper:

`scripts/replay/stage4_8_boundary_audit_report.py`

The replay helper prints JSON only and does not write result files.

## Boundary Cases

The report includes:

- `near_threshold_link`
- `deadline_tight_retransmission`
- `unreachable_reliability_target`
- `weak_edge_primary`
- `center_vs_edge_primary`
- `interference_full_graph_penalty`
- `sparse_resource_efficient`
- `failed_scheduled_message`

Each row reports:

- topology name;
- selected edge count;
- link deadline delivery probability;
- network deadline delivery probability;
- per-primary reliability;
- `consensus_success_probability`;
- scheduled latency;
- successful-delivery latency;
- energy;
- reliability feasibility;
- diagnostic flags.

## Latency Semantics

Stage 4.8 separates Stage 3 network latency into:

- `network_scheduled_latency_s`: route or broadcast occupancy from scheduled
  communication.
- `network_successful_delivery_latency_s`: latency visible only for positive
  delivery probability.
- `network_latency_s`: compatibility alias for
  `network_successful_delivery_latency_s`.

Stage 4 protocol accounting and message-matrix deadline gating use
`network_scheduled_latency_s`. This prevents a failed but scheduled message
from appearing to have no time cost.

## Inverse Reliability Diagnostics

`solve_required_transmission_time_for_reliability` returns:

- `required_transmission_time_s`
- `required_reliability_met`
- `required_transmission_time_capped`
- `success_probability_at_required_time`
- `required_time_search_max_s`

If a target reliability cannot be reached within the configured maximum search
time, the returned time may be capped but `required_reliability_met` is false
and `required_transmission_time_capped` is true.

## Metric Governance

Stage 4.8 does not add metric names. The audit report maps rows only to:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Per-primary reliability, deadline delivery probability, scheduled latency,
successful-delivery latency, inverse-cap flags, and full-graph-baseline status
are diagnostics, not new metrics.

## PBFT Boundary

Consensus reliability remains:

`pbft_expected_initiator_mean_field_v1`

The audit uses Stage 4.4 expected-initiator PBFT reliability. It does not use a
single fixed primary as topology-level reliability, does not use Monte Carlo,
does not use random sampling, and does not enumerate success subsets.

## Tests

Implemented tests cover:

- all required cases are present;
- at least one consensus reliability value is non-saturated;
- failed scheduled messages have positive scheduled latency and energy while
  successful-delivery latency is zero;
- unreachable inverse reliability targets expose cap diagnostics;
- weak-primary and center-primary cases are distinguishable;
- full graph is a baseline and not an oracle;
- metric rows are registered and no new metric names are introduced;
- replay script is print-only;
- source scan rejects v5, old metric aliases, reward implementation, training,
  actor, critic, COMA, MAPPO, Monte Carlo, random sampling, and subset
  enumeration routes.

## Residual Risks

- The boundary audit is deterministic and small; it is not a full scenario
  calibration suite.
- Synthetic PBFT matrices are used for weak-primary and center-primary tests to
  isolate protocol math from Stage 3 routing details.
- The Stage 4.4 reliability formula remains a mean-field approximation.
- Stage 5 must freeze reward/objective semantics before any reward
  implementation or training.
