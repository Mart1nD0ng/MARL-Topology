# Stage 4.6 Protocol Latency And Energy Accounting Review

## Controlled Object

The controlled object is the Stage 4.6 protocol accounting layer:

`src/marl_topology/protocol/pbft_accounting.py`

It consumes Stage 3 `NetworkCommunicationRecord` objects grouped by PBFT
phase and accounts protocol-level `latency` and `energy` without changing PBFT
reliability, reward, actor, critic, or training behavior.

## Desired State

Stage 4.6 should make PBFT communication cost observable and explicit:

- reliability remains produced only by the Stage 4.2/4.4 PBFT reliability
  evaluators;
- `network_delivery_probability` remains Stage 3 communication evidence;
- protocol latency is derived from scheduled phase communication records;
- protocol energy is derived from scheduled attempted communication records;
- failed, late, or zero-delivery communication is not silently converted to
  zero cost when a scheduled attempt consumed energy;
- only registered metric concepts are reported: `latency`, `energy`, and
  `topology_diagnostics`.

## Implemented Interfaces

Active accounting model id:

`stage4_pbft_protocol_accounting_v1`

Public interfaces:

- `PBFT_PROTOCOL_ACCOUNTING_MODEL_ID`
- `PBFTPhaseAccountingRecord`
- `PBFTProtocolAccountingRecord`
- `account_pbft_protocol_latency_energy`

## Inputs

- `node_ids`: consensus committee node ids.
- `phase_records`: mapping from PBFT phase name to Stage 3 network records.
- `phase_budgets`: `PBFTPhaseBudgets` with `pre_prepare`, `prepare`, and
  `commit` budgets.

Valid phase names are:

- `pre_prepare`
- `prepare`
- `commit`

## Latency Accounting

Per phase:

```text
max_scheduled_latency_s = max(record.network_scheduled_latency_s for scheduled records)
phase_latency_s = min(max_scheduled_latency_s, phase_budget_s)
```

Empty phases have zero phase latency.

Round-level protocol latency:

```text
latency = sum_phase phase_latency_s
```

This mode is recorded as:

`phase_max_clipped_to_budget`

The clipping means a late phase consumes its declared phase budget for this
review sensor. The delivery probability for late messages remains handled by
the Stage 4.3 message-matrix adapter and PBFT reliability evaluator.

## Energy Accounting

Per phase:

```text
phase_energy_j = sum(record.network_energy_j for scheduled records)
```

Round-level protocol energy:

```text
energy = sum_phase phase_energy_j
```

This mode is recorded as:

`scheduled_attempt_energy_sum`

Energy is counted for scheduled communication attempts even when the message is
late or has zero delivery probability. Disconnected records with no attempted
transmission still contribute zero because Stage 3 network records report zero
energy.

## Stage 4.8 Boundary Update

Stage 4.8 makes the Stage 3 network latency split explicit:

- `network_scheduled_latency_s`: scheduled occupancy visible to protocol
  accounting.
- `network_successful_delivery_latency_s`: success-conditioned latency view.
- `network_latency_s`: compatibility alias for
  `network_successful_delivery_latency_s`.

Protocol accounting uses scheduled latency, not the compatibility alias. This
keeps failed scheduled messages from becoming zero-cost communication.

## Diagnostics

Each phase records:

- scheduled record count;
- scheduled message count;
- deadline-eligible message count;
- deadline-filtered message count;
- zero-delivery message count;
- maximum scheduled latency;
- clipped phase latency;
- scheduled attempt energy.

Diagnostics are exported through `topology_diagnostics`. They are not new
metric names.

## Boundary

Stage 4.6 does not export `consensus_success_probability`.

Stage 4.6 does not implement reward.

Stage 4.6 does not:

- alter Stage 3 channel, link, or network records;
- alter Stage 4 PBFT reliability;
- introduce timeout, quorum, or effective-success aliases;
- train models;
- implement actor, critic, COMA, GNN, or LSTM behavior;
- migrate or run v5 code.

## Stage 4.5 Integration

Stage 4.5 baseline and oracle-candidate review now uses Stage 4.6 protocol
accounting for its `latency` and `energy` fields.

The active diagnostics are:

- `stage4_6_sum_phase_max_clipped_to_budget`
- `stage4_6_sum_scheduled_attempt_energy`

This replaces the earlier review-only `sum_all_directed_pair_records_times_three_phases`
placeholder.

## Tests

Implemented tests cover:

- protocol latency sums phase-max clipped latency;
- protocol energy sums scheduled attempt energy;
- late messages consume phase budget and keep scheduled energy visible;
- zero-delivery records are diagnostics, not consensus reliability;
- metric rows use only registered names;
- Stage 4.5 uses Stage 4.6 accounting;
- oracle-labelled records and self-message records are rejected;
- source scan rejects v5, reward, training, actor, critic, COMA, MAPPO, and old
  effective-success routes.

## Residual Risks

- The accounting mode is a deterministic review sensor, not a detailed PBFT
  scheduler.
- Parallelism within broadcast and multicast phases is simplified to phase-max
  latency.
- Energy is scheduled-attempt energy, not success-conditioned energy.
- Queueing and contention beyond Stage 3 records remain deferred.
