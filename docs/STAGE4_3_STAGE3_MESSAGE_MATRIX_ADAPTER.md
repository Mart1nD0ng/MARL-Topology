# Stage 4.3 Stage 3 Message-Matrix Adapter

## Controlled Object

The controlled object is the Stage 4.3 adapter:

`src/marl_topology/protocol/message_matrix_adapter.py`

It converts Stage 3 `NetworkCommunicationRecord` objects into declared PBFT
phase message-delivery matrices.

## Desired State

Stage 4.3 should connect Stage 3 communication records to Stage 4.2 PBFT
reliability without collapsing the layer boundary.

Desired properties:

- consume Stage 3 network communication records;
- produce `pre_prepare`, `prepare`, and `commit` message matrices;
- apply phase latency budgets as delivery gates;
- preserve network delivery as communication evidence, not a consensus metric;
- keep latency, energy, timeout, reward, and consensus reliability separate;
- optionally call Stage 4.2 after producing declared matrices;
- avoid training, actor/critic, COMA, GNN, LSTM, v5 code, and reward logic.

## Implemented Interfaces

- `MESSAGE_MATRIX_ADAPTER_ID`
- `PBFTPhaseBudgets`
- `PBFTMessageMatrices`
- `build_pbft_message_matrices_from_network_records`
- `evaluate_pbft_reliability_from_network_records`

Active adapter id:

`stage4_stage3_network_to_pbft_matrix_v1`

## Inputs

- `node_ids`
- `phase_records`: mapping from phase name to Stage 3 network records
- `phase_budgets`

Each phase name must be one of:

- `pre_prepare`
- `prepare`
- `commit`

Stage 3 records provide:

- `source_id`
- `target_ids`
- `network_delivery_probability`
- `network_scheduled_latency_s`
- `network_successful_delivery_latency_s`

## Matrix Rule

For each Stage 3 record and each target:

```text
M_phase[source, target] =
    network_delivery_probability
    if network_scheduled_latency_s <= phase_budget_s
    else 0
```

The previous shorthand `network_latency_s <= phase_budget_s` is superseded by
the scheduled-latency gate above because `network_latency_s` is now a
successful-delivery latency alias.

If multiple records map to the same directed pair, the adapter keeps the maximum
delivery probability. Missing pairs remain absent and Stage 4.2 treats missing
directed entries as zero delivery.

## Diagnostics

`PBFTMessageMatrices` records:

- phase budgets;
- record count by phase;
- deadline-filtered count by phase;
- zero-delivery count by phase;
- `uses_stage3_network_records = true`;
- `exports_consensus_metric = false`.

The adapter intentionally does not expose `consensus_success_probability`.
Only Stage 4.2 may produce that registered metric concept.

## Tests

Implemented tests cover:

- Stage 3 records create expected directed matrix entries;
- late records are zeroed by phase budget;
- disconnected Stage 3 records produce zero matrix entries;
- same-resource interference produces lower matrix delivery than orthogonal
  resource delivery;
- zero-delivery zero-latency communication records do not become consensus
  success;
- wrapper evaluation through Stage 4.2 works for complete all-one records;
- invalid budgets and unknown nodes are rejected;
- source scan rejects v5, reward, training, actor/critic, COMA, GNN/LSTM, and
  old metric aliases.

## Boundary

Stage 4.3 does not:

- change Stage 3 channel, link, or network behavior;
- define a new metric;
- export CSV rows;
- compute protocol latency or energy;
- implement reward;
- run training;
- implement actor, critic, COMA, GNN, or LSTM;
- migrate v5 code.

## Stage 4.8 Boundary Note

Stage 4.8 changes the adapter deadline gate to scheduled network latency.
`network_latency_s` remains a compatibility alias for successful-delivery
latency and must not hide failed-but-scheduled message occupancy.

## Residual Risks

- The current adapter uses `network_scheduled_latency_s` only as a phase-gate
  condition; it does not define protocol round latency.
- The adapter uses max probability when multiple communication records map to
  the same directed pair; future route diversity may need a richer composition
  rule.
- Stage 4.4 must evaluate baseline and oracle candidates carefully so full
  graph remains a baseline, not an oracle.
