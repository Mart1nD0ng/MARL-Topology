# Stage 13 Local GNN Edge Scorer

## Controlled Object

The controlled object is the actor architecture comparison between the Stage 11
Local MLP edge scorer and a local ego-graph edge scorer.

## Desired State

Implement and evaluate `LocalGNNEdgeScorer` while preserving Dec-POMDP actor
boundaries. The actor graph is only the local incident candidate-edge set for
one actor sample. Outputs remain edge scores only.

## State Variables

- Validation edge loss.
- Pairwise ranking accuracy.
- Assembler-level feasibility, latency, and energy diagnostics.
- Parameter count and runtime.
- Actor graph scope and leakage flags.

## Sensors

- Unit tests for local group aggregation, variable neighbor sets, output schema,
  and comparison report fields.
- Smoke script: `python scripts\train\stage13_supervised_gnn_actor.py`.
- Full pytest and harness validation.

## Actuators

- `src/marl_topology/models/local_gnn_edge_scorer.py`
- grouped actor tensor support;
- Stage 13 comparison script, docs, tests, and self-review.

## Disturbances

The Stage 7 evidence is small and contains topology-variant label conflicts.
If GNN underperforms MLP, the result is reported as a diagnostic rather than
hidden by extra complexity.

## Coupling Map

Actor-safe local rows become grouped edge tensors. The GNN aggregates only
within each actor sample's incident-edge group. Scores feed the existing
assembler. No global topology, oracle label, or critic output enters actor
input.

## Acceptance Criteria

- GNN implemented and tested.
- MLP-vs-GNN comparison report produced.
- No global actor graph, GRU, LSTM, Transformer, PPO/MAPPO, COMA, checkpoint,
  final tau selection, reward-weight calibration, or v5 migration.
- Best current actor selected as Stage 14 starting point with a reason.

## Verification

```powershell
python scripts\train\stage13_supervised_gnn_actor.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

Local GNN evidence is limited by the current small supervised dataset. A future
stage may need richer mobility or neighborhood diversity before promoting GNN
as the default actor.
