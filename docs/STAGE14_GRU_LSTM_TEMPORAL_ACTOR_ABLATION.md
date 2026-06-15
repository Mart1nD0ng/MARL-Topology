# Stage 14 GRU/LSTM Temporal Actor Ablation

## Controlled Object

The controlled object is local temporal actor memory under the Dec-POMDP actor
boundary.

## Desired State

Test `LocalGRUEdgeScorer` first. If GRU passes fixture sanity, test
`LocalLSTMEdgeScorer` as a replacement. Use only actor-safe local edge history.

## State Variables

- Sequence feature tensor, target tensor, mask, time order, and hidden state.
- GRU initial/final loss and output-collapse flag.
- LSTM initial/final loss and output-collapse flag.
- Future leakage flag and hidden reset status.
- Best Stage 15 actor candidate.

## Sensors

- Unit tests for time order, no future leakage, variable masks, hidden reset,
  and edge-score output shape.
- Smoke script: `python scripts\train\stage14_temporal_actor_ablation.py`.
- Full pytest and harness validation.

## Actuators

- `src/marl_topology/models/local_gru_edge_scorer.py`
- `src/marl_topology/models/local_lstm_edge_scorer.py`
- `src/marl_topology/training/sequence_batching.py`
- Stage 14 script, docs, tests, and self-review.

## Disturbances

The real Stage 7 evidence has only `time_step = 0`. Stage 14 therefore uses a
controlled actor-safe temporal fixture for implementation and leakage sensors,
but does not promote temporal models over the Stage 13 GNN for Stage 15.

## Coupling Map

Actor-safe local edge features become ordered sequences. GRU and LSTM consume
only current and past local features. Their outputs are edge scores at each
time step; environment-side assembler ownership is unchanged.

## Acceptance Criteria

- Sequence batching preserves time order.
- Future leakage checks pass.
- Hidden state reset works.
- Variable sequence masks work.
- GRU is tested before LSTM.
- LSTM is tested only after GRU sanity passes.
- No global actor memory, Transformer, PPO/MAPPO, COMA, checkpoint, final tau
  selection, reward-weight calibration, or v5 migration.

## Verification

```powershell
python scripts\train\stage14_temporal_actor_ablation.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

Temporal evidence is fixture-level only. Real multi-step Stage 7 evidence is
needed before selecting GRU or LSTM for deployment or scaling.
