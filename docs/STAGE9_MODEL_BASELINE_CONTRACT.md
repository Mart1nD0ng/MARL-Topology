# Stage 9 Forward-Only Model Baseline Contract

## Controlled Object

The controlled object is the first forward-only learnable model stack:
`LocalMLPEdgeScorer` actor logits, `CentralizedMLPCriticBaseline` training-only
heads, tensorizers, registry entries, and an assembler dry-run.

## Desired State

Stage 9 keeps the active actor output schema as
`actor_policy_local_edge_score_output_v1`. The actor consumes only actor-safe
local incident-edge fields and emits logits/scores. The environment-side
assembler owns hard topology selection. The centralized critic is
training-only and exposes value, feasibility, consensus, latency, energy, and
edge-delta heads.

## State Variables

- Actor input fields: Stage 6.1 actor-safe fields only.
- Actor output fields: `EdgeScoreBatch` records only.
- Critic input fields: centralized training-only topology context.
- Critic output heads: value, feasibility, consensus probability, latency,
  energy, add/remove/keep edge-delta predictions.
- Forbidden execution state: optimizer, backward pass, checkpoint, training
  artifact, PPO/MAPPO/COMA, GNN/GRU/LSTM/Transformer.

## Sensors

- Unit tests for tensorization, actor output schema, critic heads, registry,
  assembler dry-run, and forbidden-field rejection.
- Smoke script: `python scripts\replay\stage9_0_model_forward_smoke_report.py`.
- Full verification: `python -m pytest -q` and
  `python harness\scripts\validate_tasks.py`.

## Actuators

- `src/marl_topology/models/tensorizers.py`
- `src/marl_topology/models/local_mlp_edge_scorer.py`
- `src/marl_topology/models/centralized_mlp_critic.py`
- `src/marl_topology/models/model_registry.py`
- Stage 9 smoke script and tests.

## Disturbances

Legacy activate-style action records may be confused with neural actor output.
Centralized critic fields may accidentally leak into actor tensors. Model
scaffolds may drift into training loops before loss and manifest gates exist.

## Coupling Map

Actor-safe rows become local edge tensors. The actor maps those tensors to
directed edge scores. `ConflictAwareGreedyAssembler` projects scores into a
selected topology. The critic consumes centralized training-only evidence and
must not be serialized into deployment actor inputs.

## Acceptance Criteria

- Actor score forward pass succeeds.
- Critic multi-head forward pass succeeds.
- Assembler dry-run consumes actor scores.
- Active actor output schema has no `activate` field and no selected topology.
- No optimizer, backward pass, checkpoint, training artifact, COMA,
  PPO/MAPPO, GNN, GRU, LSTM, Transformer, final tau selection, reward-weight
  calibration, or v5 migration is introduced.

## Verification

```powershell
python scripts\replay\stage9_0_model_forward_smoke_report.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

The MLP and critic are untrained. Stage 10 must compute supervised losses
without parameter updates before any training stage is allowed.
