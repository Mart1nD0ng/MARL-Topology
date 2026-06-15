# Stage 9.0 Local MLP Edge Scorer Baseline Scaffold

## Controlled Object

The controlled object is the first learnable actor scaffold surface:
`LocalMLPEdgeScorer` maps Stage 6.1 actor-safe local observations to Stage 8
directed edge-score outputs.

## Desired State

Stage 9.0 introduces a minimal local MLP edge scorer scaffold while preserving
the Stage 8 environment-side topology assembler boundary.

Desired state:

```text
stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution
active_actor_output_schema = actor_policy_local_edge_score_output_v1
model_family = local_mlp
training_execution_allowed = false
checkpoint_io_allowed = false
artifact_writes_allowed = false
```

## Baseline

Stage 8 closed with actor outputs as directed edge scores, an environment-side
topology assembler, and centralized critic interfaces separated as
training-only. Before Stage 9.0, the code still carried a legacy
`actor_policy_local_edge_decision_v1` schema from Stage 8.0. Stage 9.0 unifies
the active actor output schema so edge-score schema is the main output.

activate belongs only to legacy local decisions or assembler-selected topology.
The active actor model must not emit `activate`.

## Actor Schema

Input remains exactly the Stage 6.1 actor-safe schema:

- `agent_id`;
- `agent_kind`;
- `time_step`;
- `local_position_m`;
- `local_neighbor_observations`;
- `local_messages`;
- `local_history`.

Forbidden deployment actor inputs remain global graph state, selected topology,
critic features, oracle labels, edge-delta targets, reward surrogate fields,
registered objective metrics, Stage 4 PBFT results, and future outcomes.

## Feature Schema

`LocalMLPEdgeScorer` uses only local actor and incident-neighbor fields:

- `agent_kind_is_vehicle`;
- `agent_kind_is_rsu`;
- `neighbor_kind_is_vehicle`;
- `neighbor_kind_is_rsu`;
- `local_position_x_m`;
- `local_position_y_m`;
- `local_position_z_m`;
- `distance_3d_m`;
- `link_success_probability`;
- `estimated_link_latency_s`;
- `estimated_link_energy_j`.

This schema is deliberately small and local. It does not include global
topology, oracle labels, objective values, reward surrogate values, or critic
outputs.

## Model Scaffold

The scaffold is:

- `LocalMLPEdgeScorerConfig`;
- `LocalMLPEdgeScorer`;
- `encode_actor_local_edges`;
- `build_stage9_0_local_mlp_report`.

The scorer uses PyTorch as an allowed Stage 9 model dependency. Parameters are
frozen by default for Stage 9.0 and inference uses no-gradient evaluation.

## Output Schema

Active output schema:

```text
actor_policy_local_edge_score_output_v1
```

Output records are `EdgeScoreRecord` values with:

- `agent_id`;
- `neighbor_id`;
- `edge_id`;
- `directed_edge_id`;
- `score`;
- optional `probability`;
- `score_source`;
- `time_step`.

Hard activation remains environment-side. The topology assembler consumes edge
scores and declared constraints to produce selected directed topology.

## Explicit Non-Implementation List

Stage 9.0 has:

- no training execution;
- no optimizer.step;
- no checkpoint;
- no training artifact;
- no PPO/MAPPO/COMA;
- no GNN/GRU/LSTM/Transformer;
- no replay writer;
- no reward-weight calibration;
- no final tau selection;
- no v5 migration.

## Sensors

Sensors:

- unit tests for active edge-score output schema;
- unit tests for local feature encoding;
- unit tests for MLP edge-score output;
- source scan for training/checkpoint/advanced-architecture routes;
- contract tests for docs, harness task, and project state;
- full pytest;
- harness task validation.

## Coupling Map

Actor-safe batch rows feed `ActorPolicyInput`. `LocalMLPEdgeScorer` encodes
only local incident-neighbor features and emits `EdgeScoreBatch`.
`ConflictAwareGreedyAssembler` or another environment-side assembler owns hard
activation and selected directed topology. Stage 3 communication and Stage 4
PBFT remain downstream of selected topology and message matrices, not raw MLP
scores.

## Acceptance Criteria

- Active actor output schema is `actor_policy_local_edge_score_output_v1`.
- `activate` is not part of the active actor model output.
- `LocalMLPEdgeScorer` produces `EdgeScoreBatch` from actor-safe local inputs.
- PyTorch usage is limited to the local MLP scaffold.
- No training, checkpoint, artifact, PPO/MAPPO/COMA, GNN/GRU/LSTM/Transformer,
  final tau selection, or v5 migration is introduced.

## Verification Commands

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

The scaffold is untrained and not evidence of predictive quality. The next
owner-approved task should be
`stage_9_1_supervised_edge_scoring_warm_start_plan_without_training_execution`
or a similarly bounded preflight before any training execution.
