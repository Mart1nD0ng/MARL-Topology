# Stage 6.1 Actor-Safe Batch Builder Without Model Or Training

## Controlled Object

The controlled object is the boundary between local `ActorObservation` payloads,
mixed replay/report rows, and future actor-facing batches.

Stage 6.1 implements only in-memory actor-safe batch projection. It does not
write datasets, write artifacts, create checkpoints, instantiate actor or
critic models, execute training, calibrate reward weights, select final
`tau_consensus`, or migrate v5 code.

## Desired State

The project can produce a batch that contains only deployment actor input
fields:

```text
agent_id
agent_kind
time_step
local_position_m
local_neighbor_observations
local_messages
local_history
```

Verdict:

```text
actor_safe_batch_builder_ready_model_training_blocked
```

## Inputs

Accepted sources:

- `ActorObservation` instances;
- mixed rows that pass the replay schema and can be projected to actor-safe
  fields.

Rejected sources:

- rows missing required actor fields;
- rows with extra direct actor fields;
- duplicate `(agent_id, time_step)` rows;
- unsupported replay columns such as future learning targets;
- direct actor rows containing global, oracle, metric, centralized, or reward
  training-only fields.

## Outputs

The builder returns an in-memory `ActorSafeBatch` with:

- `rows`;
- `field_names`;
- `row_count`;
- `summary()`.

The summary is report-safe and does not contain centralized training data,
evaluation metrics, oracle labels, future outcomes, checkpoints, or dataset
paths.

## Dec-POMDP Boundary

The actor-safe batch is deployment-local only. It must not contain:

- global topology;
- all-node positions;
- centralized training views;
- oracle labels;
- registered evaluation metrics;
- surrogate diagnostics;
- future outcomes;
- action labels as observations.

This stage proves local projection before any model-facing code exists.

## Acceptance Criteria

- Batches from valid `ActorObservation` values pass.
- Mixed rows project to actor-safe fields and drop centralized/evaluation
  diagnostics.
- Missing actor fields fail.
- Direct extra fields fail.
- Unsupported future learning-target columns fail.
- Duplicate `(agent_id, time_step)` rows fail.
- Replay script reports `batch_ready = true` and `writes_performed = false`.
- `result_save` remains `.gitkeep` only.
- No model, checkpoint, dataset export, artifact write, training execution,
  reward-weight calibration, final tau selection, or v5 migration occurs.

## Recommended Next Stage

`stage_7_0_local_actor_policy_interface_contract_with_owner_approval`

Reason: Stage 6 has completed the manifest guard and actor-safe batch boundary.
The next stage should decide the local actor policy interface before any model
implementation or training execution.
