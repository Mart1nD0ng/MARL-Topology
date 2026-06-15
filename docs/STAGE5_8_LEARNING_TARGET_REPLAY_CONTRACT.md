# Stage 5.8 Learning Target And Replay Contract Without Implementation

## Controlled Object

The controlled object is the future boundary between environment/report rows,
training-only learning targets, centralized training views, and deployment
actor batches.

Stage 5.8 freezes a replay and learning-target contract. It does not implement
a dataset writer, replay buffer, learner batch, target derivation, actor or
critic model, checkpoint logic, training loop, reward-weight calibration,
final `tau_consensus` selection, or v5 migration.

## Desired State

The project has explicit replay-column semantics before any learner or dataset
writer exists.

Design verdict:

```text
learning_target_replay_contract_frozen_implementation_blocked
dataset_writer_allowed = false
replay_buffer_allowed = false
learner_batch_allowed = false
training_execution_allowed = false
```

## Baseline

Stage 5.7 froze policy architecture boundaries and required replay actor
projection to exclude centralized training views, surrogate diagnostics, oracle
labels, registered evaluation metrics, per-primary reliability diagnostics,
and future outcomes.

Stage 5.8 resolves future learning-target names at contract level only. It
does not activate those fields in the current replay schema.

## Active Columns Remain Unchanged

Deployment actor input columns remain:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

Currently admitted training-only diagnostic columns remain:

- `reward_surrogate`
- `reward_reliability_penalty`
- `reward_latency_penalty`
- `reward_energy_penalty`
- `reward_config_id`

Stage 5.8 does not add active replay columns.

## Planned Learning-Target Columns

These columns are defined as future training-target fields, but remain
`planned_not_active`:

- `return`
- `advantage`
- `value_target`
- `episode_id`
- `trajectory_id`
- `transition_index`
- `discount_factor`
- `bootstrap_value`

Rules:

- They are not deployment actor inputs.
- They are not evaluation metrics.
- They require a future schema update before they can appear in replay data.
- They require owner approval before implementation.

The active replay schema must still reject `return`, `advantage`, and
`value_target` until a future owner-approved implementation task changes the
schema.

## Planned Diagnostic Columns

These training diagnostics are also `planned_not_active`:

- `policy_log_probability`
- `policy_entropy`
- `action_sparsity`
- `constraint_violation`

They are not evaluation metrics and must not replace registered objective
evidence.

## Actor Batch Contract

Actor batches may contain only deployment actor input columns. Actor batches
must exclude:

- planned learning-target columns;
- centralized training views;
- registered evaluation metrics;
- surrogate diagnostics;
- oracle labels;
- future outcomes.

## Centralized View Contract

Future centralized views are training-only and not active datasets in Stage
5.8.

Future allowed context may include scenario ids, node ids, candidate edge ids,
selected edge ids, joint actions, global topology, registered evaluation
metrics, and surrogate diagnostics.

The future centralized view must be dropped before actor projection.

## Derivation Policy

No target values are computed in Stage 5.8.

Future target derivation must declare:

- discount policy;
- bootstrap policy;
- episode boundary policy;
- normalization policy;
- single-step target policy;
- registered evaluation evidence used for success claims.

## Acceptance Criteria

- Structured Stage 5.8 contract exists and is runnable.
- Planned target columns are explicit.
- Active replay schema remains unchanged.
- `return`, `advantage`, and `value_target` remain unsupported by active replay
  validation.
- Planned target columns are not actor inputs.
- Dataset writer, replay buffer, learner batch, target derivation, checkpoint,
  and training execution remain blocked.
- Tests and harness validation pass.

## Residual Risks

- Stage 5.8 does not specify a concrete target derivation algorithm.
- Future implementation must decide episode boundaries and discounting before
  target values can be computed.
- Future learner code still needs owner approval and artifact/run-manifest
  policy.
