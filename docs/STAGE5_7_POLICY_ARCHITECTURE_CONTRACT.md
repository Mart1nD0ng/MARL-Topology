# Stage 5.7 Policy Architecture Contract Without Implementation

## Controlled Object

The controlled object is the future policy and centralized-training
architecture boundary for MARL-Topology.

Stage 5.7 freezes what future actor and critic designs may access. It does not
implement actor networks, critic networks, graph encoders, recurrent encoders,
checkpoint loading, training loops, reward-weight calibration, final
`tau_consensus` selection, or v5 migration.

## Desired State

The project has an architecture contract that is specific enough to prevent
Dec-POMDP leakage but still avoids prematurely choosing a model family.

Design verdict:

```text
architecture_contract_frozen_implementation_blocked
implementation_allowed = false
training_execution_allowed = false
model_code_added = false
owner_decision_required = true
```

## Baseline

Stage 5.6 froze the training design contract but left actor and critic
architecture as future work. Stage 5.7 resolves that at contract level only.

Existing policy code remains limited to non-learning baselines.

## Deployment Actor Contract

Deployment actors may use only:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

Deployment actors must not use:

- global topology;
- Full candidate-graph state;
- all-node positions;
- oracle labels;
- registered evaluation metrics;
- surrogate diagnostics;
- future outcomes;
- centralized training views;
- per-primary reliability diagnostics.

Actor output contract:

```text
local EdgeActionDecision values only
```

History rule:

```text
local history or permitted message history only
```

The deployment actor must be loadable and runnable without critic-only tensors
or centralized training views.

## Future Actor Options

Stage 5.7 allows these future design options, but implements none of them:

- local reactive edge scorer;
- local-history edge scorer;
- local message-aggregation edge scorer.

Each option must consume only deployment-allowed actor fields. Any future
memory feature must state whether it is local history, permitted message
history, or training-only critic state.

## Centralized Training Boundary

Centralized training views may be used only for training-only critics after a
future owner-approved implementation task.

Allowed future training-only fields include:

- `scenario_id`
- `node_ids`
- `candidate_edge_ids`
- `selected_edge_ids`
- `joint_action`
- `global_topology`
- registered evaluation metrics;
- surrogate diagnostics;
- oracle status.

These fields must not enter deployment actor inputs, actor export, actor
checkpoint serialization, or actor replay projection.

## Credit Assignment Boundary

Stage 5.7 does not select a credit-assignment mechanism.

Future review paths may include:

- centralized value estimator;
- counterfactual edge-credit review;
- difference-against-baseline review.

Required future calibration checks:

- ranking fidelity;
- sign accuracy;
- rare-failure recall;
- magnitude error;
- oracle edge-hit rate.

No local reward override is allowed if it fights the registered global
constraint and latency/energy objectives.

## Serialization Boundary

Checkpoint creation remains blocked.

Future serialization checks must prove:

- actor export contains only deployment inputs;
- critic state is not required for actor loading;
- feature schema hash is recorded;
- contract ids are recorded;
- metric registry version is recorded.

## Required Leakage Tests

Future architecture implementation must include negative tests showing:

- actor schema rejects global topology;
- actor schema rejects metrics and surrogate diagnostics;
- actor schema rejects oracle labels;
- actor schema rejects future outcomes;
- deployment export excludes centralized training views;
- replay actor projection excludes training-only columns.

## Acceptance Criteria

- Structured Stage 5.7 contract exists and is runnable.
- Implementation remains blocked.
- Training execution remains blocked.
- Deployment actor boundary remains local-only.
- Centralized training fields are training-only.
- Credit assignment is not selected.
- Checkpoint creation remains blocked.
- Source does not introduce model, training, checkpoint, v5, or legacy metric
  paths.

## Residual Risks

- The contract does not choose a model family or credit mechanism.
- Future Stage 5.8 still needs a learning-target and replay contract before
  learner implementation.
- Scenario distribution remains alpha and deterministic unless a later stage
  calibrates deployment scenarios.
