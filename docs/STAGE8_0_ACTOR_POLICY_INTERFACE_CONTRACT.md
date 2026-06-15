# Stage 8.0 Actor Policy Interface Contract With Owner Approval

## Controlled Object

The controlled object is the deployment actor policy interface boundary for
MARL-Topology.

Stage 8.0 defines how a future decentralized policy may consume local actor
rows and emit local directed edge scores. It does not implement an actor network,
critic network, graph encoder, recurrent encoder, learner update, checkpoint,
training run, reward-weight calibration, final `tau_consensus` selection, or
v5 migration.

## Desired State

The project has a policy-facing interface that is narrow enough to preserve the
Dec-POMDP deployment boundary and explicit enough to guide the next
implementation task.

Design verdict:

```text
actor_policy_interface_contract_frozen_implementation_blocked
implementation_allowed = false
training_execution_allowed = false
model_code_added = false
owner_decision_required = true
```

## Baseline

Stage 6.1 created `ActorSafeBatch` rows from `ActorObservation` values.
Stage 7 completion generated learning evidence with separated
`actor_safe_view`, `critic_centralized_view`, `learning_target_view`, and
`diagnostics_view`.

Stage 8.0 connects those boundaries at contract level only.

Stage 8.0 does not implement actor, critic, graph, recurrent, checkpoint, or training code.

## Actor Input Interface

Policy input schema:

```text
actor_policy_local_input_v1
```

Allowed fields are exactly the Stage 6.1 actor-safe fields:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

The policy interface must reject:

- global topology or complete candidate graph state;
- selected topology, selected edge ids, joint action, or oracle labels;
- objective metrics such as `consensus_success_probability`, `latency`, or
  `energy`;
- topology diagnostics and per-primary reliability diagnostics;
- learning targets, edge-delta targets, surrogate components, and future
  outcomes;
- centralized critic or training-only views.

Local neighbor observations may include local incident-edge estimates already
declared by the Dec-POMDP schema, such as distance, local link success
probability, estimated link latency, and estimated link energy. These are local
observations, not objective labels.

## Actor Output Interface

Active policy output schema after Stage 9.0 schema unification:

```text
actor_policy_local_edge_score_output_v1
```

The actor emits local edge-score records only:

- `agent_id`
- `neighbor_id`
- `edge_id`
- `directed_edge_id`
- `score`
- optional `probability`
- `score_source`
- `time_step`

The legacy local edge-decision schema remains:

```text
actor_policy_local_edge_decision_v1
```

`activate` belongs only to the legacy local decision schema and to the
assembler selected topology after environment-side projection. It is not the
active actor model output.

The actor must not output global topology rows, final selected edge sets,
oracle labels, objective metrics, learning targets, or protocol diagnostics.
Joint topology assembly and candidate-edge validation remain environment-side.

## State And History Boundary

The interface permits future stateless, local-history, or permitted-message
policies.

Any future memory state must be derived only from actor-allowed local fields or
permitted local messages. A global recurrent state is forbidden for deployment
actors.

## Critic Training Interface Reference

Stage 8.0 does not implement a critic. It references the future training-only
boundary:

- `critic_centralized_view` may be used by a future critic after owner
  approval;
- `learning_target_view` may provide future target signals;
- `diagnostics_view` may support audits and reports;
- none of those views may enter actor deployment input, actor export, or actor
  replay projection.

## Learning Target Boundary

Stage 7 edge-delta targets, oracle candidates, objective metrics, and surrogate
components are training-only or diagnostic-only. They can supervise or evaluate
future learning, but they are not actor observations.

Stage 8.0 does not select a credit-assignment method.

## Acceptance Criteria

- Structured Stage 8.0 contract exists and is runnable.
- Actor input schema is exactly the Stage 6.1 actor-safe field set.
- Actor output schema is local edge scores only.
- `activate` is restricted to legacy local decisions or assembler-selected
  topology.
- Centralized critic and learning-target views remain training-only.
- Oracle labels and objective metrics remain outside actor input.
- Implementation, training, checkpoint creation, artifact export, reward-weight
  calibration, final tau selection, and v5 migration remain blocked.

## Verification Commands

```powershell
python scripts\replay\stage8_0_actor_policy_interface_contract.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

- The interface does not choose a model family.
- The interface does not prove that actor-local features are predictive enough
  for a learned policy.
- The next implementation task should create a minimal non-learning policy
  interface skeleton, still without actor/critic model code or training.
