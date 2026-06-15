# Policy Architecture Contract

## Current Status

The active policy architecture contract is design-only.

Authoritative Stage 5.7 document:

- `docs/STAGE5_7_POLICY_ARCHITECTURE_CONTRACT.md`

Structured manifest:

- `src/marl_topology/policies/architecture_contract.py`

## Execution Boundary

Stage 5.7 does not implement actor networks, critic networks, checkpoint
loading, training loops, reward-weight calibration, final `tau_consensus`
selection, or v5 migration.

Existing policy code remains limited to non-learning baselines.

## Deployment Actor Boundary

Deployment actors may consume only the `ActorObservation` local fields declared
by the Dec-POMDP contract.

Forbidden deployment inputs include global topology, oracle labels, registered
evaluation metrics, surrogate diagnostics, future outcomes, centralized
training views, and per-primary reliability diagnostics.

## Centralized Training Boundary

Centralized training views are allowed only for future training-only critics.
They must not leak into actor input batches, actor export, checkpoint actor
serialization, or deployment evaluation.

## Credit Boundary

Credit assignment is not selected in Stage 5.7. Future credit mechanisms must
be calibrated against registered objective evidence and must not introduce
local reward signals that conflict with the consensus reliability constraint
and latency/energy objectives.

## Stage 8.0 Actor Policy Interface

Stage 8.0 freezes the policy-facing interface without implementing a policy
model.

Authoritative Stage 8.0 document:

- `docs/STAGE8_0_ACTOR_POLICY_INTERFACE_CONTRACT.md`

Structured manifest:

- `src/marl_topology/policies/interface_contract.py`

Deployment policy inputs are exactly the Stage 6.1 `ActorSafeBatch` fields.
After Stage 9.0 schema unification, active deployment policy outputs are local
directed edge-score batches only. The legacy local edge-decision schema still
exists for older non-learning baselines, but `activate` belongs only to that
legacy schema or to assembler-selected topology. Joint topology assembly remains
environment-side.

Stage 7 evidence views remain separated:

- `actor_safe_view` can become policy input;
- `critic_centralized_view` is future training-only context;
- `learning_target_view` is future training-only target evidence;
- `diagnostics_view` is audit-only evidence.

Stage 8.0 still does not implement actor networks, critic networks, learner
updates, checkpoint creation, training execution, reward-weight calibration,
final `tau_consensus` selection, or v5 migration.

## Stage 9.0 Local MLP Edge Scorer Scaffold

Stage 9.0 is the first owner-approved actor model scaffold. It adds only a
local MLP edge scorer that maps actor-safe local observations to the active
edge-score output schema. It does not train, create checkpoints, write training
artifacts, implement PPO/MAPPO/COMA, or implement GNN, GRU, LSTM, or
Transformer modules.
