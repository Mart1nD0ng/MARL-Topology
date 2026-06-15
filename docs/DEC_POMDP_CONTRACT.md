# Dec-POMDP Contract

## Stage 2.1 CTDE Decision

The project uses Dec-POMDP deployment constraints with CTDE as the allowed training paradigm.

- Deployment actors are decentralized and may use only local observations, local history, and permitted local messages.
- Training may use centralized information through declared training-only interfaces.
- CTDE is a boundary, not an algorithm commitment. IPPO/MAPPO-style CTDE, COMA, Q critics, GNN actors, LSTM actors, and direct edge-delta critics remain future choices gated by separate evidence.
- No actor, critic, COMA, GNN, LSTM, reward, or training loop is implemented by this contract.

## Actor Allowed Observations

Deployment actors may use:

- Local agent state.
- Local link observations.
- Local neighbor messages available under the communication protocol.
- Local history or recurrent state derived only from allowed local observations.
- Agent id, role, or type if declared in the observation schema.
- Time-step or round index if available to every deployed actor.

Stage 2.1 actor schema fields:

| Field | Meaning | Deployment status |
| --- | --- | --- |
| `agent_id` | Local agent identifier | allowed |
| `agent_kind` | Local role such as vehicle or RSU | allowed |
| `time_step` | Shared round index | allowed |
| `local_position_m` | Local agent 3D position in meters | allowed |
| `local_neighbor_observations` | Incident candidate-edge observations only | allowed |
| `local_messages` | Permitted local messages | allowed |
| `local_history` | History derived only from allowed local fields | allowed |

## Actor Forbidden Global Information

Deployment actors must not use:

- Full global topology.
- Full positions of all vehicles, RSUs, or buildings unless locally observable by contract.
- Future mobility, future channel state, or future consensus outcome.
- Centralized critic features.
- Oracle feasibility labels.
- Full committee success labels during action selection.
- Evaluation-only metrics such as episode-level consensus success, latency, energy, or oracle diagnostics.

Stage 2.1 validators reject fields such as `global_topology`, `full_graph`, `candidate_graph`, `all_node_positions`, `future_consensus_outcome`, `critic_features`, `oracle_label`, `consensus_success`, `consensus_success_probability`, `latency`, `energy`, and `topology_diagnostics`.

## Centralized Critic Allowance

Training critics may use centralized information when declared:

- Global state.
- Full graph or topology candidate.
- Full scenario state.
- Joint actions.
- Global rewards and constraints.
- Counterfactual labels or oracle outputs if training-only.

Centralized information must be routed through training-only interfaces and excluded from deployment actor serialization.

Stage 2.1 names this training-only schema `CentralizedTrainingView`. It may contain scenario id, node ids, candidate edge ids, selected edge ids, metric names, and oracle status. It is not a deployment actor input.

## Action Schema

Stage 2.1 uses a local edge-decision schema:

- `EdgeActionDecision(agent_id, neighbor_id, activate)` is a local proposal about one incident candidate edge.
- `JointTopologyAction` is assembled environment-side from local decisions and candidate-edge validation.
- The actor does not receive oracle labels, full topology metrics, or full graph state in order to make this decision.
- Fixed threshold policy behavior is not part of this schema and remains a future baseline gate.

## Stage 2.2 Decentralized Non-Learning Baselines

Stage 2.2 adds deterministic, non-learning baseline rules that consume only `ActorObservation` and emit `EdgeActionDecision`.

Implemented local rules:

- `no_edges`: every local incident edge is inactive.
- `all_local_edges`: every local incident edge is proposed active.
- `reliability_threshold`: incident edges are proposed active only when local link success probability is greater than or equal to an explicit threshold parameter.
- `top_k_reliability`: each agent proposes up to `k` locally best reliability edges, with deterministic tie-breaking.
- `local_random`: seeded local random proposal rule with explicit edge probability.

Stage 2.2 does not implement deployment policy selection, actor models, critic models, reward learning, or training. Threshold values are explicit baseline parameters and are not deployment defaults.

Joint topology assembly:

- Local decisions are validated against the candidate graph environment-side.
- Non-candidate edge decisions fail validation.
- `JointTopologyAction` can use any-active or mutual-active local proposal rules.
- Full candidate graph may be produced by the all-local-edges baseline, but it remains a baseline and is not an oracle.

## Stage 2.3 Minimal Dec-POMDP Env Wrapper

Stage 2.3 adds `MinimalDecPOMDPEnv` as a reset/step wrapper around the existing scene, candidate graph, link records, Dec-POMDP schema, and topology evaluator.

`reset()`:

- returns one `ActorObservation` per graph node;
- sets the environment time step;
- returns environment-side metadata such as scenario id, physics regime, node count, candidate edge count, and horizon;
- does not expose global topology, oracle labels, critic features, or evaluation metrics to actor observations.

`step(local_decisions)`:

- accepts local `EdgeActionDecision` values;
- assembles a validated `JointTopologyAction` environment-side;
- rejects non-candidate edge decisions;
- evaluates the selected topology through `TopologyEvaluator`;
- may receive an optional environment-side topology label for unambiguous reporting;
- advances time and returns the next local observations;
- returns `TopologyEvaluation` to the environment caller, not to deployment actor observations;
- does not compute reward, does not train, does not store replay, and does not instantiate actor or critic models.

The wrapper is intentionally deterministic and single-scenario. It is a boundary for later environment work, not a full simulator.

## Stage 2.4 Baseline Report Boundary

Stage 2.4 uses `MinimalDecPOMDPEnv.reset()` and `step(local_decisions)` to evaluate decentralized non-learning baselines. The report may use unique environment-side topology ids for evidence, but those ids and all metric rows remain outside deployment actor observations.

## Stage 2.6 Replay Dataset Column Boundary

Stage 2.6 defines replay/dataset column classes before any replay writer or learner exists.

- Deployment actor input columns must be a subset of `ActorObservation`.
- Local action columns may be stored as decisions or labels, but are not observations.
- Centralized training-only columns may support future CTDE critics, but are excluded from deployment actors.
- Evaluation-only columns include metric rows, oracle evidence, and report identifiers.
- Reward, return, advantage, value-target, future outcome, and old v5 effective-success columns remain unsupported until future contracts admit them.

The code boundary lives in `src/marl_topology/data/replay_schema.py`.

## Deployment-Time Boundary

Before deployment evaluation:

- Actor feature schema must be frozen.
- Replay/dataset columns must be audited for forbidden fields.
- Checkpoint loading must verify actor and critic feature separation.
- Evaluation must run with actor-local inputs only.

## Required Leakage Tests

- Feature-schema test: actor input fields are a subset of allowed observations.
- Dataset test: deployment actor batch excludes centralized columns.
- Checkpoint test: actor model can load and run without critic-only tensors.
- Negative test: injecting a forbidden global field into actor schema fails validation.
- Replay test: policy replay uses local history, not full future trajectory.
- Stage 2.1 test: actor observation built from the demo scene includes only incident edge observations.
- Stage 2.1 test: centralized training view payload fails actor observation validation.
- Stage 2.1 test: local action assembly rejects non-candidate edges.
- Stage 2.3 test: `reset()` observations pass actor schema validation and contain no registered evaluation metrics.
- Stage 2.3 test: `step(local_decisions)` advances time and evaluates topology.
- Stage 2.3 test: `step()` before `reset()` fails.
- Stage 2.3 test: non-candidate local decisions fail.
- Stage 2.4 test: decentralized report rows use `ActorObservation` to `EdgeActionDecision` boundaries and unique baseline topology ids.
- Stage 2.6 test: deployment actor input column validation rejects global, oracle, metric, action, reward, and future columns.
- Stage 2.6 test: mixed replay rows can be classified and projected to actor-safe columns.

## Acceptance

- No actor architecture review passes without an observation boundary.
- No centralized critic claim is accepted without deployment leakage tests.

## Stage 5.5 Training Preflight Review

Stage 5.5 reviews training readiness without authorizing actor or critic
implementation.

Preflight conclusion:

- the existing deployment actor boundary is ready for a future design contract;
- the future actor-feature contract is still missing;
- the future critic/training-only interface contract is still missing;
- checkpoint and deployment serialization checks remain future gates;
- learning targets and replay columns remain blocked until a later contract.

No deployment actor may consume reward-surrogate diagnostics, objective metrics,
oracle labels, centralized training views, per-primary reliability diagnostics,
or future training targets.

Stage 5.5 therefore recommends only
`stage_5_6_training_design_contract_without_execution`; it does not implement
actor, critic, COMA, GNN, LSTM, checkpoint, or training code.

## Stage 5.6 Training Design Contract

Stage 5.6 freezes the design-only information boundary for future training.

Deployment actor fields remain exactly local:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

Deployment actor fields must still exclude global topology, oracle labels,
registered evaluation metrics, surrogate diagnostics, future outcomes, and
centralized training views.

Future centralized critics may use centralized state only through a separate
training-only architecture contract. Stage 5.6 does not implement actor, critic,
COMA, GNN, LSTM, checkpoint, or training code. The next recommended task is
`stage_5_7_policy_architecture_contract_without_implementation`.

## Stage 5.7 Policy Architecture Contract

Stage 5.7 freezes the design-only policy architecture boundary in:

- `docs/STAGE5_7_POLICY_ARCHITECTURE_CONTRACT.md`
- `docs/POLICY_ARCHITECTURE_CONTRACT.md`
- `src/marl_topology/policies/architecture_contract.py`

The verdict is:

```text
architecture_contract_frozen_implementation_blocked
```

Deployment actor inputs remain local-only and unchanged from the actor
observation contract. Centralized training views may be used only by future
training-only critic interfaces and must not enter actor input batches, actor
export, checkpoint actor serialization, or deployment evaluation.

Stage 5.7 permits future design review of local reactive, local-history, and
local message-aggregation actor options. It implements none of them.

Credit assignment is not selected. Future credit mechanisms must be calibrated
against registered objective evidence and must not introduce local reward
signals that fight the consensus reliability constraint or latency/energy
objectives.

## Stage 5.8 Learning Target And Replay Contract

Stage 5.8 freezes learning-target and replay-column semantics without
implementation.

Planned future columns such as `return`, `advantage`, `value_target`,
`trajectory_id`, `discount_factor`, and `bootstrap_value` are training-only
concepts. They are not deployment actor inputs and remain inactive in the
active replay schema.

Actor batches must remain subsets of `ActorObservation`. Centralized training
views, planned target columns, surrogate diagnostics, registered evaluation
metrics, oracle labels, and future outcomes must be excluded from deployment
actor projections.

## Stage 6.0 Minimal Training Stack Guard

Stage 6.0 adds only a dry-run stack guard around manifest validation and
contract references. It does not implement a deployment actor, centralized
critic, model, checkpoint loader, replay writer, batch builder, or training
execution.

The next implementation surface must still prove actor-safe projection before
any model-facing code is introduced.

## Stage 6.1 Actor-Safe Batch Builder

Stage 6.1 implements actor-safe in-memory batch projection.

Allowed batch fields are exactly:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

Mixed rows may be projected only after replay-schema validation. Centralized
training views, oracle labels, registered evaluation metrics, surrogate
diagnostics, future outcomes, and action labels must not enter actor batch rows.

Stage 6 is closed after this projection boundary. Future model-facing code
belongs to Stage 7 with owner approval.

## Stage 7.0 Learning Evidence Dataset

Stage 7.0 keeps the Stage 6.1 actor-safe projection as the only deployment
actor input view.

The learning evidence dataset must separate:

- `actor_safe_view`: local actor observations only;
- `critic_centralized_view`: training-only centralized context;
- `learning_target_view`: training-only edge-delta and objective-derived
  targets;
- `diagnostics_view`: audit-only metadata.

Oracle labels, registered objective metrics, selected topology, global topology,
future outcomes, and surrogate diagnostics must not enter `actor_safe_view`.

Actor policy interface work is deferred to Stage 8 after a Stage 7 data quality
report.

Stage 7.1 confirms the minimal evidence report has zero actor-safe leakage.
Stage 8 may define the policy interface with owner approval, but model
implementation and training remain blocked.

## Stage 8.0 Actor Policy Interface Contract

Stage 8.0 defines the deployment policy interface while preserving the same
Dec-POMDP boundary.

Policy inputs are exactly the Stage 6.1 actor-safe fields:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

After Stage 9.0 schema unification, active policy outputs are local directed
edge-score records only. The legacy local edge-decision schema remains for
older non-learning baselines. `activate` belongs only to that legacy schema or
to the assembler-selected topology after environment-side projection.

Stage 7 oracle labels, objective metrics, edge-delta learning targets,
surrogate diagnostics, critic centralized views, and future outcomes remain
outside actor deployment inputs. They may be used only in later training-only
interfaces after owner approval.

Stage 8.0 does not implement actor, critic, graph, recurrent, checkpoint, or
training code.

## Stage 9.0 Local MLP Edge Scorer

Stage 9.0 adds a minimal local MLP edge scorer scaffold. Its inputs remain
exactly the Stage 6.1 actor-safe fields and local incident-neighbor features.
It outputs edge scores through `actor_policy_local_edge_score_output_v1`, not
hard activations or selected topology.

Stage 9.0 does not train, write checkpoints, write training artifacts, run a
learner loop, implement PPO/MAPPO/COMA, or implement GNN/GRU/LSTM/Transformer
modules.
