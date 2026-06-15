# Stage 8.1 Policy Architecture Decision Record

## Controlled Object

The controlled object is the Stage 8 policy architecture boundary for
MARL-Topology: decentralized actor edge scoring, centralized training-only
critic interfaces, and the environment-side topology assembly layer.

## Desired State

Stage 8 freezes the architecture direction and deploys interface scaffolds
without implementing neural networks, optimizers, training execution,
checkpoints, reward-weight calibration, final tau selection, or v5 migration.

State marker:

```text
stage8_policy_architecture_and_assembler_closed
actor_outputs_edge_scores_only = true
environment_side_topology_assembler_required = true
model_implementation_allowed = false
training_execution_allowed = false
owner_decision_required_for_stage9 = true
```

## Baseline

Stage 8.0 froze the actor-safe input schema as the Stage 6.1 actor-safe batch
field set and froze actor output as local edge decisions. Stage 7 supplies
separated actor-safe, critic-centralized, learning-target, and diagnostic views.
Stage 4 consensus consumes communication-layer delivery/message matrices, not
policy logits or actor scores.

V5 inheritance check: lessons `L003_full_mask_not_oracle`,
`L006_fixed_threshold_deployment_risk`, `L007_credit_calibration_not_impossibility`,
`L008_phase_script_entropy`, `L009_dec_pomdp_boundary_before_actor`, and
`L010_training_after_oracle_baseline_contracts` are active. Stage 8 does not
migrate v5 code or adopt v5 phase-script structure.

## Actor Role

The actor is a decentralized local outgoing directed edge scorer.

Actor input is restricted to actor-safe local observations: local agent
identity/type, local position, local neighbor observations, local link
estimates, local messages permitted by the scenario, and local history.

Actor output is an edge score, logit, and optional probability for candidate
directed edges. Hard activation is not produced by the actor. Hard activation is produced by the environment-side topology assembler.

Forbidden actor inputs remain:

- global graph state or complete topology;
- oracle labels or edge-delta targets;
- consensus objective values;
- reward-surrogate components;
- future outcomes;
- critic-only centralized features.

## Action Semantics

Agent `i` scores a directed candidate edge `i -> j`.

The selected directed topology is assembled environment-side from actor scores
and declared candidate/resource constraints. The selected edge represents an
active outgoing communication attempt or scheduled communication candidate at
the communication layer.

Stage 4 PBFT reliability consumes communication-layer delivery/message
matrices produced after communication simulation. Stage 4 does not consume raw
actor scores, policy logits, or actor probabilities.

## Actor Architecture Ladder

The approved ladder is:

- `A1 Local MLP edge scorer` - first learnable baseline for later Stage 9.
- `A2 Local GNN edge scorer`.
- `A3 Local GRU edge scorer`.
- `A4 Local GNN + GRU edge scorer` - likely main candidate after baseline
  evidence.
- `A5 Local GNN + LSTM edge scorer` - must be tested once GRU benefit is
  confirmed.
- `A6 Local neighbor-attention / local transformer edge scorer` - ablation,
  not first implementation.
- `A7 Hybrid local GNN + temporal memory + attention` - ablation, not first
  implementation.

Any memory feature must be local history or permitted-message history. Global
recurrent actor state is forbidden for deployment.

## Critic Architecture Ladder

The centralized critic is training-only. The approved ladder is:

- `C1 global pooled MLP critic`.
- `C2 centralized GNN critic`.
- `C3 centralized GNN critic + edge-delta heads`.
- `C4 centralized Graph Transformer critic`.

Critic heads are:

- global value head;
- feasibility / constraint head;
- `consensus_success_probability` prediction head;
- latency head;
- energy head;
- edge-delta add head;
- edge-delta remove head;
- edge-delta keep head.

Critic outputs must never become actor deployment observations or deployment
assembler repair signals.

## Transformer Policy

Transformer models are allowed later for the centralized critic, especially as
a Graph Transformer critic ablation. They are also allowed later for local
actor attention over local neighbors or local ego-graph context.

A global actor transformer is forbidden. Transformer modules are not
implemented in Stage 8.

## COMA Decision

COMA is not mainline and has no Stage 8 implementation.

COMA is a counterfactual multi-agent credit-assignment method. It may be
considered only as a future optional ablation after the direct edge-delta
critic fidelity gate passes and PPO/MAPPO credit remains insufficient.

Direct edge-delta critic evidence is preferred for topology planning because
the topology decision is naturally edge-local: add, remove, and keep effects
can be evaluated against explicit candidate-edge changes, feasibility
constraints, latency, energy, and consensus prediction heads. That gives a
clearer sensor for ranking and calibration than adopting COMA as the default
credit mechanism.

## Training Sequence

The future owner-approved sequence is:

1. Supervised edge-scoring / edge-delta warm start.
2. Centralized critic pretraining.
3. MAPPO/PPO fine-tune.
4. COMA optional ablation only after the edge-delta critic fidelity gate and
   only if PPO/MAPPO credit remains insufficient.

Stage 8 performs none of those training steps.

## Not Implemented In Stage 8

Stage 8 explicitly does not implement:

- torch neural models;
- actor or critic weights;
- GNN modules;
- GRU modules;
- LSTM modules;
- Transformer modules;
- PPO or MAPPO;
- COMA;
- checkpoints;
- training loop;
- optimizer;
- training artifact writer.

## Sensors And Acceptance

Acceptance sensors:

- architecture decision document exists;
- topology assembler contract document exists;
- actor interface emits edge scores and not final topology;
- critic interface is training-only;
- deployment assembler is environment-side;
- fixed top-k is marked baseline only;
- conflict-aware greedy assembler is implemented and tested;
- source scan finds no torch, tensorflow, training loop, checkpoint creation,
  COMA implementation, PPO/MAPPO implementation, or v5 migration.

## Regression Check

Protected non-target behavior:

- Stage 6.1 actor-safe batch fields remain the actor deployment schema.
- Stage 7 critic and learning-target views remain excluded from actor input.
- Stage 4 PBFT reliability remains downstream of communication-layer message
  matrices, not actor scores.
- Full graph remains a baseline, not an oracle.

## Residual Risks

Stage 8 does not prove that local edge scores are learnable, calibrated, or
sufficient for Stage 4 reliability. Stage 9 needs owner approval and should
start with either `stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution`
or `stage_9_0_model_implementation_plan_for_mlp_edge_scorer`.
