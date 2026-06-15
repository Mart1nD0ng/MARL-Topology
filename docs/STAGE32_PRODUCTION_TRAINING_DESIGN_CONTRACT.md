# Stage 32 Production Training Design Contract (Without Execution)

This contract freezes the design of the Stage 32 production-scale training run.
It is design-only. It does **not** authorize training execution, model code
beyond what already exists, checkpoint creation, scaled data generation runs, a
tau change, reward-weight tuning beyond the Stage 31 owner-approved
recalibration, or v5 migration. Execution requires the owner decision recorded
in `docs/STAGE32_DECISION_PACKET.md`.

## Controlled object

The transition gate:

```text
Stage 31 readiness verified (ready_for_production_training_scale_up)
  -> Stage 32 production-scale training execution
```

This is permission to design the production training run and its gates, not to
run it.

## Inherited frozen decisions (unchanged from Stage 31)

- `tau_requirement_min = 0.9` remains a hard feasibility gate.
- Reward surrogate is `feasibility_first_barrier_v2`
  (`src/marl_topology/objectives/surrogate_signal.py`); references are
  recalibrated on the feasible train split only.
- Action semantics use `physical_budget_aware_sequential_sampler` with
  kind-aware endpoint budgets (RSU=4, vehicle=2, pedestrian=1).
- Consensus stays the mean-field expected-initiator PBFT model (no message-level
  simulator).
- Dec-POMDP deployment locality is preserved; teacher/oracle/critic targets
  never enter actor inputs.

## Actor architecture (design)

- Active deployment actor: the existing message-passing GNN edge scorer
  `local_message_passing_gnn_edge_scorer_v2`
  (`src/marl_topology/models/local_gnn_edge_scorer.py`), consuming the v2
  constraint-aware actor-safe features
  (`actor_local_edge_tensor_v2_budget_aware`). It replaces the Stage 31 readiness
  MLP. Output: per-physical-edge logits; decentralized, local-observation only.
- Variable proposal size: a learned size/stop signal (design) bounded below by
  the quorum-connectivity requirement and above by the kind-aware endpoint
  budgets, replacing the Stage 31 fixed `N-1` proposal. The deployment policy
  must select the proposal from scores only (no objective access at deployment).

## Critic architecture (design)

- Centralized value critic: `centralized_message_passing_graph_value_critic_v1`
  (`src/marl_topology/models/centralized_message_passing_graph_critic.py`), the
  Stage 27-selected, Stage 28-validated critic (explained variance ~0.95). It is
  training-only and provides the policy-gradient baseline
  (`advantage = return - V`). It must not leak into actor inputs, actor export,
  or checkpoint actor serialization.

## Reward and normalization

- `feasibility_first_barrier_v2`, with latency/energy references recalibrated on
  the scaled feasible train split via `feasible_positive_max_v1`. tau stays 0.9.
  No weight tuning beyond the recalibration already approved in Stage 31.

## Data protocol (design)

- Scaled procedural generator
  (`src/marl_topology/data/stage31_scenario_generator.py`): target on the order
  of thousands of unique scenario contexts (recommended 2,000-5,000), node counts
  4-9, shared-spectrum interference regime, with a *measured* feasibility
  gradient (feasible fraction in roughly [0.3, 0.8], full graph rarely optimal).
- Leakage-checked train/eval/test split keyed on the unique scenario context
  (`stage31_production_dataset.py`), with every split holding feasible and
  infeasible scenarios.
- Scalable heuristic teacher labels for supervised warm-start; teacher labels are
  supervision targets only, never actor inputs.

## Training protocol (scaled from the Stage 25 base protocol)

- Phase 1 - supervised warm start of the GNN actor on teacher labels (BCE),
  ~30 epochs.
- Phase 2 - clipped policy-gradient / MAPPO fine-tune with the graph critic
  baseline and GAE: `gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.2`,
  `entropy_coef=0.01`, `max_grad_norm=0.5`, minibatched, `update_epochs` per
  batch, keep-best on the eval signal.
- Seeds: at least 3, recommended 5 for production; report per-seed and aggregate.
- Scale: more contexts and update steps than Stage 25; learning rate tuned within
  the design, not copied from v5.

## Diagnostics and metrics

- Registered evaluation metrics only: `consensus_success_probability`,
  `latency`, `energy`, `topology_diagnostics`.
- Training diagnostics (not promoted to metrics): `tau_feasible_rate`,
  `violation_rate`, `projection_rejection_rate`, surrogate objective-rank
  inversion, critic explained variance and value-return correlation, approximate
  KL, entropy, clip fraction, edge sparsity, seed variance.

## Readiness and stop gates (scaled from Stage 25)

- Reliability (hard): held-out `tau_feasible_rate` must not degrade beyond
  `0.05`; `violation_rate` increase `<= 0.05`.
- Critic health: explained variance `>= 0.5` (target ~0.9); positive
  value-return correlation.
- Projection: `projection_rejection_rate` ~ `0.0` (budget-aware sampler).
- Learning: held-out feasible rate improves over the warm-started policy and
  trends toward the teacher ceiling.
- Stop on: non-finite loss, approximate KL `> 0.03`, entropy below `50%` of
  initial, or reliability degradation beyond the bound.

## Artifact policy (OPEN — requires owner decision)

Production training will create checkpoints and training artifacts. `result_save`
is governed by frozen allowlists fixed at the Stage 25-28 scopes. Stage 32 must
choose one of:

- (recommended) extend the `result_save` allowlist with a single new scope
  `stage32_production_training`, write a Stage 5.9-compliant run manifest, and
  validate artifact-root containment with the Stage 5.10 validator; or
- use a dedicated artifact root outside `result_save` for checkpoints, keeping
  `result_save` frozen.

Manifests must record owner approval id, stage id, config id, scenario set id,
split id, seed, seed group id, code version marker, contract ids, metric registry
version, physics regime id, protocol model id, objective contract id, surrogate
config id, normalization reference id, architecture contract id, replay schema
version, and artifact policy id.

## Boundaries that remain in force

- tau fixed at 0.9; no final tau selection.
- mean-field PBFT kept; no message-level simulator.
- no v5 migration.
- Dec-POMDP locality; teacher/oracle/critic targets never in actor inputs.
- COMA, Transformer, and recurrent PPO remain out of scope unless a later stage
  identifies temporal memory as the primary limiter with evidence.

## Required before execution

- Owner approval (decision packet).
- The scaled generator and leakage-checked split run and quality-gated.
- GNN actor and graph critic wired into the training harness with Dec-POMDP
  leakage tests and a variable-proposal-size design test.
- The artifact-root decision resolved and a validated run manifest in place.
