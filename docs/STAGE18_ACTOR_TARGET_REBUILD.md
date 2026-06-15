# Stage 18 Actor Target Rebuild

## Controlled Object

The controlled object is the actor target view. Stage 18 stops treating the
actor as a hard add/remove/keep classifier. The actor target is now local edge
priority: soft utility plus pairwise ranking. Hard labels are diagnostics only
unless a contradiction-free high-confidence subset exists.

## Actor Soft Utility Target

Field group:

- `actor_edge_utility_target`
- `actor_edge_utility_confidence`
- `actor_target_source`
- `actor_training_role`
- `actor_target_ambiguity_level`
- `actor_target_allowed_for_hard_supervision`

Rules:

- `actor_edge_utility_target` is a continuous score in `[0, 1]`.
- `actor_target_source = distilled_global_counterfactual` when global edge
  delta information contributes to the soft target.
- `actor_training_role = soft_distillation_only`.
- High ambiguity samples receive lower confidence.
- The target is not a global oracle hard label and does not expose raw
  consensus, latency, energy, feasibility, objective, or oracle fields to actor
  input.

## Pairwise Ranking Target

Field group:

- `ranking_pair_id`
- `agent_id`
- `time_step`
- `preferred_edge_id`
- `less_preferred_edge_id`
- `preference_margin`
- `ranking_confidence`
- `ranking_source`

Ranking pairs are generated inside one agent, one time step, and one local
candidate set. They are better aligned with an edge scorer than hard topology
actions. High-conflict samples do not create high-confidence ranking pairs.

## Hard Label Limited Use

Hard add/remove/keep labels remain only as diagnostics or a
contradiction-free high-confidence subset.

Required fields:

- `hard_label_allowed_for_actor_training`
- `hard_label_conflict_free`
- `hard_label_source`
- `hard_label_confidence`

If a rebuilt signature has conflicting hard labels, then
`hard_label_allowed_for_actor_training = false`.

## Critic-Only Global Targets

The following remain available only in `critic_target_view`:

- `delta_consensus_success_probability`
- `delta_latency`
- `delta_energy`
- `delta_feasibility`
- `delta_surrogate_diagnostic`
- `oracle_edge_used`
- `oracle_topology_membership`

They are marked with `target_role = critic_only` or diagnostic-only semantics
and do not enter `actor_safe_view`.

## Implementation

Durable logic lives in:

`src/marl_topology/data/disambiguated_targets.py`

