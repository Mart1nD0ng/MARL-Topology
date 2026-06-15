# Stage 18 Data Quality Gate

## Purpose

This gate decides whether Stage 19 may rerun the supervised actor stack on
disambiguated evidence. It does not authorize PPO/MAPPO, COMA, Transformer,
scale-up training, reward-weight tuning, final tau selection, or checkpoint
creation.

## Stage 19 Allowed Conditions

Stage 19 may be recommended only if:

- `actor_safe_view` contains no forbidden global fields.
- Hard-label allowed subset contradiction rate is below the declared threshold
  of `0.05`, or hard labels are no longer the primary actor target.
- Soft utility target coverage is at least `0.95`.
- Ranking pair count is nonzero.
- Critic-only global targets and actor targets are separated.
- Actor observation signatures are no longer `100%` contradictory for the
  primary actor target.
- A real multi-step actor-safe sequence is present.
- The report declares how MLP, GNN, GRU, and LSTM should be rerun.

## Stage 19 Forbidden Conditions

Stage 19 remains blocked if:

- Hard contradictory labels are directly used for actor training.
- Oracle labels enter actor view.
- Actor features still lack local topology/resource context.
- Target source is not registered.
- Confidence or ambiguity is not registered.
- Contradiction analysis is missing or fails.
- Tests are missing.

## Current Stage 18 Gate Result

The Stage 18 gate passes for an owner-approved supervised rerun on rebuilt
evidence:

- `hard_label_allowed_subset_contradiction_rate = 0.0`
- `soft_utility_target_coverage = 1.0`
- `ranking_pair_count = 412`
- `actor_safe_view_has_no_forbidden_global_fields = true`
- `actor_target_view_has_no_global_delta_fields = true`
- `critic_only_global_targets_separated = true`
- `real_multistep_actor_safe_sequence_present = true`

Recommended next task:

`stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence`

Owner approval is still required.

