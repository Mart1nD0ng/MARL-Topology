# Stage 9 To Stage 15 Model Stack Review

## Controlled Object

The controlled object is the first learnable MARL topology-control stack from
forward-only local edge scoring through a controlled policy-gradient pilot.

## Desired State

Progress from Local MLP edge scoring to supervised losses, supervised MLP
training, centralized critic pretraining, local GNN comparison, temporal
GRU/LSTM ablation, and a small controlled policy-gradient pilot without COMA,
Transformer, final tau selection, reward-weight calibration, checkpoint writes,
or v5 migration.

## Models Implemented

- `LocalMLPEdgeScorer`: actor-safe local edge scorer producing edge logits.
- `CentralizedMLPCriticBaseline`: training-only critic with value,
  feasibility, consensus-success-probability, latency, energy, and
  add/remove/keep edge-delta heads.
- `LocalGNNEdgeScorer`: local ego-graph actor using only per-actor incident
  candidate-edge aggregation.
- `LocalGRUEdgeScorer`: actor-safe temporal edge scorer for fixture-level
  recurrent sanity testing.
- `LocalLSTMEdgeScorer`: LSTM replacement tested after GRU sanity.

## Ablation Winners

- Stage 11 MLP warm start passed tiny-batch overfit but showed limited
  validation quality on contradictory Stage 7 labels.
- Stage 13 GNN beat MLP on validation edge loss in the current comparison
  (`GNN - MLP = -0.0317`) and improved pairwise ranking diagnostic. It became
  the provisional best supervised actor.
- Stage 14 GRU and LSTM passed synthetic temporal fixture sanity; LSTM achieved
  lower fixture loss than GRU. They were not promoted for Stage 15 because real
  Stage 7 evidence has only `time_step = 0`.
- Stage 15 therefore used the Stage 13 local GNN actor.

## Critic Fidelity Status

Stage 12 critic pretraining passed the basic fidelity gate:

- non-collapsed predictions;
- add-edge ranking Spearman about `0.413`;
- remove-edge ranking Spearman about `0.505`;
- balanced sign accuracy add `1.0`;
- balanced sign accuracy remove about `0.917`;
- feasibility accuracy about `0.828`;
- consensus MAE about `0.067`.

This is enough for a controlled pilot, not enough for scale-up claims.

## Assembler Behavior

`ConflictAwareGreedyAssembler` remained the environment-side topology owner.
Actor models always produced proposal scores or sampled proposal indicators,
not final topology. Stage 15 explicitly logged that projected topology
log-probability was not treated as exact.

The supervised score-to-assembler path often selected dense/full topologies on
the small demo fixture. The Stage 15 sampled proposal reduced edge count from
6 to 4 and energy from about `0.357` to `0.262`, but consensus success
probability stayed about `0.529`, below `tau_requirement_min = 0.9`.

## Supervised Vs Fine-Tuned Pilot

Stage 15 completed without stop-condition failure:

- violation rate did not worsen: before `1.0`, after `1.0`;
- actor did not collapse to empty or full graph after the update;
- projection rejection rate stayed `0.0`;
- no checkpoint or artifact write occurred;
- no COMA or Transformer was introduced.

The pilot improved resource use in the sampled rollout but did not satisfy the
reliability requirement. This is diagnostic evidence, not convergence evidence.

## Remaining Risks

- The current Stage 7 evidence is small and has contradictory labels across
  topology variants with identical local observations.
- The demo fixture used by Stage 15 appears infeasible under the owner
  requirement baseline `tau_requirement_min = 0.9`, so policy learning cannot
  prove requirement satisfaction there.
- Temporal results are synthetic fixture-level only.
- The Stage 15 surrogate config is an explicit non-calibrated pilot config.
- Single-seed/single-fixture pilot evidence is not scale-up evidence.

## Stage 16 Recommendation

Recommended Stage 16: **E. data quality improvement**.

Reason: the model stack is now wired end to end, but the decisive blocker is
evidence quality and feasibility coverage under `tau_requirement_min = 0.9`.
Before scale-up, the project needs more actor-safe supervised/online evidence
with feasible positive examples, richer scenario diversity, and real temporal
sequences. Reward/critic repair may become useful after that sensor improves,
but scale-up training is premature.

## Verification Summary

- Stage 9: smoke, pytest, and harness validation passed.
- Stage 10: smoke, pytest, and harness validation passed.
- Stage 11: smoke, pytest, and harness validation passed.
- Stage 12: smoke, pytest, and harness validation passed.
- Stage 13: smoke, pytest, and harness validation passed.
- Stage 14: smoke, pytest, and harness validation passed.
- Stage 15: smoke, pytest, and harness validation passed.

Final observed full-suite gate: `python -m pytest -q` reported `667 passed`.
Harness gate: `python harness\scripts\validate_tasks.py` passed.

## Owner Decision Required

Do not auto-start Stage 16. Owner decision is required.
