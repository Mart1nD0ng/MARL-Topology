# Q5 — local_hysteresis imitation actor (the deployable anchor)

## Hypothesis (the single thing this stage gates)
The dynamic actor can REPRODUCE the deployable `local_hysteresis` anchor via behavior cloning — i.e. the
actor, decoded by the local mutual-acceptance decoder, matches the anchor's topology (high F1), reaches
the anchor's held feasibility, and keeps low switches. If it cannot reproduce the anchor, RESIDUAL RL
(Q6+) must NOT start — fix the actor/features first (workflow Q5 gate). The anchor is the base the
residual policy edits from, so the actor must first BE the anchor.

## The anchor (deployable, NOT a central teacher)
`training/dynamic_baselines.py::local_hysteresis_action(ef, edge_ids, edges, budgets, prev,
keep_threshold=0.4, add_threshold=0.6)` — keep previous-frame edges above keep_threshold, fill remaining
budget with the best NEW edges above add_threshold; an edge is active iff BOTH endpoints propose it. Uses
ONLY local edge features + the node's own previous topology → 0 evaluator calls (Contract §10.1
deployable). Thresholds match the D13 deployable hysteresis baseline.

## Single change (one variable)
Expose `local_hysteresis_proposals` (the per-node accept subsets + mutual topology) and build a
decoder-aware HYSTERESIS teacher trajectory (the per-node BCSP proposals = the anchor's accept sets, 0
eval calls — vs the existing D6 myopic-greedy teacher which calls the evaluator). Measure imitation in a
new `scripts/diagnostics/anchor_imitation.py`. No reward/PBRS, no residual yet.

## What is measured (workflow Q5)
Warm-start a DynamicRecurrentActor (memoryless) on train scenes toward the hysteresis proposals (BC =
the decoder-aware BCSP-subset NLL, the SAME loss as D6 warm-start), then on HELD scenes:
- **BCSP-subset NLL**: teacher-subset NLL before vs after warm-start (does BC converge?).
- **decoded topology F1**: F1 between the actor's decoded topology and the anchor's, per frame.
- **held feasibility**: of the actor's decoded topology vs the anchor's own feasibility (the target).
- **switches/frame**: of the actor's rollout vs the anchor's.

## Controlled variables
- Data `--dyn-data random` + urban; current CSI (imitation is orthogonal to stale CSI — one variable);
  actor and anchor both roll with their OWN previous topology (deployed-style); same scenes/seeds.

## Failing-first tests (fail on HEAD)
- `test_hysteresis_proposals_decode_to_action` — `local_hysteresis_proposals(...)[1] ==
  local_hysteresis_action(...)` (refactor byte-identical) and accept sets respect the budget.
- `test_hysteresis_teacher_zero_eval_calls` — the hysteresis teacher trajectory calls NO evaluator
  (reward_of) — it is the deployable anchor, not the central myopic teacher.
- `test_f1_metric` — the topology-F1 helper on synthetic sets.
- `test_warmstart_reduces_teacher_nll` — BC on the hysteresis teacher reduces the teacher-subset NLL.

## Success criterion (Q5 passes iff)
1. Tests pass; the diagnostic runs real-shard and writes a report.
2. The actor REPRODUCES the anchor: decoded F1 high (target ≳ 0.6) AND held feasibility within a small
   gap of the anchor's own feasibility (the actor is not far below the anchor it imitates).
3. BC converges (final teacher NLL well below the initial).

## Failure criterion (RL paused per workflow Q5)
Actor cannot reproduce the anchor (low F1 or feasibility far below the anchor) → STOP, do NOT start
residual RL (Q6); diagnose actor capacity / features / decoder mismatch.

## Out of scope
Residual action space (Q6); D_quorum-guided repair (Q7); PBRS (Q9). The hysteresis teacher's trunk flag
(`--dyn-warmstart-teacher`) is added in Q6 where the residual base needs it.
