# Q4 — D_quorum ↔ true C alignment test (the gate before reward)

## Hypothesis (the single thing this stage decides)
The cheap proxy `D_quorum` (Q3) is monotonically anti-aligned with the TRUE reliability `C` under LOCAL
topology edits, so that "reduce D_quorum" is a faithful surrogate for "increase C". If alignment is good,
`D_quorum` MAY enter the reward at Q9 (as a potential, Spec S8.3). If alignment is poor, it MUST NOT —
recorded as an honest gate result.

## The non-obvious fact this stage must handle
The dynamic operating-point regime uses `fault_model="fixed_set"` → the true `C` is the **robust
worst-case** reliability (`robust_consensus_reliability`, a fixed-fault-set search), NOT the
expected-initiator average that `D_quorum` mirrors. So the alignment test compares an expected-initiator
-style proxy against a robust true C. Whether they still track under local edits is the empirical
question — exactly what this gate answers (no assuming).

## Single change (one variable + its support)
1. A behavior-preserving refactor: extract `Stage21ObjectiveStackEvaluator._reliability_inputs(selected)`
   (the schedule/records/matrices/validators/f the reliability uses) from `evaluate()`; `evaluate()`
   calls it. Pinned byte-identical by the full suite + a bridge-C==evaluate-C test.
2. `src/marl_topology/training/quorum_deficit_bridge.py` (training-only, gate-exempt): unwrap the
   reference evaluator + compute `D_quorum` from the SAME matrices the evaluator uses.
3. `scripts/diagnostics/quorum_deficit_alignment.py`: the alignment sweep + metrics (eval-only).

## What is measured (Spec S8.3)
Sample real `--dyn-data {random,urban}` topologies (a base topology per frame + the anchor topology),
apply LOCAL edits (add one edge / remove one edge), and for each edit record:
- `ΔC = C(x') − C(x)` (true robust reliability), `Δenergy`, and `ΔD = D_quorum(x) − D_quorum(x')`
  (deficit IMPROVEMENT — positive ΔD means the edit reduced the shortfall).
Report:
- `Spearman(ΔD, ΔC)` — the core alignment (want strongly positive).
- top-k repair hit rate: of the k edits with the largest ΔD, what fraction are in the top-k by ΔC.
- `ΔD>0 but C worsens` rate (the proxy says "better" but C drops) — want low.
- `ΔD>0 but energy explodes` rate.

## Controlled variables
- C, energy, and D_quorum all from the SAME reference evaluator (Contract D5.5 same-口径).
- Edits sampled from incident candidate edges; base topologies from a fixed seed; random + urban.
- D_quorum is read-only here (NOT in any reward).

## Failing-first tests (fail on HEAD)
- `test_reliability_inputs_reproduce_true_C` — robust C on `_reliability_inputs` matrices == evaluate() C.
- `test_bridge_deficit_zero_for_full_graph_high_for_empty` — D_quorum ~0 for the full graph (high C),
  large for a near-empty topology (C~0).
- `test_alignment_metrics_core` — Spearman / top-k hit / worsen-rate on synthetic ΔD/ΔC arrays.
- `test_perfect_alignment_synthetic` — when ΔD = −ΔC exactly, Spearman = +1 and worsen-rate = 0.

## Success / failure criterion (the gate)
- PASS (D_quorum eligible for Q9 reward): Spearman(ΔD,ΔC) strongly positive (target ≳ 0.5) on real
  topologies for BOTH data sources, with a low ΔD>0-but-C-worsens rate (target ≲ 0.2).
- FAIL (D_quorum NOT eligible): weak/negative Spearman or high false-improvement rate → record honestly;
  Q9 PBRS is blocked or needs a robust-consistent deficit variant.
- Either way, the result is a measurement, NOT spun. A FAIL is a legitimate gate outcome.

## Out of scope
Wiring D_quorum into reward (Q9, only if this passes); the residual policy (Q5+); a robust-style deficit
variant (only if this stage shows the expected-initiator deficit fails to align).
