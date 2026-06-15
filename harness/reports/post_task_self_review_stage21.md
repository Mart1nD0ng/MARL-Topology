# Post-Task Self-Review - Stage 21

Completed task: Stage 21 objective-stack-aligned and assembler-aware evidence
rebuild, supervised MLP/GNN rerun, fair same-assembler evaluation, and
conditional policy-gradient gate.

Intended desired state: Stage 21 evidence uses Stage 3 finite-blocklength
communication and Stage 4 expected-initiator PBFT; targets are assembler-aware
soft/ranking signals; MLP/GNN are rerun on actor-safe Stage 21 data; baselines
are fair projected baselines; policy-gradient runs only if all hard gates pass.

Actual achieved state: evidence, target, supervised rerun, fair evaluation,
tests, harness task, docs, and failure review were implemented. Five of six
hard gates passed. The actor performance gate failed, so policy-gradient was
correctly skipped.

Evidence:

- Stage 21 evidence rows: `70`.
- Actor edge samples: `994`.
- Ranking pairs: `685`.
- Target high/mid/low counts: `167 / 513 / 314`.
- Best actor: `GNN`.
- GNN projected tau-feasible rate: `0.4714285714`.
- Full graph projected tau-feasible rate: `0.4571428571`.
- Projected greedy tau-feasible rate: `0.4857142857`.
- Stage 20 best actor tau-feasible rate: `0.5593220339`.

Tests:

- `python -m pytest -q`: `745 passed`.
- `python harness\scripts\validate_tasks.py`: passed for `78` tasks.

Gates passed:

- objective-stack alignment;
- actor target quality;
- fair same-assembler evaluation;
- projection mismatch improved;
- boundary safety.

Gate failed:

- actor performance sufficient for policy-gradient pilot.

Gates deferred:

- policy-gradient pilot;
- COMA;
- Transformer;
- reward-weight tuning;
- final tau selection;
- scale-up training.

New risks:

- Stage 21 final-stack data is still small fixture-level evidence.
- Above-threshold actor proposal rejection remains high.
- GNN only narrowly beats projected full graph and remains below projected
  greedy.

Regressions protected:

- Actor input remains local-only.
- Global edge-delta targets remain critic-only.
- Full graph remains a baseline, not oracle.
- Raw baselines remain diagnostic only.
- `v5` remains read-only.

Candidate next tasks:

- actor feature or supervised training repair based on Stage 21 failure report;
- assembler constraint diagnostics focused on high-score rejection reasons;
- critic-only edge-delta calibration review for future policy-gradient
  readiness.

Recommended next task:

`stage_22_actor_feature_or_supervised_training_repair_based_on_stage21_report`

Owner decision required: yes. Codex must not continue to Stage 22 or run
policy-gradient without owner approval.

Required Stage 21 questions:

1. Did evidence use the final Stage 3/4 objective stack? Yes.
2. Did assembler-aware targets fix Stage 20 mismatch? Partially; target quality
   and top-proposal rejection improved, but actor performance is still too low.
3. Did MLP/GNN improve under fair projected baselines? GNN beat projected full
   graph but not projected greedy and not the Stage 20 best actor threshold.
4. Was policy-gradient pilot run? No, because the actor performance gate failed.
5. If pilot ran, did it improve or harm? Not applicable.
6. What remains blocked? Policy-gradient, scale-up, COMA, Transformer,
   reward-weight tuning, final tau selection, checkpoint creation.
7. Is COMA still deferred? Yes.
8. Is Transformer still deferred? Yes.
9. Should Stage 22 scale, repair, or collect more evidence? Repair actor
   features/supervised training and high-score rejection calibration before any
   scale-up.
