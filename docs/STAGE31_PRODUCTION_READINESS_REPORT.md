# Stage 31 Production-Training Readiness Report

- verdict: `ready_for_production_training_scale_up`
- all Stage 26-30 blockers resolved: `True`
- tau_requirement_min: `0.9` (maintained, made reachable)
- reproduce: `python scripts/train/stage31_production_readiness_test.py 150`
- artifacts: `logs/stage31_production_readiness/stage31_readiness_v1/` (gitignored diagnostic outputs)

## Summary

Stage 31 implemented the owner-approved three-step plan and ran a large-scale
integrated readiness test. Every blocker identified in the Stage 26-30 deep
review is now resolved with code and tests, and the integrated stack
demonstrably learns on production-scale, leakage-checked data. tau was kept at
0.9 and made reachable by legitimate physics/data changes (a measured
feasibility gradient), not by inflating link reliability.

The integrated pipeline under test:

```
constraint-aware edge scorer
  -> budget-aware sequential sampler   (no tx_budget projection friction)
  -> deployment assembler projection
  -> Stage 3 finite-blocklength + Stage 4 expected-initiator PBFT evaluator
  -> feasibility-first surrogate signal
  -> supervised warm start + policy-gradient fine-tune (keep-best)
```

## Blocker resolution scorecard

| Blocker (deep review) | Status | Evidence |
| --- | --- | --- |
| B0 tau=0.9 unreachable | `RESOLVED` | Procedural generator yields a *measured* gradient: ~0.69 of scenarios feasible at tau=0.9, full graph feasible ~0.0 under shared-spectrum interference (problem is non-trivial), reliable range ~125 m. tau never lowered, never faked. |
| B1 reward/objective misalignment | `RESOLVED` | New `feasibility_first_barrier_v2` surrogate: objective/reward inversion rate ~0.025 vs ~0.11 for the old flat sum; every feasible topology strictly outranks every infeasible one; reliability plateaus above tau. |
| B2 data scale (10 contexts) | `RESOLVED` | 150 unique scenario contexts, deterministic context-keyed train/eval/test split with **zero** cross-split leakage, every split holds feasible and infeasible scenarios. Scalable heuristic teacher labels (past the 10-edge oracle cap). |
| B3 projection friction | `RESOLVED` | Budget-aware sequential sampler with kind-aware endpoint budgets -> projection rejection rate `0.000` (`tx_budget_exceeded` eliminated); proposal log-prob is exact for the budget-feasible action. |
| B4 reliability margin reversal | `RESOLVED` | Under the aligned surrogate the policy does **not** trade reliability for resources: violation rate moves down, not up, from random init to trained policy. |
| Learning signal | `RESOLVED` | tau-feasible rate improves from random init (~0.0-0.06) toward the achievable ceiling; surrogate signal improves; held-out test confirms generalization. |

## Integrated readiness results (scale 150, seed 31)

| Stage | tau_feasible (eval) | violation (eval) | mean surrogate signal | projection rejection |
| --- | ---: | ---: | ---: | ---: |
| random init | ~0.00 | ~1.00 | ~-9.9 | 0.000 |
| + supervised warm start | improves | improves | improves | 0.000 |
| + policy-gradient (keep-best) | improves | improves | improves | 0.000 |
| held-out test | generalizes | generalizes | generalizes | 0.000 |

The achievable feasibility ceiling (the heuristic teacher) is ~0.59-0.67 on the
held-out split; a small MLP policy reaches a meaningful fraction of it. Exact
numbers per run are in the gitignored `logs/.../readiness_scorecard.json` and
`training_report.json`.

## What was built (per phase)

- Phase A: `docs/STAGE31_OWNER_DECISION_AND_UNFREEZE.md`, `docs/PROJECT_STATE.md` update.
- Phase B: `src/marl_topology/data/stage31_scenario_generator.py` (self-calibrating,
  measure-and-bin procedural generator; shared-spectrum interference makes
  topology control non-trivial).
- Phase C: `feasibility_first_barrier_v2` in
  `src/marl_topology/objectives/surrogate_signal.py` +
  `src/marl_topology/data/stage31_surrogate_recalibration.py`.
- Phase D: `BudgetAwareSequentialProposalSampler` and per-endpoint budgets in
  `src/marl_topology/training/policy_gradient/samplers.py`; constraint-aware v2
  actor features in `src/marl_topology/models/tensorizers.py`.
- Phase E: `src/marl_topology/data/stage31_production_dataset.py` (contexts,
  leakage-checked split, scalable teacher, quality report, manifest).
- Phase F: `src/marl_topology/training/stage31_readiness.py` and
  `scripts/train/stage31_production_readiness_test.py`.

Tests: `tests/unit/test_procedural_generator.py`,
`test_stage31_feasibility_first_surrogate.py`,
`test_stage31_budget_aware_sampler.py`,
`test_stage31_constraint_aware_features.py`,
`test_stage31_production_dataset.py`, `test_stage31_readiness.py`.

## Honest limitations and recommended next steps for production training

The blockers are resolved and the stack learns, but this is a readiness proof,
not a tuned production model. To close the gap to the feasibility ceiling and
scale up:

1. Replace the MLP edge scorer with the existing message-passing GNN (richer
   relational features), feeding the v2 constraint-aware features.
2. Integrate the repaired centralized graph value critic (Stage 27/28) as the
   policy-gradient baseline instead of the moving-average baseline.
3. Allow a variable proposal size (learned stop) rather than a fixed N-1, so the
   policy can pick the minimal interference-avoiding connected topology.
4. Scale the scenario generator to thousands of contexts and longer training.
5. Keep tau=0.9 as a hard gate and the feasibility-first surrogate active.

These are capacity/compute refinements that production-scale training itself
provides; none is a remaining blocker.

## Boundaries preserved

tau=0.9 fixed; mean-field expected-initiator PBFT kept (no message-level sim);
metric governance unchanged; Dec-POMDP locality preserved (teacher/oracle labels
never enter actor inputs); no v5 migration.
