# D7 — decision

**Result: KEEP.** A fair-baseline framework now separates DEPLOYABLE policies (local-only action, zero
evaluator calls) from CENTRAL references (evaluator-greedy candidate search), with a per-method
evaluator-call budget report. Adversarial verification PASS (no gaps).

## What changed (one variable: a fair-baseline framework, additive)
- New `training/dynamic_baselines.py`:
  - Deployable, local-only action functions (no evaluator/solver/global decoder): `local_threshold_action`
    (accept incident edges with link-success-prob ≥ threshold, top-budget) and `local_hysteresis_action`
    (keep previous edges still good, fill remaining budget with the best new edges) — both via mutual
    acceptance (the deployed decoder semantics), budget-respecting.
  - `evaluate_deployable_baseline` (`action_evaluator_calls = 0`, `group = "deployable_policy"`) and
    `evaluate_central_reference` (per-frame myopic-greedy; `action_evaluator_calls > 0`,
    `group = "central_reference"`), both scored with the SAME dynamic metric (the one metric eval/frame
    is identical across methods and is NOT an action call — the honest §10.1 distinction).
  - `baseline_budget_report` — groups the methods + flags the deployable group as evaluator-free at
    action time (Contract §10.1/§10.2).

## Tests (failing-first; fail on `70eeb7e` with ModuleNotFoundError, pass after)
- `test_deployable_baseline_does_not_call_evaluator_for_action` (action calls 0; pure local action fn;
  budget-respecting), `test_reference_uses_evaluator_and_is_marked_central` (action calls > 0; central),
  `test_baseline_budget_report_records_eval_calls` (grouped per-method budget).
- Unit suite **678/0**; contract **63/0**. Adversarial verify PASS (4 lenses): action evaluator-free,
  central counted/labeled, mutual-acceptance budget-respecting, honest no-overclaim.

## Real-shard smoke (DIAGNOSTIC — 4 operating-point scenes N∈{8,12}, NOT a headline)
| group | method | action eval calls | per-frame feas | return | switches/frame |
|---|---|---|---|---|---|
| deployable | local_threshold_0.5 | **0** | 0.9375 | −0.727 | 0.81 |
| deployable | local_hysteresis | **0** | 0.9375 | −0.640 | 0.56 |
| central ref | myopic_greedy | 96 | 0.6875 | −1.259 | 0.88 |

**Honest finding (diagnostic, smoke-scope):** on these scenes the DEPLOYABLE local baselines BEAT the
CENTRAL myopic-greedy reference. This directly motivates the contract's §10.1 separation: the prior
campaign's "myopic-greedy reference beats the learned actor" must NOT be read as "a simple DEPLOYABLE
baseline wins" — the myopic-greedy is a weak *central* reference (reconfig-blind, limited to 6 named
candidates), while the local threshold/hysteresis baselines (which ARE deployable) are stronger here.
This is a 4-scene smoke, not a result; the proper multi-seed comparison is D8/D13.

## Scope / next
- D7 provides the framework + 2 deployable baselines + 1 central reference. Additional central
  references (horizon-DP already in the Temporal-Value-Test; SA/LNS witness; exact small-N oracle) and
  the learned actors (deployable, via `dynamic_eval`) slot into the same groups for the D8/D13 tables.
- Next: **D8** — the recurrent-vs-memoryless retest across {current-CSI, velocity, recurrent,
  velocity+recurrent}, now on the fully-corrected pipeline (D2 objective, D3 val split, D4 energy, D5
  velocity, D6 warm-start protection), reported against these DEPLOYABLE baselines and the CENTRAL
  references separately. This is the first stage that produces a (still scoped, single-RSU until D1)
  comparative result rather than a mechanism.
