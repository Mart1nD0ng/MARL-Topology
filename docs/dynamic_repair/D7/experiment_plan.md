# D7 — experiment_plan

**Stage:** D7 — fair deployable baselines + separately-grouped central references (Plan §9; gap #8).

**Hypothesis:** The dynamic comparison only has the myopic-greedy CENTRAL reference (it calls the
evaluator to search named candidates each frame) — there is NO fair DEPLOYABLE non-learned baseline, so
"the learned actor is beaten" risks being read as "beaten by a simple deployable baseline" when it is
actually a centralized oracle (Contract §10.1, forbidden §13.11). D7 adds deployable baselines whose
ACTION SELECTION uses only local observations (zero evaluator calls) and a budget report that groups
deployable policies vs central references, recording per-method action-evaluator-calls.

**Single change (one variable = a fair-baseline framework), additive / opt-in (new module + driver):**
- New `training/dynamic_baselines.py`:
  - Deployable action functions (PURE, local-only — no evaluator/solver/global decoder): each node
    proposes a budget-capped subset of its incident edges from LOCAL edge features, then mutual
    acceptance (the deployed decoder semantics).
    - `local_threshold_action` — accept incident edges with link-success-prob ≥ threshold (top-budget).
    - `local_hysteresis_action` — keep previous edges still above a keep threshold; fill the remaining
      budget with the best new edges above an add threshold (reduces switching).
  - `evaluate_deployable_baseline(action_fn, scenes, …)` — rolls the action_fn (action uses only the
    local `ef`), scores with the SAME discounted-return/feasibility/switches METRIC as `dynamic_eval`
    (the metric evaluator call is the same for every method); reports `action_evaluator_calls = 0`,
    `group = "deployable_policy"`.
  - `evaluate_central_reference(scenes, …)` — the per-frame myopic-greedy over the named candidate
    variants; the ACTION SELECTION calls the evaluator (counted); reports `action_evaluator_calls > 0`,
    `group = "central_reference"`.
  - `baseline_budget_report(results)` — groups deployable_policies / central_references with per-method
    action-evaluator-calls (Contract §10.2 budget fairness; §10.1 grouping).
- A small diagnostics driver (smoke) runs both on real operating-point scenes and emits the budget report.
- The learned static/dynamic actors are ALSO deployable (evaluated via `dynamic_eval`, which decodes
  with the torch-free `local_mutual_assemble`); D7 notes them in the deployable group. The horizon-DP /
  SA / exact-oracle are additional CENTRAL references (horizon-DP already exists in the Temporal-Value-Test).

**Controlled variables:** the dynamic objective/metric (D2), splits (D3), evaluator (D4), observation
(D5) — all unchanged. D7 only ADDS baselines + a report; no existing path changes.

**Required failing tests (fail on `70eeb7e` with ImportError, pass after):**
- `test_deployable_baseline_does_not_call_evaluator_for_action` — `evaluate_deployable_baseline` reports
  `action_evaluator_calls == 0` and `group == "deployable_policy"`, and the action fn is a pure function
  of local features (no context/evaluator argument).
- `test_reference_uses_evaluator_and_is_marked_central` — `evaluate_central_reference` reports
  `action_evaluator_calls > 0` and `group == "central_reference"`.
- `test_baseline_budget_report_records_eval_calls` — `baseline_budget_report` groups the methods and
  records per-method action-evaluator-calls (deployable 0, central > 0); flags the deployable group as
  evaluator-free at action time.

**Expected activation:** the budget report JSON groups {deployable_policy, central_reference} with
per-method `action_evaluator_calls`.

**Success criterion:** the 3 tests pass; full unit + contract suites green; a real-shard smoke runs both
baseline kinds on operating-point scenes and emits a budget report with deployable action-evaluator-calls
= 0 and the central reference > 0.

**Failure criterion:** none expected (additive). If a deployable baseline is trivially infeasible
(0 feasibility everywhere), report it honestly — it is still a fair deployable lower bound, not removed.
