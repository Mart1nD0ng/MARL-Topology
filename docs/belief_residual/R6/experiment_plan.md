# R6 — experiment plan: evidence-gated residual action (heads ACTIVE_IN_EVAL → ACTIVE_IN_DEPLOY)

## One variable (Contract v4 §15 / Workflow §"每轮只改一个变量")
**The evidence GATE.** R3 sampled an all-candidate Bernoulli residual on the actor's residual logits → it
collapsed to the anchor (edit_rate 0.0, no direction signal). R4 proved a beneficial-edit direction signal
EXISTS; R5 proved the LOCAL-feature heads can RANK it (random CI>0, urban mean-positive). R6 introduces **one**
new mechanism: a deployable gate that filters the LOCAL candidate edits by the **frozen R5 heads'**
repair/safety/edit predictions and applies only the gate-passing edits. Everything else (anchor base, candidate
enumeration, budget-safe mutual decode, 0-eval deployment) is unchanged from Q6/R3.

## Mechanism (deployable, 0 evaluator at decision time)
At each frame, for the deployed actor:
1. **Anchor** = `local_hysteresis_action` (directly computed from local obs + own previous topology; 0 eval) —
   identical base to Q6/R3; zero gated edits → the anchor EXACTLY.
2. **Local candidates** around the anchor — add = budget-feasible non-anchor incident edges (`deg<budget` on
   BOTH endpoints), remove = anchor edges. Depends ONLY on anchor membership + local degree/budget (no eval).
   This is the candidate enumeration of R5's `build_edit_examples` MINUS the evaluator label computation.
3. **Score** candidates with the frozen R5 heads: `actor.edit_scores(nf_s, ef_s, ei)` → `edit_logit`,
   `repair_pred`, `safety_pred` (LOCAL features `[ef, h_u⊙h_v, |h_u−h_v|]`; 0 eval).
4. **Gate** (the new mechanism):
   - add-candidate passes iff `repair_pred ≥ τ_repair` AND `sigmoid(edit_logit) ≥ τ_edit` (predicted to reduce
     D_quorum AND predicted beneficial);
   - remove-candidate passes iff `safety_pred ≤ τ_safety` AND `sigmoid(edit_logit) ≥ τ_edit` (predicted low
     deletion risk AND predicted beneficial).
5. **Apply** the gated flips via the existing budget-safe mutual decode (`residual_decode_from_flips`); zero
   gated → anchor exactly.

The heads (edit/repair/safety) move from ACTIVE_IN_EVAL (R5: used only by the held metric) to ACTIVE_IN_DEPLOY
(R6: in the deployed topology-decision path).

## Honest CSI / regime scope (single-variable correction to the R5 note)
To attribute the GATE cleanly, R6 holds the observation regime fixed at the R3/R5 setting: stale-CSI overlay
**OFF**, `ef` = current-frame channel (the deployable channel; "no true CSI" = no evaluator/critic/central
computation). The stale-CSI overlay-ON evaluation across all arms is **R8** (the full pilot, urban delay1 /
current / random delay1) — NOT R6, so R6 changes exactly one variable vs R5. (This refines the R5-memory note
"eval WITH stale overlay ON" — that belongs to R8.)

## Heads are TRAINED then FROZEN (held never trains)
The R5 heads are trained on TRAIN scenes (R4 evaluator scores = training-only labels), then FROZEN and deployed
on SEPARATE held scenes. The true evaluator scores only the PRODUCED topology for the honest C/E/L/J metric
(current real channel + closed-form quorum tail) — it is NEVER in the decision path.

## Failing-first load-bearing tests (Contract v4 §3, §"先写失败测试")
`tests/unit/test_belief_residual_R6_gate.py` (fail on HEAD: `evidence_gated_action` module absent):
1. `test_gate_calls_heads_in_decision_path` — the gated action CALLS `actor.edit_scores` and uses
   repair_pred/safety_pred/edit_logit to filter (spy); not a wrapper.
2. `test_zero_gated_candidates_returns_anchor` — impossible thresholds → topology == anchor EXACTLY
   (effect-on-decision; the anchor is preserved).
3. `test_gate_blocks_bad_edits` — a stub head predicting high deletion-risk + low repair gates OUT all edits
   (→ anchor); a stub predicting good edits applies them (edit_rate>0, topology≠anchor). The gate changes the
   final topology ONLY for predicted-good edits (effect-on-decision, both directions).
4. `test_gate_is_budget_safe` — no node exceeds its budget after gated edits (any head, any thresholds).
5. `test_gate_deploy_uses_no_evaluator` — the gated-action signature/path takes only the local obs + the
   frozen actor + thresholds; no evaluator/true-CSI/critic argument (deployability audit).

## Measurement (pilot → 5-seed A/B; Contract v4 §15)
Deployable A/B on HELD scenes, 0 evaluator at decision (evaluator scores only the produced topology):
- **arm A** anchor (edit_rate 0 by construction);
- **arm B** evidence-gated (TRAINED R5 heads);
- **arm C** evidence-gated (UNTRAINED heads) — random-gate control.
Per-arm metrics: feasibility (consensus ≥ 0.9 rate), mean true return J, edit_rate, unsafe_edit_rate (a gated
edit that drops a feasible frame to infeasible), feasible-frame retention, zero_edit_rate, decision-time
eval_calls (must be 0). 5 seeds × {random, urban}; CI on (B − A) feasibility and return, and (B − C).

## Exit condition (Contract v4 §15)
Mechanism correctness REQUIRED: bad edits gated out (unsafe_edit_rate low), zero-candidate → anchor (test),
budget-safe (test), 0-eval deploy (test). Plus the deployable read:
- (B − A) feasibility/return CI **> 0** on a regime → the evidence-gated action converts the R5 ranking into a
  DEPLOYED gain → KEEP → R7 (adaptive anchor-KL/safety instead of fixed thresholds).
- (B − A) CI **spans 0** (B == A, gate too conservative / no net gain) → the heads rank but the deployed gate
  doesn't yet beat the anchor → KEEP the mechanism, tune via R7's adaptive gate (NOT a STOP — the mechanism is
  correct and R5 learnability holds); state the path-specific scope.
- (B − A) **< 0** (gate lets bad edits through, B worse than A) → the gate/thresholds are miscalibrated → REVISE
  thresholds before R7; report unsafe_edit_rate as the failure locus.

## No-substitution / hard constraints
Deployment fully decentralized (local + mutual, 0 eval); the gate reads ONLY the frozen heads' local-feature
outputs; true C/E/L/J on the current real channel + closed-form quorum tail; only anchor-relative LOCAL gated
edits (never the full oracle topology); held never trains the heads; ≥5 seeds + CI (no smoke headline);
report whether the gate changes the final topology (edit_rate / unsafe_edit / retention), not just forward;
commit to trunk, no push.
