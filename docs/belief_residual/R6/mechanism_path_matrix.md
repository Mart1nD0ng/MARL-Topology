# R6 Mechanism-Path Matrix delta (Contract v4 §2)

| mechanism | before R6 | after R6 | path / evidence |
|---|---|---|---|
| edit_head / repair_head / safety_head | ACTIVE_IN_EVAL (R5, held metric) | **ACTIVE_IN_DEPLOY (R6)** — the frozen heads' edit_logit/repair_pred/safety_pred filter the candidate edits in the deployed topology-decision path | `evidence_gated_residual` calls `actor.edit_scores`; spy test |
| evidence gate (repair/safety/edit thresholds) | NOT_PRESENT | **ACTIVE_IN_DEPLOY (R6)** — add iff repair_pred≥τ_r ∧ σ(edit_logit)≥τ_e; remove iff safety_pred≤τ_s ∧ σ(edit_logit)≥τ_e | new `evidence_gated_action.py`; zero-gated→anchor; budget-safe; 0 eval |
| local candidate enumeration (deployable) | n/a (R5 used it WITH evaluator labels) | **ACTIVE_IN_DEPLOY (R6)** — add=budget-feasible non-anchor incident edges, remove=anchor edges; LOCAL only | `local_candidates(anchor, edge_ids, ends, budgets)` — no evaluator |
| residual decode (budget-safe, mutual, 0 eval) | ACTIVE (Q6/R3 `residual_decode_from_flips`) | unchanged — reused to apply the gated flips | zero gated flips → anchor exactly |
| all-candidate Bernoulli residual (R3) | ACTIVE_IN_LOSS (R3 PPO, == anchor edit_rate 0) | unchanged (R6 does not touch the R3 trainer) | R6 is a deployable DECODER over frozen R5 heads, not a new trainer |

**ACTIVATION SUMMARY.** R6 promotes the R5 heads from ACTIVE_IN_EVAL (used only by the held top-k metric) to
ACTIVE_IN_DEPLOY: the frozen heads' local-feature predictions now FILTER which anchor-relative edits the
deployed actor applies. The gate is the single new mechanism. The decision path remains fully decentralized
(local features + mutual decode, 0 evaluator); the true C/E/L/J are computed by the evaluator ONLY on the
produced topology (current real channel + closed-form quorum tail), never in the decision path. Stale-CSI
overlay is OFF in R6 (current-frame channel; single-variable vs R5); the overlay-ON full pilot is R8.

**EFFECT-ON-DECISION (measured).** The gate DOES change the final topology (load-bearing on the action, not a
forward-only no-op): at tau_edit 0.5 the trained gate edits ~2–4% of frames (edit_rate 0.001, zero_edit 0.97 →
mostly == anchor), the untrained gate edits far more and DEGRADES (C_feas 0.28/0.70 < A 0.375/0.815). Lowering
tau fires more edits (edit_rate up to 0.11 urban) and turns net-harmful.

**RESULT (5-seed A/B + threshold sweep) — KEEP MECHANISM / HONEST NEGATIVE on the deployed gain.** B − A
feasibility/return spans 0 at the best threshold (B == anchor) and is net-negative in mean at every firing
threshold (unsafe_edit rises to 0.09 urban; each per-tau B−A CI itself spans 0, so the load-bearing claim is
"no tau's B−A lower bound > 0", not per-tau significance). NO operating point beats the anchor — the best deployable edit
policy over the anchor is "no edit". The gate mechanism is correct/safe/deployable (budget-safe, zero→anchor,
0 unsafe at tau 0.5, 0 eval) and the trained heads have deployable SUPPRESSION value (B > random-gate C in
mean), but the R5 local ranking (~40% top-k precision) does NOT convert into a deployed feasibility/return gain
(a deployable CONVERSION gap). This is a DATA/precision limit, NOT a mechanism/path/trainer/leak bug. The sweep
empirically pre-empts R7's adaptive-anchor-KL premise: the optimal deviation from the anchor is ZERO.
