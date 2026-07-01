# DF3 — Claim Card

**Claim (two-part, calibrated to the evidence):**
1. **STRONG / well-powered:** NO realizable objective — plain MSE or any decision-focused (BWAR-family) variant —
   comes close to closing the true-CSI oracle gap. Paired `true − mse = +0.131`, CI `[0.079, 0.184]`, 5/5 seeds,
   p = 0.0022. ⇒ the owner's hypothesis that a decision-focused objective would *convert* the stale-drop is
   **refuted**; the limit is **information-in-features (aleatoric)**.
2. **WEAK / underpowered:** whether decision-focus adds a *small* (<0.05) net feasibility sliver over MSE is
   **inconclusive at n=5** — the point estimate is negative in all three variants but every CI spans 0 (p =
   0.11–0.41; ~10–25% power vs a +0.03 effect). Directionally negative, not proven zero.
3. **Engagement (B.3(iii)):** the decision-focused objective was NOT under-firing — BWAR improved per-edge
   decision-DIRECTION accuracy over MSE on the echo-wrong edges (`dir_acc +0.079`, CI `[−0.009, +0.167]`, 4/5
   seeds). But both stay below chance (0.40, 0.32) there ⇒ the info isn't available. Engaged-but-aleatoric.

**Path covered:** goal 1 (decision-focused prediction). Honest negative on conversion; the deployable belief
objective stays plain MSE.

**Evidence:** `df3_bwar_metrics.json` (default BWAR + recalibration control), `df3b_hpsweep_metrics.json` (10-config
val sweep), `df3c_marginal_slot_metrics.json` (budget-margin variant), `df3d_boundary_audit_metrics.json`
(B.3(iii) engagement audit). 5 seeds each; fast R8 urban mobility; leak-free features; true CSI = training label
only; deployed 0-eval; anchor code path shared train==deploy (τ from prev-topo membership, gates 0.4/0.6).

**Adversarial verification:** Workflow `wf_0ad828fa-a22`, verdict REVISE / conclusion_robust = FALSE (v1
over-claimed). v2 corrections: (a) downgraded "provably refuted / decisive / triply-confirmed" → the well-powered
oracle-gap claim + the underpowered small-effect caveat; (b) presented the true-CSI gap as the paired CI (its
strongest, CI-excludes-0 result); (c) noted the 4/5 (not 5/5) sign + seed-1 wins + correlated experiments +
1/64-granular metric + weak recal adversary + noisy val selector; (d) RAN the missing B.3(iii) engagement audit.

**Falsify:** a realizable objective that reaches the true-CSI ceiling (none does; gap +0.131 excludes 0); or a
decision-focus config with a CI-positive feasibility gain over MSE at adequate power (none at n=5; a higher-N study
could tighten the small-effect question, but the effect is deployably negligible — <2 frames either way).

**Scope:** BWAR-family + marginal-slot, 10-config sweep, N≤16, n=5 (underpowered vs small effects). SPO+ /
differentiable-optimization untested (predicted to fail — they fix gradient-routing, not the feature-limited
precision the oracle gap exposes).
