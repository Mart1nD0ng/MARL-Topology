# DF3 — Experiment Plan (goal 1 HEADLINE: decision-focused CSI prediction)

**The owner's first-listed goal.** Optimizing MSE is ineffective (predicting the *average* channel better does not
help the decision-critical edges). DF2 showed MSE-trained recovery does NOT convert. DF3 asks the sharper
question: **does a DECISION-FOCUSED objective beat MSE at the anchor's keep(0.4)/add(0.6) boundary?** Base HEAD
`40c6962` (DF2). Design = DF0 validation_design Part B, hardened by the DF0-verify recalibration control.

## Method — BWAR (Boundary-Weighted Asymmetric Regret-surrogate)
Per edge, predicted `p̂`, true `p` (label), active gate `τ = 0.4` if edge ∈ prev_topo (keep) else `0.6` (add):
```
w = 1 + κ·exp(−(p−τ)²/(2h²));  hinge = c_fd·max(0,τ−p̂)·1[p≥τ] + c_fk·max(0,p̂−τ)·1[p<τ]
L = α·w·(p̂−p)² + β·w·hinge + γ·BCE(σ((p̂−τ)/T), 1[p≥τ])   (κ=4,h=.05,T=.05,α=β=1,γ=.3,c_fd=1,c_fk=.5)
```
`τ` is built from the SAME anchor code path as deployment (`_KT=0.4,_AT=0.6`, verified at edit_head_training.py:23)
— train==deploy. Physics-residual head `p̂=clamp(stale+net,0,1)`; inference reads ONLY leak-free features; true
psucc is a training-only label.

## Arms (fed to the SAME anchor; true evaluator scores; 0 eval at decision)
- `echo` (stale floor), `mse` (plain-MSE predictor = the DF2 realizable), `bwar`, `true` (oracle ceiling).
- **`recal_mse` — the fake-positive CONTROL (DF0-verify MAJOR-1):** the MSE predictor composed with a 2-param
  global monotone transform `g(p)=σ(a·logit(p)+b)` fit on TRAIN to MAXIMIZE anchor feasibility (grid over a,b).
  Zero per-edge capacity. **If BWAR does not beat this, its "decision-focus" is a global calibration shift, not
  boundary precision.** BWAR's asymmetric hinge (`c_fd>c_fk`) is a direct incentive to shift predictions upward,
  so this control is essential.

## Metric & gate
- **PRIMARY: `Feas(bwar) − Feas(recal_mse)`** on HELD scenes, ≥5 seeds + CI. Secondary: `Feas(bwar)−Feas(mse)`,
  `Feas(bwar)−Feas(echo)`; the MSE landscape (decision-focus may WORSEN MSE while improving feasibility — the
  expected signature; if BWAR *improves* MSE, the "gain" is more likely calibration).
- **POSITIVE** = `Feas(bwar) − Feas(recal_mse)` CI `> 0` ⇒ decision-focus converts beyond a global shift.
- **HONEST NEGATIVE (per DF0 B.3, pre-registered):** requires (i) the marginal-slot BWAR variant (budget/mutual
  weighting) ALSO spans 0, and (ii) a val hyperparameter sweep `(β, c_fd, κ)` whose val-best config still spans 0
  on held — else the negative is premature. Only then: the limit is information-in-features (aleatoric), and the
  gain over plain MSE is calibration, not decision-focus.

## Preliminary (1-seed smoke, NOT a headline)
`echo 0.583 | mse 0.625 | recal_mse 0.708 | bwar 0.667 | true 0.750`. BWAR beats plain MSE (+0.042) — but the
**recalibration control beats BOTH** (`bwar−recal = −0.042`), exactly the DF0-verify-predicted fake-positive: the
gain over plain MSE looks like decision-focus but a 2-param global recalibration captures more. BWAR also *lowered*
MSE (0.100 vs 0.116) — a calibration signature, not the decision-focus signature. The 5-seed run confirms with CI.

## Exit criteria
- [x] `df3_decision_focused_gen.py` (BWAR + recal control), `df3b` (val sweep), `df3c` (marginal-slot),
      `df3d` (B.3(iii) engagement audit) — all built + 5-seed.
- [x] Pre-registered DF0 B.3 gate RUN in full: (i) marginal-slot spans 0; (ii) val-best spans 0; (iii) engagement
      audit — BWAR engaged (dir_acc +0.079) but info-limited.
- [x] Adversarial verification (Workflow `wf_0ad828fa-a22`, REVISE / conclusion_robust=FALSE) → v2 corrects the
      statistical over-claim + adds the missing B.3(iii) audit.
- [x] `decision.md` v2 + claim_card + MPM written (calibrated: strong claim well-powered, small-effect
      inconclusive, engaged-but-aleatoric).
- [ ] DF3 committed (no push); DF4 next.
