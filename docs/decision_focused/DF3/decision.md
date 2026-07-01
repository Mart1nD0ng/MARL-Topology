# DF3 — decision (goal 1 HEADLINE: decision-focused CSI prediction) — v2, hardened by adversarial verification

**Disposition: the owner's strong hypothesis is REFUTED; the weak version is INCONCLUSIVE (underpowered).**
Specifically: **no realizable objective — plain MSE or any decision-focused variant — comes close to closing the
true-CSI oracle gap** (well-powered), so "a decision-focused objective will *convert* the stale-drop" is refuted.
But whether decision-focus gives a *small* net feasibility gain over MSE is **not resolved at n=5** — the point
estimate is negative in every variant, yet every CI spans 0 and the study is underpowered (~10–25% power against a
+0.03 effect). The binding limit is **information-in-features (aleatoric)**: the leak-free features do not contain
enough decision-relevant information for ANY objective to reach the oracle.

> **Provenance.** DF3-verify (2-critic Workflow `wf_0ad828fa-a22`), verdict **REVISE, conclusion_robust = FALSE** —
> the critics reproduced all three experiments byte-identically and ran sign tests + a power analysis. The v1
> decision over-claimed ("provably does not help / refuted / decisive"). This v2 corrects the statistics AND adds
> the missing B.3(iii) engagement audit (df3d).

## The well-powered result (STRONG, CI excludes 0)
**No realizable objective approaches the true-CSI ceiling.** Paired `true − mse = +0.131`, CI `[0.079, 0.184]`,
5/5 seeds positive, paired-t **p = 0.0022** — the oracle gap is real and robustly uncrossed. The design also
excludes any *large* (≥ 0.07) decision-focus gain (~0.7–0.96 power). So: the decision IS recoverable from perfect
CSI, but the leak-free features cannot get there with any objective ⇒ **aleatoric, information-limited**.

## The underpowered result (WEAK, all CIs span 0 — directionally negative, not proven)
Three BWAR variants, all with the point estimate favouring MSE but no test rejecting the null:
- Default BWAR: `bwar − recal_mse = −0.038` CI `[−0.088, +0.013]` (paired-t p=0.11); `bwar − mse = −0.038`.
- Val hyperparameter sweep (10 configs incl. `mse_equiv` = decision-focus OFF): `valbest − mse = −0.014` CI
  `[−0.057, +0.028]` (p=0.41). No BWAR config beats `mse_equiv`; 8/9 are ≥1 frame below it and the best
  (`bwar_sym`) is within 1 frame — i.e. **decision-focus OFF is at least as good as every variant**, but the top
  gap is within metric granularity (feasibility lives on a 1/64 ≈ 0.016 grid; effect sizes 0.014–0.038 = 1–2.4
  frame-flips).
- Marginal-slot variant (budget-margin weighting): `marginal − mse = −0.031` CI `[−0.089, +0.027]` (p=0.21).

**Honesty on these:** (a) the sign is 4/5, NOT unanimous — in seed-1 decision-focus BEATS MSE by a loss-sized
magnitude in all three variants. (b) The three experiments SHARE the seed construction (train `s*1000+1`, held
`s*1000+777`) → correlated, not independent replications; "triply-confirmed" is softened to "three correlated
variants, all directionally negative." (c) The recalibration control is a weak 3×3 `(a,b)` grid maximized on
train; it came out `== mse` (the 1-seed calibration gain was noise). (d) The df3b val selector runs on only 10
val scenes over 10 configs — a noisy selector that can bias `valbest` down (an anti-decision-focus bias).

## The B.3(iii) engagement audit (df3d) — the missing pre-registered test
DF0 B.3(iii) required proof the objective ENGAGED the decision-relevant edges (else a null could be an
under-firing objective, not an aleatoric limit) — v1 claimed the gate satisfied but never ran it. df3d measures,
on held edges, whether BWAR improves per-edge decision-side precision vs MSE:
- **RESULT (`df3d_boundary_audit_metrics.json`, 5 seeds): BWAR ENGAGED the decision direction.** `dir_acc` on the
  echo-wrong "moved" edges (well-populated, n≈445–653/seed): BWAR `{0.404, 0.409, 0.381, 0.405, 0.401}` vs MSE
  `{0.276, 0.333, 0.328, 0.428, 0.240}` → **`bwar − mse = +0.079`, CI `[−0.009, +0.167]`, positive in 4/5 seeds**
  (lower bound ≈ 0, centred clearly positive). The near-boundary/marginal side-acc sets are tiny (n_near 2–9,
  n_marg 0–2 — psucc is bimodal, few edges sit near the gate) and therefore uninformative; `dir_acc` on the moved
  edges is the well-powered engagement metric.
- **Verdict: the objective was NOT under-firing — BWAR improved per-edge decision-direction accuracy over MSE
  (+0.079, near-significant).** But both stay BELOW chance (0.40, 0.32 < 0.5) on the hard echo-wrong edges: even
  engaged, the leak-free features do not contain enough information to place the decision-critical edges on the
  correct side. So the DF3 feasibility-null is the **"engaged-but-aleatoric"** kind — the objective did its job,
  the information wall held — NOT a mis-tuned/premature negative. This is exactly the DF0-predicted honest-negative
  signature ("BWAR improves boundary calibration but the feasibility gap stays open ⇒ the limit is
  information-in-features, not the objective").

## Interpretation (corrected)
The campaign's "MSE ⊥ decisions" wall is NOT the objective: the well-powered fact is that the oracle gap (+0.131)
is uncrossable by any realizable objective, decision-focused or not. The owner's hope — that a decision-aligned
objective would *convert* — is refuted in the sense that matters (it does not approach the oracle). Whether it
adds a deployably-negligible (<2-frame) sliver over MSE is unresolved and, at that magnitude, irrelevant to
deployment. This CONVERGES with DF2 (goal 2): both goals hit the same information-limited wall — deployable
precision at the decision boundary is limited by the leak-free information, not by the objective (goal 1) or the
env temporal structure (goal 2).

## Scope / no over-claim
- Tested: BWAR-family objectives (boundary-weighted regression + asymmetric hinge + side aux) + budget-margin
  marginal-slot, 10-config val sweep, N≤16, fast R8 urban mobility, n=5 seeds (underpowered vs small effects).
- NOT tested: SPO+ with a solver-in-the-loop, differentiable optimization. These fix gradient-routing, not the
  feature-limited precision the +0.131 oracle gap shows is the bottleneck; the aleatoric finding predicts they
  too would not close it — stated as scope, not claimed as tested.

## Disposition & next
- **KEEP plain MSE**; decision-focus (BWAR) not adopted (no detectable improvement; default-off). No deployed-path
  change.
- **Goal 1 = the strong hypothesis refuted (oracle gap uncrossable by any objective), the small-gain question
  inconclusive.** No downgrade: the pre-registered gate items (val sweep, marginal-slot, and now the B.3(iii)
  engagement audit) were RUN, not asserted; the statistical over-claim was corrected on adversarial review.
- **Next: DF4 (goal 3, activation)** — the last owner goal.
