# DF2 — Claim Card

**Claim:** Raising the shadow decorrelation distance (`d_corr`, the DF1 knob) does **NOT** make the stale-CSI drop
convertible into recovered feasibility. At the faithful operating point a realizable MSE predictor produces a real
per-link MSE improvement but **zero feasibility movement** (`feas(realizable) == feas(stale)`), and on idealized
links a **pure-temporal** predictor (two stale psucc lags, no distance) gives ≤0.006 decision-side advantage over
the echo at every ρ. The only decision recovery comes from **current geometry**, and only at **low ρ**.

**Path covered:** the goal-2 GATE (does the env modification convert). Honest NEGATIVE, mechanistically explained.

**Load-bearing evidence (TIGHT — the negative rests here, not on wide intervals):**
- `oracle_regate_metrics.json` (5 seeds): `feas(realizable) − feas(stale)` = `{+0.000, −0.009, +0.000, +0.003,
  +0.000}` (point-identity) WHILE `MSE(realizable) < MSE(echo)` with non-overlapping CIs at every `d_corr`.
- `df2b_persistent_recovery.json` (v2, 200 links × 5 field seeds): `temporal2` side-advantage over echo ≤ 0.006 at
  all ρ (MSE-advantage Spearman −0.9 in ρ); `geometry` side-advantage +0.044 at the lowest ρ, shrinking (Spearman
  −0.9); `MSE(echo)` collapses `0.27→0.03` as ρ rises.
- Sanity: `RGF(true_csi)=1`, headroom CI-positive 5/5 (perfect CSI DOES convert → valid, not inconclusive).

**Adversarial verification:** Workflow `wf_4242d978-612`, verdict REVISE / **conclusion_robust = TRUE**. All
required caveats folded into `decision.md` v2: (1) anchored on the point-identity not the wide RGF CI; (2) ρ scoped
(deployment ρ≈0; DF2b ρ≤0.42; high-ρ feasibility UNTESTED; mechanism transported not measured); (3)
geometry-vs-temporal shown non-tautologically via the pure-temporal arm; (4) `flip_recovery` floor stated +
Spearman not endpoint-diff; (5) pre-registered `RGF(real)−RGF(geo)` labeled VOID (geometry degenerate).

**Seeds/CI:** oracle 5 seeds + CI; DF2b 200 links × 5 field seeds + Spearman over the ρ sweep.

**What would INVALIDATE / falsify:** a predictor that moves `feas` above the stale echo at any `d_corr` (none
does); a pure-temporal arm whose decision-side advantage rises materially with ρ (it does not, ≤0.006); or
`RGF(true)≠1` (would make it inconclusive — it is 1).

**Scope (no over-claim):** shadow-faded urban→highway regime, anchor N≤16, MSE-trained predictors, ρ up to ~0.42;
convertibility of genuinely high ρ (>0.42) on feasibility is UNTESTED. The negative is specifically about
MSE-trained recovery — a DECISION-focused objective is DF3.
