# DF2 — Experiment Plan (goal 2 GATE: oracle re-gate under the decorrelation sweep)

**Question:** does the temporal structure DF1 created (tunable, marginal-invariant `ρ`) become *CONVERTIBLE*?
I.e., under higher `ρ`, does a REALIZABLE predictor beat the distance-only geometry null and close a non-trivial,
CI-positive fraction of the oracle feasibility gap at the anchor's N≤16 decision boundary? Base HEAD `e0c59a7`
(DF1). Contract v4; design = DF0 validation_design v2 Part A (A.2 arms, A.3 gate).

## Reuse (the T1 oracle harness is 90% of this)
`scripts/diagnostics/t1_oracle_recovery_gen.py` already: feeds the anchor an arbitrary psucc vector
(`_anchor_with_psucc`), scores the produced topology with the TRUE evaluator (`g6._cons(ev, topo) >= _TAU_FEAS`,
0 eval at decision), trains leak-free predictors of true psucc (`_train_predictor`, `_edge_features`,
`_true_psucc`), and reports per-seed + CI. `build_csi_scenes` (r8) builds delay-1 stale scenes and **leaves
`motion_features` OFF** (so the `csi_delta` current-psucc leak is already closed — verified in the T1 docstring).
DF2 writes `scripts/diagnostics/df2_oracle_regate_gen.py` importing these building blocks.

## Adaptation (single new variable: the channel `d_corr`)
1. **Sweep `d_corr`:** build scenes with `dataclasses.replace(operating_point_regime(20.0),
   shadow_decorrelation_distance_m=d_corr)` for `d_corr` in a swept range. The regime's `channel_config()` (DF1)
   threads the knob to the true channel, so stale obs + true label + evaluator are ALL consistently at that
   `d_corr` (faithful; NLOSv ON here, unlike the DF1 isolation probe).
2. **Mobility / ρ axis (KEY):** the operating-point scenes are FAST (`dt=2 s`, 15–30 m/s → 30–60 m/frame), so at
   `d_corr=10` the field decorrelates almost completely (`ρ≈exp(−45/10)≈0.01`) and even `d_corr=100` gives only
   `ρ≈0.64`. Therefore DF2 **measures the actual `ρ(psucc_t, psucc_{t-1})` per setting** (reusing the DF1 probe's
   estimator on the scene rollout) and reports **conversion vs measured `ρ`**, not vs `d_corr` nominally.
   To span `ρ` from ~0 to ~0.8 either (a) widen `d_corr` to {10,25,50,100,200,400} (scoping ≥50 m as extrapolated
   slow-fading stress), and/or (b) add a slower-mobility scene variant (urban 5–15 m/s, `dt=1 s` — physically
   valid urban, closer to the DF1 probe). Report both `d_corr` and measured `ρ`.
3. **Nested-subset arms (M2 fix — the whole point):** all fed to the SAME anchor, scored on the SAME true channel:
   - `A stale_echo` — inject stale psucc (the deployed floor).
   - `geometry_only` — best causal predictor of true psucc from **distance alone** (`ef[:,5]`); the "current
     geometry is observed" null.
   - `mse_realizable` — best MSE predictor from the **full leak-free set** (`_edge_features`: stale psucc/lat/en,
     current distance, rel-vel, dist-delta, csi_age); the realizable nowcaster proxy (the T1 physics-residual).
   - `D true_csi` — inject true psucc (the ceiling).
4. **Metric — RGF, pooled-mean ratio, scene-bootstrap CI (M3 fix):**
   `RGF(arm) = (mean_scenes Feas(arm) − mean_scenes Feas(stale)) / (mean_scenes Feas(true) − mean_scenes Feas(stale))`,
   CI by bootstrap over scenes; report only where the pooled denominator CI excludes ε. **Primary endpoint:**
   `RGF(mse_realizable) − RGF(geometry_only)` (incremental value of the temporal features over distance).
5. **Statistical leak battery (built here with the predictor arms):** G1 R²(residual `shadowing_db` | leak-free)
   < τ_leak; G2 incremental `R²(psucc | full) − R²(psucc | distance)` (reported); G3 boundary-restricted
   AUC/flip-rate among edges within ±0.1 of the 0.4/0.6 gate; G4 per-scene fixed-effect leak. These certify the
   leak-freeness the DF1 knob has by construction, now at the observation level.

## GATE (validation_design A.3) — ALL must hold to proceed to DF3
0. Marginal-invariance (DF1: PASS) + knob-proven `ρ` monotone (DF1: PASS, re-confirmed on these scenes).
1. Headroom exists and grows: pooled `Feas(true) − Feas(stale)` CI-positive and increasing in `ρ`.
2. Sanity `RGF(true_csi) ∈ [0.9, 1.1]`.
3. Realizable converts: at the standards-defensible knob, `RGF(mse_realizable) − RGF(geometry_only)` CI `> 0`
   (pre-registered lower bound `> 0.15`).
**Honest NEGATIVE (fully valid):** `ρ` proven + marginal invariant, but the incremental endpoint spans 0 at every
`ρ` ⇒ even with genuine tunable temporal autocorrelation, a realizable predictor does not convert the oracle gap
over what current distance already gives → the wall is deployable predictor precision, regime-invariant (the 6th
honest negative, refining T6). **Kill:** if `RGF(true_csi) ∉ [0.9,1.1]`, inconclusive not negative — check first.

## Context: the T1 result to beat
At the DEFAULT channel (`d_corr=10`, `ρ≈0.01` on these fast scenes) the T1 oracle already found the realizable
arm HONEST NEGATIVE (`realizable C−A` spans 0). DF2 tests whether RAISING `ρ` (the DF1 knob) changes that — the
central goal-2 question. If it does not even under strong `ρ`, that is the decisive, scoped negative.

## Methodological validation (smoke-discovered — validating BEFORE any headline run)
Two real subtleties surfaced in the DF2 smoke; both must be understood for the test to be valid (owner's demand):
1. **Headroom shrinks as `d_corr` rises (expected, not a bug).** Raising the decorrelation distance makes the
   shadow evolve LESS between frames → the stale observation becomes MORE accurate (smaller stale-drop to recover)
   AND more predictable (higher ρ). So `d_corr` trades off the *size* of the gap against its *recoverability*.
   The RGF metric (fraction of the gap that EXISTS at each `d_corr`) is the right normalization; we report RGF only
   where the pooled headroom `Feas(true) − Feas(stale)` denominator exceeds a guard (0.02), and we report the
   headroom itself per `d_corr` so the reader sees where a gap exists. The interesting regime is intermediate
   `d_corr`: enough headroom AND enough ρ.
2. **ρ of per-edge psucc is confounded by saturation at the operating point (tx=20 dBm).** Many links have psucc
   ≈ 1 (constant series → excluded from the lag-1 estimate), so the AGGREGATE ρ is noisy/low even where the shadow
   IS temporally correlated. DF1 already cleanly proved `ρ(psucc | d_corr)` rises (0.044→0.429) at boundary
   tx-power with persistent links — DF2 leans on DF1 for the ρ-proof and reports the psucc distribution
   (`psucc_mean`, `frac_saturated_hi`, `frac_boundary`, `rho_defined_frac`) as a diagnostic so the saturation is
   visible, not hidden. `d_corr` (DF1-proven to map to ρ) is the primary x-axis.

**Validation diagnostic RESULTS (`df2_diag.json`, 3 seeds; slow mobility) — three confounds found, all fixed:**
1. **psucc is BIMODAL** at the operating point (`frac_boundary ≈ 0.00`, `frac_saturated_hi ≈ 0.5`) — almost no
   edges in the soft [0.3,0.7] middle. So per-edge ρ of the *saturated psucc* is meaningless.
2. **The `geometry_only` (distance-only) null is DEGENERATE** (feas 0.26 ≪ stale 0.67) — distance alone barely
   predicts the bimodal psucc (shadowing/LOS dominate), so a distance-only predictor collapses to ~constant and
   the anchor makes near-random picks. ⇒ `RGF(realizable)−RGF(geometry)` is a meaningless inflated ratio.
   **Distance is uninformative here, so the correct baseline is the STALE ECHO**, not the geometry null.
3. **Slow mobility → headroom → 0 by `d_corr=200`** (stale≈current) → RGF undefined.

**Redesign (applied):**
- **FAST operating-point mobility** (`dt=2 s`, 15–30 m/s — the R8 regime with the big stale-drop / headroom).
- **Drop the in-scene ρ as a headline.** ρ measured on trajectory frames is confounded (4 short frames + churning
  candidate-edge set + bimodal state → even the logit lag-1 autocorr is ~0). **DF1 already cleanly proved
  `ρ(d_corr)` rises 0.044→0.429 on idealized persistent links.** DF2 uses `d_corr` as the x-axis and reports the
  in-scene ρ/psucc-distribution only as a *diagnostic* (honestly showing it is noisy), NOT as a claim.
- **PRIMARY metric = `RGF(realizable)` vs the stale echo** (does the MSE-realizable predictor close a CI-positive
  fraction of the true-CSI oracle gap, and does it rise with `d_corr`?). `geometry_only` is kept as a diagnostic
  (its degeneracy demonstrates distance-uninformativeness).
- **Honest note (a real finding in its own right):** DF1's clean `ρ(d_corr)` was on *persistent* links; the fast
  deployment trajectories may not exhibit that structure. If `RGF(realizable)` is flat/≈0 across `d_corr`, the
  conclusion is that **raising the shadow decorrelation distance does not convert into recoverable feasibility in
  the deployment mobility regime** — the temporal structure the knob creates in idealized links does not survive
  fast short-horizon mobility and/or the bimodal decision boundary. That refines the binding limit; it is NOT
  softened to force a positive.

## Exit criteria
- [x] `df2_oracle_regate_gen.py` built; smoke runs across the sweep; instrumented (psucc dist + headroom + ρ).
- [x] Validation diagnostic understood → fast headline regime chosen; 5-seed oracle run done.
- [x] Attribution probe `df2b_persistent_recovery_probe.py` (v2, nested arms incl. pure-temporal) run.
- [x] Adversarial verification (Workflow `wf_4242d978-612`, REVISE / conclusion_robust=TRUE); all caveats folded.
- [x] Gate evaluated → HONEST NEGATIVE (mechanistically explained); `decision.md` v2 + claim_card + MPM written.
- [ ] DF2 committed (no push); DF3 next.
- [ ] ρ measured per setting; RGF + primary endpoint with scene-bootstrap CI; leak battery reported.
- [ ] Gate evaluated; `decision.md` (proceed to DF3 / honest negative) written; committed (no push).
