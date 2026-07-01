# DF0 — Validation Design (THE CRUX) — v2, hardened by adversarial verification

> Owner's demand: 仔细思考前两个目标在开发过程中应该如何验证，确保验证方案的正确性才能得到有效的答案。
> An invalid validation gives an invalid answer.
>
> **Provenance.** v1 was produced by three independent design lenses. It was then attacked by two independent
> adversarial critics (each verifying claims against code). **Both returned REVISE** and found real defects that
> would have produced invalid answers. **This v2 folds in every MAJOR + MINOR fix.** The headline changes:
> - **Goal 2 / M4 (most dangerous):** the `d_corr` sweep must move ONLY the autocorrelation `ρ`, with the psucc
>   **marginal distribution held fixed** — else a "recovery converts" result is a marginal-shift artifact. New
>   marginal-invariance gate + field-seed averaging.
> - **Goal 2 / M1+M2 (unification):** every predictor arm (`geometry_only`, `mse_realizable`, `recovered`) is the
>   **best causal predictor from a nested feature subset**. The leak test becomes **incremental** R² over
>   distance; the geometry null becomes the distance-only regression; they are the same coherent object.
> - **Goal 2 / M3:** RGF is a ratio of **pooled means bootstrapped over scenes**, never a per-frame fraction.
> - **Goal 1 / MAJOR-1:** add a **monotone-recalibration control arm**; the headline is `BWAR − recalibrated-MSE`,
>   because BWAR's asymmetric hinge can win by a global bias shift, not boundary precision.
> - **Goal 1 / MAJOR-2:** the marginal-slot/mutual audit is a **pre-registered gate on the negative**, not a
>   post-hoc upgrade; boundary-flips are defined on the **mutual-decoded** topology.
>
> Every "guard" below is a named failing-first load-bearing test, written before the corresponding experiment.

---

## Part A — Goal 2: env temporal-autocorrelation modification

### A.0 What the modification actually is (verified)
- Shadowing is **already ON** (`operating_point_regime`: `v2x_37885`+`shadowing_37885`+`nlosv_37885`;
  `dynamic_rl.py:562`). Do NOT report "enable shadowing" as new.
- `_shadow_field_value` decorrelates over `SHADOW_DECORRELATION_DISTANCE_M = 10 m` (hardcoded module constant,
  `channel/model.py:38`). At `dt=1 s`, urban ~10 m/s, displacement/frame ≈ 10 m ≳ cell ⇒ `ρ_frame ≈ 0.37` ⇒
  near-i.i.d. ⇒ T6's weak recoverability.
- **The modification** = expose the decorrelation distance as a config-threaded, swept knob (governing ratio
  `v·dt/d_corr`). Diff: add `shadow_decorrelation_distance_m: float = 10.0` to `ChannelModelConfig` (validate
  `>0`); thread into `_shadow_field_value`; add to `PhysicsRegime`+`channel_config()`; CLI `--shadow-decorr-m`.
  Default `10.0` ⇒ off-sweep byte-identical.

### A.1 Knob range + standards basis (+ the marginal-invariance requirement)
| d_corr (m) | Regime | Basis | Target ρ_frame ≈ exp(−v·dt/d_corr) |
|-----------|--------|-------|-----------------------------------|
| 10  | urban V2V (HEAD default) | TR 36.885 A.1.4-1 | 0.37 |
| 25  | highway | TR 36.885 A.1.4-1 | 0.67 |
| 50  | suburban / UMa | TR 38.901 Table 7.5-6 | 0.82 |
| 100 | **extrapolation stress** (no urban-V2V basis) | — never headline | 0.90 |

Pre-register the primary endpoint at 25 m (highway); 50/100 m exploratory. Urban-preserving equivalent: hold
`d_corr=10 m`, refine `dt` (real V2X cadence is 100 ms). **CRITICAL (M4):** raising `d_corr` must change the
*autocorrelation* without changing the *marginal* psucc distribution; A.2(d) makes marginal-invariance a hard
precondition (the field's per-query renormalization does not guarantee it, so it is tested, not assumed).

### A.2 The unified predictor framing + the guards

**Nested-subset predictor arms (M1+M2 unification — the backbone of the whole design).** Every "arm" that feeds
the anchor is the **best causal predictor of `psucc_t` from a specified leak-free feature subset**, fit on train,
evaluated on held. Nested subsets:
- `geometry_only` := best predictor from `{d_t}` (current distance alone) — the null; distance legitimately
  predicts psucc, so this arm captures exactly the "current geometry is observed" baseline.
- `mse_realizable` := best MSE predictor from the full leak-free set `{d_t, stale_psucc, velocity, csi_age}`.
- `recovered` := the trained nowcaster (same leak-free set; later BWAR-trained).
- `stale_echo` := inject `psucc_src` (raw delay-1). `true_csi` := inject `psucc_t` (ceiling).
This makes the leak test, the geometry null, and the primary endpoint **one object**: the incremental value of the
temporal features `{stale_psucc, velocity, age}` over distance alone.

**(a) LEAK-FREE (M1 — rewritten; the #1 risk).** The old "R²(psucc | leak-free) < 0.05" is WRONG: current distance
legitimately drives psucc, so that R² is structurally 0.3–0.8. Three separately-preregistered tests instead:
  - **G1 residual-shadow leak:** fit the leak-free set → the *residual shadow* `shadowing_db` (`model.py:290`,
    i.e. psucc after removing the deterministic distance+LOS path-loss). **FAIL if R²(shadow_resid | leak-free)
    exceeds a preregistered τ_leak** (default 0.05 on the *residual*, not raw psucc).
  - **G2 incremental leak:** `Δ = R²(psucc_t | full leak-free set) − R²(psucc_t | d_t alone)`. This is the value the
    temporal features add *over distance*. It is expected to be > 0 (that is the recoverable signal); the leak
    check is that it is **not implausibly high** (a current-CSI leak would make it ~1) — but Δ is primarily the
    *recoverability signal*, reported, not gated. The gate is G1 + G3 + G4.
  - **G3 boundary-restricted leak:** among edges with true psucc within ±0.1 of the active gate (0.4/0.6), can the
    leak-free set predict which SIDE of the gate the edge lands on above chance? Report **AUC / flip-rate**; **FAIL
    if AUC exceeds a preregistered bound** (a leak that "only bites near the boundary" is invisible to pooled R²).
  - **G4 per-scene fixed-effect leak:** shadow is ~constant within a scene (same `_scene_shadowing_seed`,
    `model.py:364`); test for a per-scene leak a pooled regression averages away (add a scene fixed effect; FAIL if
    it explains the residual shadow).
  - **Interference caveat (m4):** SINR includes interference from other transmitters whose rx-power depends on
    *their current* shadow (`model.py:259-275`) — current psucc leaks a function of other pairs' current shadow.
    This is covered by G1/G3/G4 provided the leak-free feature set contains NO other-pair current-channel term
    (assert it) — the incremental-over-distance framing bounds it.

**(b) NO CURRENT CSI IN THE ACTOR OBS (verified leak vectors).** (i) col 5 = current `distance_3d_m`, never
staleified (leak-free by design — it is current *geometry*, not current *channel*; G1–G4 certify it does not
reconstruct current shadow). (ii) `motion_features` `csi_delta = psucc_t − psucc_{t-1}` uses **current** psucc
(`dynamic_frames.py:259`). **Preregister `motion_features=False` for all leak-critical arms; DROP the "or patch to
stale" escape** (m1: even a stale-difference encodes true-trajectory curvature). Tests: `test_no_current_psucc_in_obs`
(every `obs["ef"]` col equals a stale or non-CSI value); `test_seed_not_in_features`.

**(c) FAIR BASELINE (paired, same realization).** Within a knob and field-seed, all arms share the same scenes,
seed, and per-frame TRUE channel realization; only the injected psucc differs; evaluate per `(scene,frame)` on the
identical true channel. `test_arms_share_realization`: assert the evaluator's `ChannelRecord.shadowing_db` is
bytewise identical across arms at every frame. `test_evaluator_ignores_ef` (m2 fix): perturb `obs["ef"]` psucc by
**+0.05 in-range** AND flip a near-boundary edge across τ; FAIL if the reward changes (evaluator must read
`obs["context"].link_records`, not `ef`; +10 was out-of-range → vacuous).

**(d) CONTROLLED KNOB — and MARGINAL-INVARIANCE (M4, the single most important control).**
  - `ρ(d_corr) = Corr(psucc_t, psucc_{t−1})` over edge-frames. `test_autocorr_monotone_in_decorr`: **averaged over
    ≥5 FIELD-seed realizations per knob** (the seeds vary the shadow field, not just the RL init), FAIL unless the
    seed-averaged `ρ` is monotone non-decreasing and `ρ(100) − ρ(10) > 0.1`. Tolerance for the `exp(−v·dt/d_corr)`
    sanity is derived from the **2-endpoint** field (both endpoints sampled, per-node velocities), not the 1-D
    Gudmundson scalar (m3).
  - **`test_marginal_invariant_across_decorr` (THE gate precondition):** the marginal distribution of `psucc_t`
    (and of `shadowing_db`) must be **invariant across `d_corr`** (two-sample KS / quantile match within a
    preregistered tolerance). Bilinear-interp variance is only renormalized *in expectation over position*
    (`model.py:442-443`), so different `d_corr` can shift the sampled marginal → the knob would move *difficulty*,
    not just autocorrelation, and RGF would track the marginal shift. **If the marginal is not invariant, the field
    MUST be renormalized to a fixed marginal before the sweep** (calibrate a per-`d_corr` affine on `shadowing_db`
    to match the `d_corr=10` marginal). No sweep result is valid until this test passes.
  - **`lru_cache` hygiene (m6):** `_shadow_field_corner` is a module-level `@lru_cache` keyed by `(seed,ix,iy)`.
    Fold `d_corr` into the field seed key AND `cache_clear()` between sweep points in the harness; assert
    `_shadow_field_value` differs across `d_corr` at fixed (x,y) (`test_field_changes_with_decorr`). Note folding
    `d_corr` into the seed re-randomizes the field per knob — which is WHY (d) requires field-seed averaging (so
    cross-knob monotonicity/headroom CIs contain realization noise) and the marginal-invariance test.
  - **NLOSv confound (guard):** `nlosv_37885=True` adds its own autocorrelation independent of `d_corr`; hold it
    fixed across the sweep (attribute ρ-change to `d_corr`) or run a `nlosv=False` sensitivity arm.

**(e) RIGHT METRIC — RGF as a pooled-mean ratio, bootstrapped over scenes (M3).**
```
RGF(arm) = ( mean_scenes Feas(arm) − mean_scenes Feas(stale_echo) )
           / ( mean_scenes Feas(true_csi) − mean_scenes Feas(stale_echo) )
```
`Feas` = deployed anchor feasibility (decode via `_mutual`, score on the true channel). **Pooled means, NOT
per-frame ratios** (per-frame `Feas(true)−Feas(stale) ∈ {−1,0,+1}` is 0 on most frames → 0/0). CI by **bootstrap
over scenes** (resample scenes → recompute both pooled gaps → take the ratio per resample → CI on the ratio).
Report RGF only where the pooled denominator CI excludes ε (headroom exists); this is a *precondition for defining
RGF*, not a post-hoc frame filter. **Primary endpoint (M2): `RGF(recovered) − RGF(geometry_only)`** — the
incremental value of temporal features over distance; CI over the same scene-bootstrap. `RGF(true_csi) ≈ 1` sanity
band = **[0.9, 1.1]** preregistered (m5).

**(f) NO GOALPOST-MOVING + (g) BYTE-IDENTICAL OFF.** Report a matrix {deterministic-baseline row, shadowed@sweep}
× {stale, recovered, mse_realizable, geometry_only, true}; name the regime; scope `d_corr ≥ 50 m` as extrapolated.
`test_byte_identical_at_default_decorr`: at `shadow_decorrelation_distance_m=10.0`, every `ChannelRecord`, `obs`,
result JSON bytewise identical to HEAD (extend `tests/unit/test_37885_shadowing_nlosv.py`).

### A.3 Oracle re-gate (DF2) — gate + kill
4-arm oracle {stale_echo, geometry_only, mse_realizable, true_csi} across `d_corr ∈ {10,25,50,100}`, ≥5 field
seeds, RGF (pooled-ratio, scene-bootstrap CI).
**GATE (proceed to DF3, ALL must hold):**
0. **Marginal-invariance passes** (A.2d) — else the sweep is confounded; STOP and renormalize.
1. Knob proven: seed-averaged `ρ(d_corr)` monotone, `ρ(100)−ρ(10) > 0.1`.
2. Headroom exists and grows: pooled `Feas(true) − Feas(stale)` CI-positive and increasing in `d_corr`.
3. Sanity: `RGF(true_csi) ∈ [0.9,1.1]`.
4. Realizable converts non-trivially: at 25 m, `RGF(mse_realizable) − RGF(geometry_only)` CI `> 0` (preregistered
   lower bound `> 0.15`).
**Honest NEGATIVE (valid):** knob proven + marginal invariant, but `RGF(mse_realizable) − RGF(geometry_only)` CI
spans 0 at every knob ⇒ **even with genuine, tunable, standards-based temporal autocorrelation (marginal held
fixed), a realizable predictor does not convert the oracle gap over what current distance already gives** —
removes the "channel was structureless" escape hatch and localizes the wall to deployable predictor precision,
regime-invariant. The 6th honest-negative, refining T6. **Kill-criterion:** if `RGF(true_csi) ∉ [0.9,1.1]`, the
anchor can't use perfect CSI either ⇒ **inconclusive, not negative** (wrong lever) — check first.

---

## Part B — Goal 1: decision-focused CSI prediction

### B.1 The objective (DF3) — Boundary-Weighted Asymmetric Regret-surrogate (BWAR)
SPO+ collapses to a boundary hinge for a threshold decision; differentiable-opt fixes gradient-routing, not
feature-limited precision — over-engineered. BWAR = boundary-weighted regression + asymmetric wrong-side hinge +
optional side aux; plain PyTorch, warm-start from MSE. Per edge, predicted `p̂_e`, label `p_e`, active gate
`τ_e = 0.4` if kept else `0.6`:
```
w_e   = 1 + κ·exp(−(p_e − τ_e)² / (2h²))
hinge = c_fd·max(0, τ_e − p̂_e)·1[p_e ≥ τ_e]  +  c_fk·max(0, p̂_e − τ_e)·1[p_e < τ_e]
L_e   = α·w_e·(p̂_e − p_e)² + β·w_e·hinge + γ·BCE(σ((p̂_e−τ_e)/T), 1[p_e ≥ τ_e])
```
Start `α=1,β=1,γ=0.3,κ=4,h=0.05,T=0.05,c_fd=1.0,c_fk=0.5`. **CONFOUND WARNING (MAJOR-1):** `c_fd > c_fk` is a
direct incentive to shift predictions **upward globally**; against a conservative over-dropping anchor a pure
up-shift can raise feasibility with NO boundary-precision gain. This is guarded by the recalibration control arm
(B.2) — without it a bias-shift win would be mis-read as decision-focus.

### B.2 Guards (fake-positive paths)
- **Right metric:** score anchor feasibility, NOT MSE. (MSE may worsen — but see the recalibration control; a
  worse-MSE-better-feasibility result is NOT sufficient evidence of decision-focus on its own.)
- **RECALIBRATION CONTROL ARM (MAJOR-1, blocking):** add `recalibrated_MSE` = the frozen MSE predictor composed
  with a **2-parameter global monotone transform** `g(p̂)=σ(a·logit(p̂)+b)` fit on the TRAIN split to maximize
  anchor feasibility (zero per-edge capacity). The headline becomes **`Feas(BWAR) − Feas(recalibrated_MSE)`**. If
  BWAR does not beat a 2-parameter global shift, its "decision-focus" is bias exploitation, not boundary precision.
- **Same decision train==deploy:** build `τ_e`/`m_e` by calling the SAME `local_hysteresis_proposals` path
  (0.4/0.6, real `prev_topo`, budget, `_mutual`); assert byte-identical thresholds.
- **Effect-on-decision, MUTUAL-decoded (MAJOR-2):** define a "flip" on the **mutual-decoded topology** (an edge
  whose final active/inactive state differs between arms), NOT per-node accept sets (a node-side flip that dies in
  the AND is not a decision flip). Boundary-flip audit + **marginal-slot audit** (fraction of BWAR's boundary
  gain that lands on edges at budget ranks `b−1,b,b+1` AND admitted by both endpoints).
- **Leak-free + label-in-weight conditionality (MINOR-1):** `p_e` enters only loss/weights; `p̂_e=f(stale,geom)`;
  deployment-parity test. Using the true `p_e` in `w_e` is safe *only if the held split is truly disjoint by
  scenario* — state the dependency (it is importance-weighting, not a leak, under a genuine split).
- **Split by SCENARIO, not sequence (MINOR-2):** `build_split_manifest` splits on `sequence_id` (`dynamic_rl.py:354`)
  — disjoint *sequences*, not disjoint *scenarios/layouts*. Add `test_scenario_key_disjoint` asserting the
  scenario key (layout + mobility-seed family) is disjoint across splits, or prove `sequence_id ≡ scenario` 1:1.
- **Held + CIs:** true-CSI ceiling; headline on **held** scenes, ≥5 seeds, paired per-seed CI.

### B.3 Decision rule + honest-negative branch (with the MAJOR-2 gate + MINOR-3 robustness)
Arms {stale_echo, MSE, **recalibrated_MSE**, BWAR, BWAR-marginal-slot, true_csi}, identical anchor/decoder/budgets.
**POSITIVE:** paired `Feas(BWAR) − Feas(recalibrated_MSE)` CI `> 0` on held (≥5 seeds) ⇒ decision-focus converted
beyond a global shift; proceed to DF4.
**NEGATIVE is honest ONLY IF (pre-registered gate, MAJOR-2 + MINOR-3):** (i) the **BWAR-marginal-slot** variant
(budget/mutual-aware weighting) is ALSO run and also spans 0 — else the null is a train≠deploy artifact, not
aleatoric; (ii) a val-split sweep over `(β, c_fd, κ)` and both functional forms (fixed-`w_e` vs marginal-slot),
selected on **val feasibility**, has its **val-best** config still spanning 0 on held — else the negative is
premature (a mis-tuned config); (iii) the boundary-flip audit shows per-edge calibration provably improved on
**rank-marginal, mutually-live** edges (so we know the objective hit the decision-relevant edges). Only then:
decisive evidence the limit is **information-in-features (aleatoric)** — extends (not contradicts) T6. This is why
Goal 2 (raise `ρ` so the info exists) and Goal 1 (convert it) are **complementary**: DF2 says whether the info
exists; DF3 says whether it converts.

---

## Part C — Goal 3: activation (DF4, empirical)
Per-edge leaky-tanh is sound (bounded + gradient bounded away from 0); softmax over per-edge logits is a category
error (global competition, N-dependence, wrong off-diagonal gradient). High-value alternative: an **edit-selection**
head (top-k / Gumbel-softmax over near-boundary candidate edits) that consumes only the rankable **direction**
signal (`dir_acc≈0.92`) and sidesteps the magnitude wall. **A/B arms** {plain-tanh (saturation control),
leaky-tanh, small-scale-linear + `λ‖z‖²` penalty, top-k/Gumbel edit-selector}; hold everything but the head fixed;
feasibility vs the stale-degraded anchor with paired CIs.
- **Matched-capacity control (MINOR-4):** the heads differ in parameter count / output dim; add a matched-capacity
  arm (e.g. linear+penalty sized to the selector) so an edit-selector win is attributed to the *edit-selection
  inductive bias*, not raw capacity.
- **Selection-appropriate temporal-use check (MINOR-4):** the recurrent-vs-memoryless *non-identity of the scalar
  logit* check does NOT transfer to a discrete edit-selector (two logit fields can give identical top-k). Use a
  selection-appropriate metric (the selected edit-set distribution, or expected selection under Gumbel, differs
  between recurrent and memoryless) for that arm, or drop the check for it and state why.
**Overturn condition:** the edit-selector's paired CI clears the anchor where leaky-tanh does not (at matched
capacity) ⇒ the binding limit was the per-edge magnitude requirement ⇒ switch to edit-selection.
