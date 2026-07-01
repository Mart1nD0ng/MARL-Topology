# DF1 — Experiment Plan (goal 2 capability: the decorrelation-distance knob)

**Single variable:** expose the shadow-fading spatial decorrelation distance
(`SHADOW_DECORRELATION_DISTANCE_M`, a hardcoded 10 m constant) as a config-threaded, swept knob
`shadow_decorrelation_distance_m`, so the decision-critical psucc gets tunable **temporal** autocorrelation
(governing ratio `v·dt/d_corr`). Base HEAD `735e5a7` (DF0). Contract v4.

## Why (from DF0 validation_design v2 A.0–A.2)
The campaign channel is already shadow-faded, but at `dt=1 s` / urban speed the 10 m field decorrelates in ~1
frame (`ρ≈0.37`) → near-i.i.d. → the T6 weak recoverability. Raising `d_corr` is the physically-correct lever
(TR 36.885 urban 10 m → highway 25 m → UMa 50 m; `ρ ≈ exp(−v·dt/d_corr)`). DF1 delivers the knob + PROVES it is
honest (leak-free by construction, byte-identical off, knob does what we claim).

## What DF1 did
1. **Failing-first tests** (`tests/unit/test_df1_shadow_decorrelation_knob.py`, 7 tests) — all failed on HEAD
   (knob absent), then pass after the minimal diff.
2. **Minimal diff (byte-identical default 10 m):**
   - `ChannelModelConfig.shadow_decorrelation_distance_m: float = SHADOW_DECORRELATION_DISTANCE_M` (+ `>0`
     validation).
   - `_shadow_field_value(seed, x, y, decorr_m=…)` and `_shadowing_37885_db(…, decorr_m=…)` thread the cell size;
     `_channel_terms` passes `config.shadow_decorrelation_distance_m`. **No lru_cache collision** — the cached
     corner normals are cell-size-independent, only the position→lattice mapping is re-scaled (so no seed-fold is
     needed; the field re-scales cleanly and the same underlying lattice is held across the sweep, which is the
     *cleaner* controlled knob).
   - `PhysicsRegime.shadow_decorrelation_distance_m` (appended LAST for positional-construction safety) +
     `channel_config()` passthrough (`getattr` default 10 m for pre-existing pickled regimes).
3. **Env-property probe** (`scripts/diagnostics/df1_decorr_env_probe.py`): moving-vehicle trajectories, per-frame
   decision-critical psucc via `evaluate_channel`+`evaluate_link_transmission`, NLOSv **off** to isolate the
   shadow knob, field-seed averaged. Measures `ρ(psucc_t, psucc_{t-1})` monotonicity and the psucc marginal (KS).

## Results (150 links × 6 field seeds × 8 frames, tx=−8 dBm → psucc at the 0.4/0.6 boundary)
| d_corr (m) | mean ρ (95% CI half) | psucc mean/std | KS vs d10 |
|-----------|----------------------|----------------|-----------|
| 10  | +0.044 (0.022) | 0.637 / 0.477 | 0.000 |
| 25  | +0.314 (0.026) | 0.655 / 0.472 | 0.020 |
| 50  | +0.391 (0.015) | 0.638 / 0.477 | 0.004 |
| 100 | +0.429 (0.036) | 0.665 / 0.468 | 0.031 |

- **ρ-monotonicity gate: PASS** — monotone non-decreasing, rise +0.384 ≫ 0.1, tight CIs. The knob creates
  tunable temporal autocorrelation of the decision-critical psucc.
- **Marginal-invariance gate (critic-1 M4): PASS** — success rate 0.637–0.665, **no monotone drift**, max KS 0.031
  (an earlier KS 0.085 at 2 field seeds was realization noise). The sweep moves ρ, **not** difficulty → no
  fixed-marginal renormalization needed. DF2 keeps the geometry-null differencing as belt-and-suspenders.

## Leak-free disposition
The knob lives entirely in `ChannelModelConfig` (evaluator/channel side). The actor observation is built by the
staleifier, which never sees the channel config or the shadow field → **the knob is leak-free by construction**
(it changes the true channel the evaluator scores, adds nothing to the actor obs). The *statistical* leak battery
(incremental-R²-over-distance, boundary-restricted, per-scene) and the `motion_features=False` enforcement
(csi_delta uses current psucc) are built with the predictor arms in **DF2**.

## Exit criteria
- [x] Failing-first tests → 7/7 pass; existing 37885 shadowing suite still 10/10 (byte-identical).
- [x] ρ-monotonicity + marginal-invariance gates PASS (probe artifact `env_probe_metrics.json`).
- [x] Full unit suite regression-clean: **850 passed** (843 + 7 new DF1), 0 failures.
- [ ] DF1 committed (no push); DF2 next (nested-subset arms + oracle re-gate).
