# DF1 — Claim Card

**Claim:** Exposing the shadow-fading decorrelation distance as a config knob `shadow_decorrelation_distance_m`
gives the decision-critical psucc **tunable temporal autocorrelation** — `ρ(psucc_t, psucc_{t-1})` rises
monotonically from 0.044 (d_corr=10 m) to 0.429 (d_corr=100 m) — **while the psucc marginal stays ~invariant**
(success rate 0.64–0.67, no monotone drift, max KS 0.031). The knob is **byte-identical to HEAD at the 10 m
default** and **leak-free by construction** (channel/evaluator side only).

**Path covered:** the ENV-CAPABILITY path. DF1 asserts the knob exists, is honest, and produces the intended
temporal structure. It does **NOT** assert that a nowcaster converts this into feasibility (that is DF2's oracle
re-gate).

**Evidence:**
- `tests/unit/test_df1_shadow_decorrelation_knob.py` — 7 tests, failing-first on HEAD, now pass: byte-identity at
  default, positivity validation, field-value re-scales with d_corr, the module-constant default path,
  larger-d_corr-raises-spatial-correlation (the core claim), channel-record threading, regime threading.
- `tests/unit/test_37885_shadowing_nlosv.py` — 10/10 still pass (byte-identity of the existing shadowing model).
- `scripts/diagnostics/df1_decorr_env_probe.py` + `docs/decision_focused/DF1/env_probe_metrics.json` — the
  ρ-monotonicity and marginal-invariance measurement (150 links × 6 field seeds × 8 frames, tx=−8 dBm, NLOSv off).

**Seeds/CI:** the probe averages over 6 field-seed realizations with a seed-level 95% CI on ρ (per validation_design
M4). The ρ rise is CI-separated across d_corr; the marginal KS is small at all d_corr.

**Falsifiable / what would INVALIDATE this claim:**
- If ρ were NOT monotone in d_corr, or the rise ≤ 0.1 → the knob does not create temporal structure (it does:
  +0.384).
- If the psucc marginal drifted monotonically with d_corr (large KS) → the knob moves *difficulty*, a confound;
  then a fixed-marginal renormalization would be required before DF2 (it is not: KS ≤ 0.031, no drift).
- If the default (10 m) changed any channel record vs HEAD → not byte-identical (it is).

**Honesty notes:** NLOSv is held OFF in the probe to ISOLATE the shadow-decorrelation knob (its persistent
per-pair loss adds a d_corr-independent autocorrelation); the faithful (NLOSv-on) channel is what DF2's oracle
uses. The probe's psucc is the per-link `packet_success_probability` (the decision-critical col-0 feature),
measured at tx=−8 dBm so links sit near the 0.4/0.6 anchor boundary (at 20 dBm psucc saturates at 1.0 and ρ is
undefined — reported, not hidden).

**No-Silent-Citation:** all cited facts have file:line / artifact sources above.
