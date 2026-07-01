# DF1 — decision

**Disposition: KEEP the knob (opt-in, byte-identical default). Both env-property gates PASS.** DF1 delivers the
goal-2 capability and proves it is honest; it does NOT yet claim any feasibility conversion (DF2).

## Outcome
- **Knob implemented, byte-identical off:** `shadow_decorrelation_distance_m` on `ChannelModelConfig` +
  `PhysicsRegime`, threaded through the shadow field. 7 failing-first tests now pass; existing 37885 suite 10/10.
- **ρ-monotonicity gate: PASS** — `ρ(psucc_t, psucc_{t-1})` rises 0.044 → 0.429 monotonically with d_corr (rise
  +0.384 ≫ 0.1, CI-separated). The knob creates the intended tunable temporal autocorrelation of the
  decision-critical psucc.
- **Marginal-invariance gate (critic-1 M4): PASS** — success rate 0.64–0.67, no monotone drift, max KS 0.031. The
  sweep moves ρ, not difficulty → **no fixed-marginal renormalization needed** (a candidate midpoint-sample
  variant was considered and is unnecessary at this operating point). DF2 keeps the geometry-null differencing as
  belt-and-suspenders for any residual.

## Why this is not a downgrade
The env-property gates are exactly the honesty controls DF0's adversarial review demanded. Both pass on their
merits (measured, CI'd, field-seed averaged). The knob is a physically-real TR 36.885/38.901 parameter; the sweep
is scoped urban(10)/highway(25)/UMa(50)/stress(100). Nothing was weakened to force a pass — the marginal check in
particular was set up to FAIL if the knob confounded difficulty, and it passed on the evidence.

## Carried to DF2 (single-variable discipline)
- Build the **nested-subset predictor arms** (`geometry_only` = best causal predictor from distance alone;
  `mse_realizable` = from the full leak-free set) and the **statistical leak battery** (G1 residual-shadow, G2
  incremental-over-distance, G3 boundary-restricted, G4 per-scene) + `motion_features=False` enforcement — these
  need the predictor/observation harness DF2 builds.
- Run the **4-arm oracle re-gate** across the sweep; metric `RGF` (pooled-mean ratio, scene-bootstrap CI); primary
  endpoint `RGF(recovered) − RGF(geometry_only)`. GATE per validation_design A.3.
- The oracle uses the faithful channel (NLOSv ON); the DF1 probe held NLOSv OFF only to isolate the knob.

## Decision
**KEEP.** Commit DF1 (no push). Proceed to DF2. If DF2's realizable arm fails to convert the (now genuinely
present, marginal-invariant) temporal structure, that is a valid scoped honest negative — NOT a reason to inflate
the knob beyond its standards-defensible range.
