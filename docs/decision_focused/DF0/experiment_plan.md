# DF0 — Experiment Plan (framing + research + validation design)

**Stage type:** framing / research / design. **No experiments, no training, no code changes to deployed paths.**
Contract v4: `experiment_plan` + Claim Card + Mechanism-Path Matrix + validation design **before** any campaign
run. Base HEAD `1bf773f` (clean). Branch `decentralized-marl-trunk`.

## Purpose
Frame the Decision-Focused (DF) campaign — successor to Temporal-Recovery (T0–T7) — attacking that campaign's
binding limit ("deployable precision at the anchor decision boundary; MSE ⟂ decisions") on the three axes the
owner named, and (the owner's explicit demand) get the **validation design for goals 1 and 2 provably correct**
before running anything.

## What DF0 did
1. **Oriented + verified** the load-bearing facts against code (not assumed):
   - Anchor `local_hysteresis_proposals` = budget-`b_i` top-k of {kept ≥ 0.4} ∪ {new ≥ 0.6}, keep-priority, then
     mutual-accept AND (`dynamic_baselines.py:48-76`). NOT an independent per-edge threshold.
   - The campaign channel is **already shadow-faded** (`operating_point_regime`: `v2x_37885`+`shadowing_37885`+
     `nlosv_37885`; `dynamic_rl.py:562`). Weak recoverability is because at `dt=1s`/urban-speed the shadow field
     (10 m cell) decorrelates in ~1 frame (`ρ≈0.37`) — not because the channel is deterministic.
   - Verified leak vectors for goal 2: current-distance col 5 (never staleified); `motion_features` `csi_delta`
     uses current psucc; `lru_cache` seed-collision risk on the shadow field.
2. **Research (3 independent lenses, adversarial):** decision-focused prediction methods → **BWAR** objective;
   env temporal-autocorrelation → the decorrelation-distance knob + a full leak/fake-positive red-team; activation
   → leaky-tanh sound, softmax a category error, edit-selection the high-value next mechanism.
3. **Wrote** `CURRENT_DECISION_FOCUSED_STATUS.md` (framing + gap-from-HEAD + hard constraints) and
   `DF0/validation_design.md` (THE crux — every guard is a named failing-first test).

## Campaign plan (one variable per stage; a negative at any gate is a valid scoped result — NO downgrade)
- **DF1 (goal 2 capability):** expose `shadow_decorrelation_distance_m` as a config knob; failing-first tests;
  PROVE leak-free (R² gate A.2a), byte-identical-off, knob monotonicity (`ρ(d_corr)`). Commit.
- **DF2 (goal 2 gate):** 4-arm oracle {stale, mse_realizable, true_csi, geometry_only} across the decorrelation
  sweep; metric `RGF`; primary endpoint `RGF(recovered) − RGF(geometry_only)`. GATE per validation_design A.3.
- **DF3 (goal 1):** implement BWAR (opt-in); compare vs MSE vs echo vs oracle on anchor feasibility (held, ≥5
  seeds), under BOTH original and modified channel; effect-on-decision audit. Decision rule per B.3.
- **DF4 (goal 3):** activation A/B {plain-tanh, leaky-tanh, linear+penalty, edit-selection}; integrate winner.
- **DF5:** 5-seed campaign on the winning config; per-seed + CI; both channels scoped.
- **DF6:** docs close-out + 中文 report + memory.

## Method choices (from the research, with rationale)
- **Goal 1 objective = BWAR** (boundary-weighted asymmetric regret-surrogate). SPO+ collapses to a boundary hinge
  for a threshold decision; differentiable-opt is over-engineered for a feature-limited-precision bottleneck.
- **Goal 2 knob = decorrelation distance** `d_corr` (single channel-model parameter; governing ratio `v·dt/d_corr`),
  swept {10,25,50,100} m with standards scoping (urban/highway/UMa/stress). `dt`-refinement noted as the
  urban-preserving equivalent.
- **Goal 3 = keep leaky-tanh per-edge**; A/B an edit-selection (top-k/Gumbel) head as the mechanism that exploits
  the rankable direction while sidestepping the magnitude wall.

## Risks / open questions carried into DF1+
- BWAR may ALSO be an honest negative if the boundary miss is aleatoric (info not in features) — which is exactly
  why goal 2 (raise `ρ`) and goal 1 (convert) are complementary and tested together.
- Physical honesty of `d_corr > 25 m` for an "urban" claim — scoped as highway/UMa/stress, never headline-urban.
- NLOSv autocorrelation confound — hold fixed or run a sensitivity arm.

## Exit criteria for DF0
- [x] Status doc + validation_design + this plan + Claim Card + Mechanism-Path Matrix written.
- [x] The validation design adversarially verified (2 independent critics, both REVISE) → all fixes folded into
      `validation_design.md` **v2** (marginal-invariance gate, nested-subset arms, pooled RGF, recalibration
      control, marginal-slot negative-gate).
- [ ] DF0 committed (docs-only) to `decentralized-marl-trunk` (no push).
