# DF0 — decision

**Disposition: DESIGN COMPLETE + ADVERSARIALLY VERIFIED (REVISE → all fixes folded into validation_design v2).**
No code changed; no experiment run; suite unchanged.

## Adversarial verification outcome (the last DF0 gate)
Two independent critics attacked the finished `validation_design.md` (each verified claims against code). **Both
returned REVISE** — the v1 design would have produced invalid answers. All MAJOR + MINOR fixes are folded into
**v2**. The load-bearing catches:
- **Goal 2 / M4 (most dangerous):** the `d_corr` sweep could move the psucc **marginal** (difficulty), not just
  the autocorrelation → a spurious "recovery converts" positive. v2 adds a **marginal-invariance KS gate** +
  field-seed averaging; no sweep result is valid until the marginal is proven invariant.
- **Goal 2 / M1+M2:** the v1 leak gate predicted the wrong target (current distance legitimately drives psucc) and
  the `geometry_only` null was an ill-defined "distance→psucc map." v2 unifies all arms as **best-causal-predictors
  from nested feature subsets** (distance-only = geometry null); leak test → **incremental R² over distance** +
  boundary-restricted + per-scene tests.
- **Goal 2 / M3:** RGF is now a **pooled-mean ratio bootstrapped over scenes** (per-frame fractions were 0/0).
- **Goal 1 / MAJOR-1:** added a **recalibration control arm**; headline is `BWAR − recalibrated-MSE` (BWAR's
  asymmetric hinge can win by a global bias shift).
- **Goal 1 / MAJOR-2:** the marginal-slot/mutual audit + a val hyperparameter sweep now **gate the aleatoric
  negative**; boundary-flips defined on the mutual-decoded topology.

This is exactly the value the owner asked for ("确保验证方案的正确性") — the verification caught defects that would
have inverted the campaign's honest-negative posture into a fake positive.

## What DF0 established
- The campaign is framed as the successor to Temporal-Recovery, attacking its binding limit on the three
  owner-named axes; tasks DF0–DF6 created; base HEAD `1bf773f` clean.
- **Two load-bearing facts were re-verified against code** (not trusted from the summary):
  1. The deployed anchor is budget-`b_i` top-k of {kept ≥ 0.4} ∪ {new ≥ 0.6} + mutual-accept AND
     (`dynamic_baselines.py:48-76`) — a *budget/mutual* decision, which strengthens the edit-selection case.
  2. The campaign channel is already shadow-faded; weak recoverability is **fast decorrelation** (`ρ≈0.37` at
     `dt=1s`, urban speed, 10 m cell), not a deterministic channel. Goal 2 = **expose + sweep the decorrelation
     distance**, not "enable shadowing."
- The decision-focused objective is chosen (**BWAR**), the activation question is answered (leaky-tanh sound;
  softmax a category error; edit-selection the high-value next mechanism), and the **validation design for goals 1
  and 2 is written with every guard as a named failing-first test** (`validation_design.md`).

## Method dispositions (KEEP into DF1+)
- **Goal 2:** decorrelation-distance knob on the existing physical shadowing; five honesty guards; oracle re-gate
  with the geometry-null primary endpoint `RGF(recovered) − RGF(geometry_only)`.
- **Goal 1:** BWAR objective (opt-in, training-only), scored on deployed feasibility not MSE; complementary to
  goal 2 (goal 2 makes the info exist; goal 1 converts it).
- **Goal 3:** keep leaky-tanh per-edge; A/B edit-selection at DF4.

## Why no seeds/CI in this stage
DF0 makes **no empirical headline** — it is design. The only falsifiable commitments are the leak-free R² test,
the `ρ(d_corr)` monotonicity, and byte-identity, all of which are *runnable predictions* checked in DF1–DF2 with
their own Claim Cards + CIs. This is a legitimate seedless stage (Contract: seeds are required for *headlines*).

## Next
1. **Commit DF0 (docs-only)** to `decentralized-marl-trunk` (no push).
2. **DF1** (goal 2 capability): expose `shadow_decorrelation_distance_m` as a config knob; failing-first tests in
   this priority order (per v2): (a) **marginal-invariance** KS gate across `d_corr` — the highest-risk control;
   (b) the incremental-over-distance / boundary-restricted / per-scene **leak tests**; (c) `ρ(d_corr)` monotonicity
   (field-seed averaged); (d) byte-identical at `d_corr=10`; (e) `lru_cache` seed-fold + `cache_clear`. Set
   `motion_features=False` for leak-critical arms.
3. **DF2** builds the nested-subset predictor arms (`geometry_only`/`mse_realizable`) and runs the oracle re-gate.

**No downgrade:** if any gate later fails (DF2 realizable-conversion, DF3 BWAR conversion, DF4 edit-selection),
the result is reported as a scoped honest negative that *refines* the binding limit — not softened to force a
positive.
