# DF2 — Mechanism-Path Matrix

| Mechanism | State at HEAD (`e0c59a7`, DF1) | DF2 finding | Path |
|-----------|-------------------------------|-------------|------|
| `d_corr` knob → true channel | CALLABLE (DF1) | raises per-link ρ on idealized links (DF1); washed out (ρ≈0) in fast deployment scenes | ACTIVE_IN_EVAL (knob), default-off |
| MSE-realizable nowcaster → anchor psucc | IMPLEMENTED_ONLY (diagnostic) | `feas(realizable) == feas(stale)` at all `d_corr` (real MSE gain, zero feasibility movement) | measured in EVAL; NOT deployed (does not beat anchor) |
| Pure-temporal recovery (2 stale lags) | NOT_PRESENT | ≤0.006 decision-side advantage over echo at all ρ (MSE⊥decisions) | diagnostic only |
| Geometry recovery (current distance) | — | the only non-trivial decision advantage (+0.044), and only at LOW ρ | leak-free (current geometry), diagnostic |
| True-CSI oracle → anchor | (T1) | converts (RGF=1, headroom CI-positive) — the ceiling is real | training-label only |

## Effect on the decision (the campaign's recurring guard)
DF2 is explicitly an effect-on-**topology** test: it scores the produced topology's feasibility on the true
evaluator, not a forward value. The finding is that a real per-link prediction improvement (lower MSE) does **not
change the final topology feasibility** — the exact MSE⊥decisions failure mode the campaign tracks, now shown for
both the env knob (goal 2) and pure-temporal recovery.

## No new deployed path
DF2 adds only diagnostics (`df2_oracle_regate_gen.py`, `df2b_persistent_recovery_probe.py`). No deployed-path
change; the `d_corr` knob remains opt-in/default-off (DF1). Nothing here is ACTIVE_IN_DEPLOY.

## Leak-freeness
All predictor arms read only leak-free features (stale psucc lags, current distance/velocity, csi_age) — never
true current CSI (which is the training label / the oracle arm only). The `d_corr` knob is channel-side
(evaluator), never in the actor observation (DF1 by-construction argument).
