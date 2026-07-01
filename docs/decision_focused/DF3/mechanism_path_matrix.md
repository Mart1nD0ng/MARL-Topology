# DF3 — Mechanism-Path Matrix

| Mechanism | State at HEAD (`40c6962`, DF2) | DF3 finding | Path |
|-----------|-------------------------------|-------------|------|
| BWAR decision-focused objective | NOT_PRESENT | no detectable feasibility gain over MSE (all CIs span 0, point est. negative); engaged the boundary (dir_acc +0.079) but info-limited | ACTIVE_IN_LOSS (diagnostic training-only); **NOT adopted** |
| Recalibration control (2-param global transform) | NOT_PRESENT | == plain MSE at 5 seeds (no calibration gain) | diagnostic only |
| Marginal-slot (budget-margin) weighting | NOT_PRESENT | −0.031 vs MSE (CI spans 0) | diagnostic only |
| Plain MSE belief objective (`csi_belief.py`) | ACTIVE_IN_LOSS (opt-in) | the best realizable objective; small edge over echo (+0.022, calibration) | unchanged, KEEP |
| True-CSI oracle → anchor | (T1) | converts (+0.131 paired over mse, p=0.0022) — the ceiling, uncrossed by any realizable objective | training-label only |

## Effect on the decision (the campaign's recurring guard)
DF3 is an effect-on-**feasibility** test (scored on the true evaluator). The B.3(iii) audit additionally verifies
effect on the **per-edge decision**: BWAR moved per-edge decision-direction accuracy (+0.079 dir_acc) — so the
mechanism DID change the decision-relevant quantity — yet the graph-level feasibility did not move. A real per-edge
engagement that does not convert to topology feasibility: the MSE⊥decisions failure mode, now shown to persist
even when the objective is decision-aligned AND provably engaged.

## No new deployed path
DF3 adds only diagnostics (`df3_decision_focused_gen.py`, `df3b_bwar_hpsweep.py`, `df3c_marginal_slot.py`,
`df3d_boundary_audit.py`). No deployed-path change; the deployed belief objective remains plain MSE (opt-in).
Nothing here is ACTIVE_IN_DEPLOY.

## Leak-freeness (verified by DF3-verify, methodology checks PASS)
Features are leak-free (stale CSI + current geometry + age; true CSI only as the training label); train/held use
disjoint seeds; the anchor code path is shared train==deploy (τ from prev-topo membership, gates 0.4/0.6). No
leakage or evaluation bug found.
