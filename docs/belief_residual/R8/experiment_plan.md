# R8 — experiment plan: the stale-CSI PREMISE test (does the method repair the stale feasibility drop?)

## Why this is the campaign's premise (and genuinely new)
The whole campaign exists because STALE CSI degrades the deployable anchor (urban ~0.80 → ~0.66; Q14). But
R1–R7 mostly ran with the stale-CSI overlay **OFF** (current-frame channel) to isolate each mechanism as one
variable. R6/R7 showed that ON THE CURRENT CHANNEL no deployable mechanism beats the near-optimal anchor. R8
turns the overlay **ON** and asks the campaign's actual question: **when the anchor is degraded by stale CSI (so
there is ROOM to repair), does the evidence-gated action repair the drop?** This is a different regime, not a
re-run — the current-channel anchor was near-optimal (no room); the stale anchor is not.

## One variable vs R6
**The observation regime: current-channel → stale (delay-1).** Everything else is the R6 evidence-gated action
(frozen heads → repair/safety/edit gate → budget-safe mutual decode, 0 eval). The heads are RETRAINED on stale
features (the deployed actor sees stale ef; labels come from the TRUE current evaluator = training-only), so the
gate is honest for the stale regime.

## Mechanism (leak-free by construction — verified)
`CsiObservationModel(mode="delay", delay_frames=1)` passed to the scene samplers overwrites the actor's CSI
columns (0–3) with the value from frame t−1 and appends `[csi_age, csi_observed_mask]` (ef dim 8 → 10). The
`context.evaluator` is UNTOUCHED — reliability/energy/latency/reward stay on the TRUE current channel (Spec
S4.6). So the deployed actor observes STALE CSI while the score is TRUE (test
`test_stale_overlay_changes_obs_not_evaluator`: CSI cols diverge at t≥1, evaluator identical stale-vs-current).

## Arms (deployed on held, 0 eval at decision; true evaluator scores only the produced topology)
- **current_anchor** — the anchor on the CURRENT channel (the non-degraded reference / ceiling).
- **stale_anchor** — the deployable anchor under STALE CSI (the degraded baseline — what we try to repair).
- **stale_gated** — the evidence-gated action under STALE CSI, heads retrained on stale features (the test).

## Failing-first load-bearing tests (Contract v4 §3)
`tests/unit/test_belief_residual_R8_stale_csi.py` (fail on HEAD: the R8 stale build helper absent):
1. `test_stale_overlay_changes_obs_not_evaluator` — the delay-1 overlay changes the observed CSI cols at t≥1
   (ef dim +2 for [age, mask]) AND the evaluator is identical stale-vs-current (leak-free: actor stale, score true).
2. `test_build_csi_scenes_active_flag` — the helper builds stale + current scenes; the deployed decode is the
   same 0-eval `evidence_gated_residual` used at R6.

## Measurement (pilot → 5-seed; Contract v4 §15)
5 seeds × {random, urban}. Per seed: current_anchor_feas, stale_anchor_feas, stale_gated_feas, **stale_drop =
current_anchor − stale_anchor** (the drop we target), **gated − stale_anchor feasibility & return** (the
repair), stale_gated edit_rate / unsafe_edit. 95% CI on (gated − stale_anchor) feasibility.
Pilot (urban, 1 seed, tiny — NOT a headline): stale_drop 0.25 (0.625→0.375), stale_gated +0.0625 over the stale
anchor, 0 unsafe — hopeful, but the 5-seed CI decides.

## Exit condition (Contract v4 §15)
- (gated − stale_anchor) feasibility CI **> 0** on a regime → the evidence-gated action REPAIRS the stale drop →
  the campaign's FIRST genuine deployable positive (the method helps in its actual failure regime). KEEP → R9
  (full 5-seed campaign) / R10.
- (gated − stale_anchor) CI **spans 0** → does not significantly repair the drop → negative under the real
  regime too; report the stale_drop and the (mean) repair, scope it path-specifically.
- (gated − stale_anchor) **< 0** → the gate makes the stale regime WORSE → REVISE (report unsafe_edit).
Also report whether the feasibility repair costs RETURN (feasibility ↑ but return ↓ would be a trade-off, not a
clean win).

## Hard constraints
Deployment decentralized (0 eval at decision); the actor observes STALE CSI, NEVER the true current CSI / critic
/ evaluator / solver / global decoder; true C/E/L/J on the current real channel + closed-form quorum tail; the
true-evaluator labels for the heads are training-only; only anchor-relative LOCAL edits; held never trains the
heads; ≥5 seeds + CI (the 1-seed pilot is NOT a headline); commit to trunk, no push.
