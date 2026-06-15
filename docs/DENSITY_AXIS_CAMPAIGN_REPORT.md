# Density-Axis Scaling Campaign — Experimental Report

**Date:** 2026-06-15 · **Physics:** full TR 37.885 stochastic stack (v2x_37885 + shadowing +
NLOSv + coverage-gated membership + wired RSU backhaul + scheduled MAC + relay-3, τ=0.9) ·
**Metric:** M-draw (M=3) robust feasibility, graded at P̂≥0.8.

Figures: `docs/figures/fig1_scaling_law.png`, `fig2_density_sweep.png`, `fig3_decoder_lift.png`,
`fig4_end_to_end.png`.

## 0. Frozen formal model + production config (locked before the campaign)
- **Deployed actor:** size-invariant K-round message-passing GNN edge scorer, model-freeze
  artifact `_artifacts_step3.pt` (best index by `step3_result.json`), keep-best-eval selection.
- **Baseline decoder:** local mutual-acceptance. **#2** quorum-aware decoder and **#3**
  distributional/robust head are *treatment arms vs this baseline*; **#1** pooling is OFF (null).
- **Scaling control:** density-preserving — vehicles/km² AND RSU/km² held fixed (rsu:veh ≈ 1:3,
  continuous block size) so only system size N varies. N ∈ {8,12,16,24,32,48}.

## 1. Task 1a — Evaluator vectorization (enabling step)
`logs/fast_stage21.py`, opt-in, src untouched. Precomputes the topology-independent per-ordered-
pair desired rx_power once (the only ray-box work; interference is then a sum of precomputed
powers) and skips the required-time bisection (unused under a fixed transmission time), reusing
the exact src route-finding + PBFT downstream. **Bit-identical** consensus p0 (max|Δp0| = 0.0e+00
across random topologies, N=8/12/16/24 × 2 realizations); **4.5×–8.5× faster** (455 ms vs 3.85 s
at N=48). Enables the SA teacher at N≥16 and the density sweep to N=48.

## 2. Task 3b — Density isolation (fixed N=16, vary city area), GRADE metric
Vehicle density isolated. Robust work domain (a robust-feasible topology exists) rises with
density: 10.5 veh/km² → 0.33, 23 → 0.50, 40 → 0.67, 87 → 1.00 (denser city = shorter hops, more
LOS, easier consensus). A persistent **learnability gap** (work domain − frozen-actor grade)
≈ 0.17–0.33 at every density. Cliff prevalence (#3 headroom) is high (0.34–0.44) across most
densities. Real-actor #2 decoder lift (B−A) = 0 at every density cell.
*(6 scenes/cell → grade resolution 1/6; the 15 veh/km² point is within that noise.)*

## 3. Task 3a + SA work-domain ceiling — THE REFRAMING
For each N the sweep measures two things: what is **found** (frozen actor + heuristic candidate
pool) vs whether a robust-feasible topology **exists** (budget-aware SA relay search). The
earlier integer-blocks sweep showed feasibility "collapsing" to 0 at N≥24 — but that was the
heuristic pool failing, not physics:

(exact density 40 veh/km², 13.3 RSU/km², 5 scenes/cell, M=3; `logs/_task3a_clean.json`)

| N | work domain (best of pool/SA finds robust-feasible) | frozen actor (A) grade | learnability gap |
|---|---|---|---|
| 8 | 0.60 | 0.60 | 0.00 |
| 12 | 0.80 | 0.80 | 0.00 |
| 16 | 1.00 | 1.00 | 0.00 |
| 24 | 0.60 | **0.00** | **0.60** |
| 32 | 0.60 | **0.00** | **0.60** |
| 48 | 0.60 | **0.00** | **0.60** |

**Result:** the work domain does NOT collapse with N — a robust-feasible topology keeps existing
(SA finds one in ~60% of scenes at every N≥24; the heuristic pool finds **none**, feas#=0). What
collapses is the small-N-trained frozen actor's ability to FIND it: it tracks the work domain
perfectly to N=16, then drops to **0.00** at N≥24, so the **learnability/search gap is ~0.60 and
flat** across N=24/32/48. This updates the prior "innovations cannot expand the work domain"
framing: physics still sets the work domain (and SA shows it is larger than the heuristics
reveal), but the **advantage domain** — where the learned model reaches the ceiling — has a large,
sharply-opening gap at N≥24. The scale barrier is **learnability, not physics**.

## 4. Task 2 — Real-actor #2 decoder lift (replaces the proxy-logit upper bound)
The advantage-domain proxy (threshold-reliability logits) had suggested decoder lift up to +0.25.
With the **real frozen actor**, the lift is +0.22 only at N=8 (one integer-blocks cell) and **0**
across every fixed-density N=16 density cell and every clean scale cell. The real actor already
emits connected topologies, so #2's backbone repair rarely triggers; the proxy disconnected more,
inflating the lift. **Honest correction:** #2's deployment value is a *non-isolation safety floor*,
not a mean-feasibility gain on the trained actor.

## 5. Task 1b — Scaled-operating-point end-to-end (A/B/C)
A = frozen actor + local_mutual; B = frozen actor + quorum_aware (#2); C = a fresh actor
retrained on budget-aware SA-backbone targets (robust override if cliff) then executed via
quorum_aware. 24 held + 24 train scenes/cell, M=3, exact density 40 veh/km². (`fig4`.)

| Cell | A robust / grade | B robust / grade (#2) | C robust / grade (#2+#3) | #3 override |
|---|---|---|---|---|
| N=16 (4 RSU) | 0.708 / 0.667 | 0.708 / 0.667 | **0.806 / 0.708** | 0.00 |
| N=24 (6 RSU) | **0.000 / 0.000** | 0.000 / 0.000 | **0.486 / 0.417** | 0.00 |

**This is the decisive end-to-end confirmation of the reframing.** At N=24 the frozen small-N
actor achieves *literally zero* robust feasibility (the learnability cliff), while a fresh actor
retrained on SA backbones recovers it to **0.486 robust / 0.417 grade** — most of the ~0.60
work-domain ceiling. So the N≥24 collapse is closed by **scale-appropriate training on strong
targets**, proving it was learnability, not physics. At N=16 retraining gives a smaller bump
(0.708→0.806). In both cells the gain is the *retraining*, not the innovations: **#2 gives no
lift (B=A)** and **#3 never triggers (override rate 0** — the SA backbones were already robust at
these cells), consistent with Tasks 2/3b. The deployment lesson: at scale, the bottleneck is
finding/learning the feasible backbone (SA-teacher distillation), with #2/#3 as situational
safety mechanisms rather than mean-feasibility drivers.

## 6. Conclusions
- Vectorization makes scale tractable (bit-identical, ~5–8×).
- Density is a first-order axis: denser cities are more feasible; fragility (cliff) peaks at
  marginal density — both cleanly isolated.
- The central scaling result is a **learnability gap, not a work-domain wall**: feasible
  topologies exist at N≥24 but the small-N actor/heuristics fail to find them. This is the
  motivation for the scale/density chapter and the regime where stronger search/training (and
  the robustness/connectivity innovations) must operate.
