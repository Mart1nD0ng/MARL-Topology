# R6 — decision (evidence-gated residual action) — KEEP MECHANISM / HONEST NEGATIVE on the deployed gain

**Result: the evidence-gated residual action is a CORRECT, SAFE, deployable mechanism, but it does NOT beat the
anchor at any operating point — a DEPLOYABLE CONVERSION gap.** R4 proved the beneficial-edit signal exists; R5
proved the LOCAL heads can RANK it (held top-k > random). R6 asks whether that ranking, applied as a deployable
gated action (0 eval), converts into a feasibility/return gain over the anchor. Answer: **no.** At the neutral
threshold the well-trained heads suppress almost all edits → B == anchor; lowering the threshold to fire edits
makes it net-negative in mean at every firing threshold (unsafe edits dominate). The heads' only deployable value is safe *suppression*
(B > random-gate C in mean). **KEEP the mechanism (correct + safe + suppression value); HONEST NEGATIVE on the
deployed gain — the R5 ranking's ~40% top-k precision is insufficient to beat the anchor deployably at N≤16.**

## 5-seed A/B (`evidence_gate_metrics.json`; tau_edit 0.5, tau_repair 0, tau_safety 0; 0 eval at decision)
| arm | random feas | random ret | urban feas | urban ret | B edit_rate | B unsafe | B zero_edit |
|---|---|---|---|---|---|---|---|
| A anchor | 0.375 | −0.429 | 0.815 | −0.035 | 0 | — | — |
| **B gated (trained)** | 0.375 | −0.429 | 0.820 | −0.034 | 0.001 (r) / 0.001 (u) | 0.005 / 0.000 | 0.98 / 0.96 |
| C gated (untrained) | 0.280 | −0.522 | 0.700 | −0.157 | — | — | — |
- **B − A feas: random 0.000 [−0.022, +0.022]; urban +0.005 [−0.009, +0.019]** — spans 0 (B == anchor).
- **B − A return: random −0.0006 [−0.004, +0.002]; urban +0.0009 [−0.002, +0.003]** — spans 0.
- **B − C feas: random +0.095 [−0.068, +0.258]; urban +0.120 [−0.055, +0.295]** — B > C in MEAN (trained heads
  degrade far less than the random gate), not 95%-significant at n=5. The trained gate learned to REFRAIN.
- edit_rate ~0.001 (5-seed MEAN; per-seed subsets run an order higher, ~0.006), zero_edit ~0.97: at tau_edit
  0.5 the calibrated heads almost never pass the gate (most
  candidates are negatives — R5 base rate ~0.14–0.17 — so the BCE-trained edit prob rarely exceeds 0.5).

## Threshold sweep (`threshold_sweep.json`; reuse the per-seed trained actor, vary only tau_edit) — DECISIVE
B − A feasibility (mean) as the gate is loosened to fire more edits:

| tau_edit | random B−A feas | random edit / unsafe | urban B−A feas | urban edit / unsafe |
|---|---|---|---|---|
| 0.5 | **0.000** | 0.001 / 0.005 | **+0.005** | 0.001 / 0.000 |
| 0.35 | −0.020 | 0.008 / 0.010 | −0.070 | 0.093 / 0.090 |
| 0.25 | −0.045 | 0.014 / 0.025 | −0.055 | 0.097 / 0.085 |
| 0.15 | −0.060 | 0.015 / 0.035 | −0.055 | 0.099 / 0.085 |
| 0.05 | −0.050 | 0.016 / 0.030 | −0.045 | 0.106 / 0.085 |

**Downhill from the empty gate: every FIRING threshold is net-negative in mean, and NO tau beats the anchor.**
The optimal operating point is the EMPTY gate (tau 0.5, B == anchor); every threshold that fires edits is
net-negative in mean and raises unsafe_edit (urban 0.000 → 0.090). (The mean curve is not strictly monotone —
it drops sharply at the first firing threshold then partially recovers/flattens: random min at tau 0.15, urban
min at tau 0.35 — but stays net-negative throughout.) Return follows the same shape (B−A ret: random −0.0006 →
−0.022; urban +0.0009 → −0.061). **Significance note:** each individual firing-threshold B−A CI itself spans 0
(e.g. urban tau 0.35 −0.07 [−0.247, +0.107]); the firing-threshold negatives are directional/mean, and the
statistically load-bearing statement is that **NO tau has a B−A lower bound > 0** (no operating point
significantly beats the anchor), while the best point (tau 0.5) is B == anchor. **The best deployable edit
policy over the anchor is "make no edit".** (This empirically pre-empts R7's premise that an adaptive anchor-KL
trust region could help: the optimal anchor-deviation is ZERO.)

## Why (the mechanism-level diagnosis — Contract v4 §13: mechanism / path / trainer / data / obs / eval?)
This is a **DATA/precision** limit at the observation+evaluation scope, NOT a mechanism/path/trainer bug:
- The gate MECHANISM is correct: budget-safe, zero-gated→anchor exactly, 0 unsafe at tau 0.5, 0 eval at
  decision, load-bearing (5 tests + the B≫C contrast prove the trained heads drive it).
- The DIRECTION signal is real (R4) and locally rankable (R5, held top-k precision ~0.42 random / ~0.32 urban).
- BUT ~40% top-k precision means the majority of fired edits are non-beneficial, and in the feasibility metric
  a harmful edit (breaking a feasible frame) costs MORE than a beneficial edit gains → applying the ranking is
  net-negative. The head cannot separate beneficial from harmful edits sharply enough to fire only the winners.
- So the deployable policy's best move is the anchor. This SHARPENS the campaign's binding limit from
  "no direction signal" (pre-R4) to "the direction signal's local-rankable PRECISION is insufficient to beat
  the anchor deployably at N≤16".

## Effect-on-Decision (Contract v4 §3 — does it change the final topology?)
Yes, and measured: at tau 0.5 the trained gate changes the topology on ~2–4% of frames (edit_rate 0.001,
zero_edit 0.97 → most frames == anchor); the untrained gate changes far more and DEGRADES (C < A). So the gate
DOES change the final topology (unlike a forward-only no-op), and the trained heads' effect is net-neutral
(== anchor) while the random gate's effect is net-harmful. Lowering tau fires more edits (edit_rate up to 0.11
urban) and the effect turns net-harmful (B−A feas negative). The mechanism is fully load-bearing on the action.

## Path-specific scope (Contract v4 §5)
> "The evidence-gated residual action (frozen R5 heads, 0 eval, budget-safe, decentralized) does NOT beat the
> local_hysteresis anchor on deployed feasibility/return at ANY tau_edit ∈ [0.05, 0.5], on urban or random,
> N≤16, current-frame channel (stale overlay off). Best operating point = empty gate (B == anchor, 5-seed
> B−A CI spans 0); every firing threshold is net-negative in mean with rising unsafe edits (no tau's B−A lower
> bound exceeds 0). The R4 signal
> exists and the R5 local ranking is real, but its ~40% top-k precision is insufficient to CONVERT into a
> deployed gain — a deployable CONVERSION gap (distinct from R5's learnability positive: significant on random,
> strong-mean-but-not-significant on urban). The trained heads'
> deployable value is safe suppression (B > random-gate C in mean). NOT a mechanism/path/trainer failure."

## Tests (failing-first → pass)
5 in `tests/unit/test_belief_residual_R6_gate.py` (fail on HEAD: module absent → pass): gate calls the frozen
heads in the decision path (spy); zero-gated → anchor exactly; bad-head gates out / good-head applies (both
directions); budget-safe; no-evaluator signature audit. Full suite **810/0**.

## Acceptance (Contract v4 §15) / Decision
- Mechanism correctness REQUIRED — met: bad edits gated out (tests + 0 unsafe at tau 0.5), zero-candidate →
  anchor (test), budget-safe (test), 0-eval deploy (test). ✓
- Deployable gain: (B−A) CI spans 0 at the best threshold and goes NEGATIVE at every firing threshold → **NO
  deployed gain at any operating point.** Per the R6 exit rule, a span-0 (B==A) is NOT a STOP: the mechanism is
  correct and R5 learnability holds. **Decision: KEEP the mechanism; record the HONEST NEGATIVE on the deployed
  gain (deployable conversion gap).**
- **Campaign consequence:** chain 3 (direction supervision) is now fully characterized — signal EXISTS (R4) →
  locally RANKABLE (R5) → NOT deployably CONVERTIBLE (R6). The sweep pre-empts R7's adaptive-trust-region premise
  (optimal anchor-deviation = 0). The campaign is at the honest-negative wall a 4th time (v2-static / D0-D14 /
  POMDP-QP-FAR / Belief-Residual), now with the most precise diagnosis yet: the binding limit is the deployable
  PRECISION of the beneficial-edit direction signal, not its existence, its learnability, the trainer, or the
  gate. Next: R7 as a confirmatory adaptive-gate check (expected == anchor per the sweep) → R8 pilot → R10
  honest close-out; or proceed directly to the close given the decisive sweep.

## Adversarial verification (Workflow) — see below
