# R8 — decision (stale-CSI premise test) — HONEST NEGATIVE under the REAL regime; premise CONFIRMED

**Result: the campaign's PREMISE is confirmed (stale CSI significantly degrades the anchor) but the method does
NOT repair the drop at ANY operating point — a HONEST NEGATIVE in the campaign's actual failure regime.** Under
delay-1 stale CSI the deployable anchor drops significantly (random −0.055, urban −0.165; both CI > 0 — the
urban drop matches Q14's 0.80→0.66). The evidence-gated action (heads RETRAINED on stale features, labels from
the true evaluator) reproduces the stale anchor at the neutral threshold (urban edit 0 → == stale_anchor; random
+0.010 spans 0) and, over the full threshold sweep, NO tau_edit gives a (gated − stale_anchor) feasibility CI
lower bound > 0 — the same downhill-from-empty-gate pattern as R6's current channel. **The stale-degraded
anchor's "room" does NOT rescue the method: the deployable direction signal cannot convert into a repair** (R8
measures the DEPLOYED outcome — no repair at any threshold — not the stale-trained heads' top-k precision
directly; the current-channel R5 head precision was ~40%, and R8's deployed no-repair is consistent with an
equal-or-lower precision on stale features, inferred not re-measured). The 1-seed pilot's +0.0625 was noise.

## 5-seed A/B (`stale_csi_metrics.json`; tau_edit 0.5; 0 eval at decision; true evaluator scores)
| metric | random | urban |
|---|---|---|
| current_anchor_feas (ceiling) | 0.375 | 0.815 |
| stale_anchor_feas (degraded) | 0.320 | 0.650 |
| stale_gated_feas | 0.330 | 0.650 |
| **stale_drop = current − stale anchor** | **0.055 [+0.009, +0.101]** ✅ CI>0 | **0.165 [+0.118, +0.212]** ✅ CI>0 |
| **gated − stale_anchor feasibility** | +0.010 **[−0.007, +0.027]** | **0.000 [0.0, 0.0]** |
| gated − stale_anchor return | −0.003 [−0.013, +0.006] | 0.000 [0.0, 0.0] |
| stale_gated edit_rate / unsafe | 0.003 / 0.0 | 0.0 / 0.0 |

**Premise CONFIRMED:** the stale drop is real and significant on BOTH regimes (urban −0.165 matches the Q14
0.80→0.66 that motivated the whole campaign). **No repair at tau 0.5:** urban suppresses all edits (edit 0 →
== stale_anchor); random fires a tiny bit (+0.010) but spans 0.

## Threshold sweep (`stale_threshold_sweep.json`; reuse the per-seed stale-trained actor, vary tau_edit) — DECISIVE
(gated − stale_anchor) feasibility (mean) as the gate is loosened under STALE CSI:

| tau_edit | random | random edit / unsafe | urban | urban edit / unsafe |
|---|---|---|---|---|
| 0.5 | +0.010 [−0.007, +0.027] | 0.003 / 0.00 | 0.000 [0,0] | 0.0 / 0.00 |
| 0.35 | −0.015 | 0.004 / 0.01 | −0.015 | 0.008 / 0.02 |
| 0.25 | −0.040 | 0.011 / 0.025 | +0.005 [−0.081, +0.091] | 0.027 / 0.025 |
| 0.15 | −0.070 | 0.021 / 0.03 | −0.005 | 0.036 / 0.035 |
| 0.05 | −0.060 | 0.027 / 0.025 | −0.015 | 0.038 / 0.045 |

**No tau's (gated − stale_anchor) feasibility lower bound exceeds 0** — the best point is the empty gate
(≈ stale_anchor); firing edits is net-negative in mean (random clearly downhill −0.015→−0.070) with rising
unsafe (urban 0.00→0.045). Return is net-negative at every firing threshold. The "room" hypothesis (the
degraded stale anchor gives the gate room to help) is REFUTED — the same precision limit as R6, now in the
STALE regime. (This is the R6 result reproduced under the campaign's actual failure regime.)

## Why (mechanism-level diagnosis — Contract v4 §13)
Same DATA/precision limit as R6/R7, now confirmed in the STALE regime:
- The stale overlay is leak-free and correct (test: CSI cols diverge at t≥1; the evaluator stays on the true
  channel), and it genuinely degrades the anchor (premise confirmed, CI>0). So there IS room to repair.
- The DEPLOYED outcome shows the heads' stale-feature predictions do not convert: at the neutral threshold the
  gate suppresses (== stale_anchor) and at lower thresholds the extra edits are net-harmful (unsafe rises,
  feasibility/return fall). (R8 did NOT re-measure the stale-trained heads' held top-k precision; the failure is
  established from the deployed A/B + sweep, and is consistent with an equal-or-lower precision than R5's ~40%
  current-channel figure — inferred, not measured.) The stale-degraded anchor's room is not enough: the
  deployable direction signal does not convert into a repair on EITHER channel.
- So the binding limit — the deployable PRECISION of the beneficial-edit direction signal — holds in BOTH the
  current-channel (R6/R7) and the stale-channel (R8) regimes. The method does not beat the deployable anchor.

## Effect-on-Decision (Contract v4 §3)
Under stale CSI the gate DOES change the final topology when loosened (urban edit_rate up to 0.038, random up to
0.027) — but the change is net-harmful (feasibility/return down, unsafe up). At tau 0.5 it suppresses (urban ==
stale_anchor). Load-bearing on the action; the deployed effect is null-or-negative vs the stale anchor.

## Path-specific scope (Contract v4 §5)
> "Under delay-1 stale CSI (the campaign's actual failure regime, which R1–R7 ran WITHOUT), the deployable
> anchor drops significantly (random −0.055, urban −0.165 CI>0; premise confirmed), but the evidence-gated
> action (heads retrained on stale features, 0 eval) does NOT repair it: no tau_edit ∈ [0.05,0.5] gives a
> (gated − stale_anchor) feasibility CI lower bound > 0, on urban or random, N≤16; the best point is the empty
> gate (≈ stale_anchor), firing edits is net-negative in mean with rising unsafe. The stale-degraded anchor's
> 'room' does not rescue the method — the deployable direction-signal precision limit (R6) holds in the stale
> regime too. NOT a leak / mechanism / trainer failure; a DATA/precision limit."

## Tests (failing-first → pass)
2 in `tests/unit/test_belief_residual_R8_stale_csi.py`: the delay-1 overlay changes the observed CSI cols at
t≥1 (ef dim +2 [age, mask]) AND the evaluator is identical stale-vs-current (leak-free: actor stale, score
true); the deployed decode is the same 0-eval `evidence_gated_residual`. Full suite 817/0.

## Acceptance (Contract v4 §15) / Decision
- (gated − stale_anchor) feasibility CI > 0 on a regime → REPAIR (campaign's first positive). **NOT observed**
  (random +0.010 spans 0; urban 0.000; no tau in the sweep has lo>0).
- CI spans 0 → does not significantly repair (negative under the real regime). **Observed** (both regimes).
- CI < 0 → makes it worse → REVISE. **Not the neutral threshold**, but firing (lower tau) IS net-negative.
- **Decision: HONEST NEGATIVE under the real regime.** The premise is confirmed (significant stale drop); the
  method does not repair it at any operating point. This COMPLETES the campaign — no deployable arm beats the
  anchor on the current channel (R6/R7) OR the stale channel (R8). → R10 honest close-out (no R9 positive to
  consolidate).

## Adversarial verification (Workflow) — see below
