# Q7 — decision (D_quorum-guided add-only repair)

**Result: KEEP. HONEST PARTIAL POSITIVE.** D_quorum-guided TARGETED add repair beats the anchor on its
own failure scenes — it fully repairs ~22% of random anchor-failures (vs 0% for the anchor AND 0% for a
naive max-budget topology), improves true C by +0.36 on every failure, and retains 100% of the anchor.
But the repair is PARTIAL (most deeply-infeasible failures not fully fixed; urban's few NLOS failures not
helped by single-adds). The DEPLOYABLE learned repair head is deferred to Q9 (the central-reference
ceiling is already limited; the full deployable residual policy trains end-to-end there).

## What changed (mechanism + central reference; not in the deployed actor yet)
`src/marl_topology/training/residual_repair.py`:
- `greedy_dquorum_add_repair` — CENTRAL REFERENCE (training-only): greedily ADD the non-anchor incident
  edge that most reduces D_quorum (budget + mutual), until C ≥ τ or no candidate helps. Uses the
  evaluator per candidate → NOT deployable. The ceiling: can the Q4-aligned proxy guide repair?
- `repair_head_targets` — the per-edge supervised target `r_e = D(x) − D(x⊕e)` for a deployable repair
  head (computed via the bridge → training-only).
- `scripts/diagnostics/add_only_repair.py` — runs the repair on anchor-FAILURE frames; final metric =
  true C; reports the anchor vs the (central) repair separately (Contract §10.1).

## Result (pilot: 8 scenes × 6 frames; anchor-FAILURE frames only; final metric = true PBFT C)
| data | anchor failures | naive max-add feasible | **D_quorum repaired (full τ=0.9)** | mean C improvement | mean D reduction | retention | central evals/repair |
|---|---|---|---|---|---|---|---|
| random | 27 | **0** | **6 (22%)** | **+0.357** | 1.250 | **1.0** | 96 |
| urban | 3 | 0 | 0 (0%) | 0.0 | 0.0 | 1.0 | 30 |

(consistent with a 10-scene run: 7/37 = 19% random, +0.33 C.)

## The findings (honest)
1. **Guidance matters**: the targeted D_quorum repair fully fixes 6 random failures that a NAIVE
   max-budget topology (every node at full budget) fixes **0** of — adding the RIGHT edges beats adding
   the MOST. The Q4 proxy genuinely points the repair toward feasibility.
2. **Large C improvement on every failure** (+0.357 mean) with **100% anchor retention** (add-only by
   construction) → the exit condition "repair > anchor" is MET on random (improves C and crosses τ on
   22% vs the anchor's 0%).
3. **Partial / honest negatives**: 78% of random failures and ALL urban failures are NOT fully repaired.
   These are deeply infeasible (the strong anchor already picked the good edges; the residual failures
   are hard), and the greedy is myopic (stops when no single add reduces D — a 2-edge repair is missed).
   Consistent with Q4: D_quorum aligns CONDITIONALLY (strong where C moves; the plateau dominates).

## Honesty / scope
- Single-seed pilot; the 22% repair rate is a DIAGNOSTIC ceiling, not a multi-seed headline.
- The greedy is the CENTRAL REFERENCE (uses ~96 evaluator calls/repair) — NOT a deployable result. The
  DEPLOYABLE learned repair head (predicting r_e from local features, 0-eval at deploy) is **deferred to
  Q9**, where the full residual policy is trained end-to-end with PBRS (the central ceiling here bounds
  any deployable head, so building it standalone now is low-value).
- The final metric is ALWAYS the true closed-form PBFT C; D_quorum is the Q4-authorized auxiliary guide.

## Acceptance table (Contract v3 §15)
- Phase: **Q7 — add-only repair**
- Status: **VALIDATED_POSITIVE (random, central reference)** + **NEGATIVE_BUT_SCOPE_LIMITED (urban; deployable head deferred)**.
- Implemented ✓ (greedy repair + head target) / Wired into trunk ✗ (deployable head is Q9) / In this run ✓
  (central greedy on failure frames)
- Test scale: 4 unit + suite 749/0; pilot single-seed 8 scenes (27 random + 3 urban failures)
- Mechanisms active: D_quorum-guided add repair (central reference). Not tested: the DEPLOYABLE learned head (Q9).
- Positive: guidance matters (22% vs 0% naive); +0.36 C improvement; 100% retention.
- Negative: partial repair (78% random / 100% urban failures unfixed); central, not deployable; myopic greedy.
- Conclusion scope: D_quorum guides add-only repair on the subset of anchor-failures where single-edge
  adds help (random); it does NOT fully repair deeply-infeasible failures (urban). Central-reference
  ceiling only — NOT a deployable headline.
- Next action: **Q8 — conservative prune**: from FEASIBLE anchors, remove low-risk edges (safety head
  `risk_e = D(x∖e) − D(x)`), keep feasibility, lower energy/latency; report retention + E/L reduction +
  critical-edge deletion rate.
