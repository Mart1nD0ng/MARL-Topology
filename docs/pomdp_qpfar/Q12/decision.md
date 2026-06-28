# Q12 — decision (the consolidated POMDP-QP-FAR multi-seed campaign)

**Result: KEEP. The consolidated HONEST headline.** Across 5 seeds × {urban, random} × N∈{8,12,16}, the
POMDP-QP-FAR DEPLOYABLE tier (the `local_hysteresis` anchor, and — equal to it per Q9/Q11 — full-residual
± PBRS ± PNA) sits at urban **0.739** / random **0.281** per-frame feasibility with **0 evaluator calls**,
BELOW the central myopic-greedy oracle (urban **0.800** / random **0.308**, 432 eval calls/seed) — but by a
MODEST margin with overlapping CIs. **No learned arm beats the deployable anchor; the central oracle is the
ceiling.** This confirms the v2/D13 pattern across the full POMDP-QP-FAR mechanism stack — the binding limit
is feasibility-region learning. Adversarial verification: multi-lens Workflow `ww7m0vggu` (verdict below).

## What changed (consolidation driver; no new mechanism)
`scripts/diagnostics/pomdp_qpfar_campaign.py` — runs the rollout arms across 5 seeds × {urban, random},
per-seed + 95% CI + evaluator-call budget, grouped DEPLOYABLE vs CENTRAL (Contract §10.1; a tested
`group_arms` asserts the deployable arms make 0 eval calls). The residual arms are CITED from Q9/Q11
(established 5-seed paired residual−anchor diff = 0.000 → they equal the anchor), not re-trained.

## Result (5 seeds × {urban, random} × held N∈{8,12,16}; final metrics = true PBFT C/E/L)
| group | arm | data | feasibility (mean, CI95) | episode return | switches/frame | eval calls/seed |
|---|---|---|---|---|---|---|
| **deployable** | local_hysteresis (anchor) | random | **0.281** [0.154, 0.407] | −9.82 | 1.18 | **0** |
| deployable | local_threshold | random | 0.289 [0.167, 0.411] | −10.02 | 1.82 | 0 |
| **central** | myopic_greedy (oracle) | random | **0.308** [0.177, 0.439] | −9.85 | 1.35 | 432 |
| **deployable** | local_hysteresis (anchor) | urban | **0.739** [0.657, 0.821] | −2.02 | 1.88 | **0** |
| deployable | local_threshold | urban | 0.725 [0.613, 0.837] | −2.57 | 2.88 | 0 |
| **central** | myopic_greedy (oracle) | urban | **0.800** [0.715, 0.885] | −2.01 | 2.75 | 432 |

**Cited (the full deployable tier, from the prior stages — all at N≤16):**
- full-residual MLP / +PBRS / +PNA == anchor — the **5-seed** exact-null is from **Q11** (paired
  residual−anchor diff **0.000**, CI [0,0], retention 1.0, 0/5 collapse, both data); **Q9 PART 2** supplies
  the PBRS mechanism + the single-seed pilot. → the learned deployable arms equal the anchor row above (by
  construction — the trust-region clamps the residual policy to the anchor; an exact null, not a learned win).
- CENTRAL add-repair (Q7): D_quorum-guided fully repairs ~22% of random anchor-failures (vs 0% naive),
  +0.36 mean C, 100% retention, ~96 eval calls/repair. CENTRAL prune (Q8): SAFE everywhere (0 critical
  deletions), urban energy down on 47% of feasible anchors.

## The consolidated headline (honest)
1. **The deployable POMDP-QP-FAR tier = the anchor** — the anchor, full-residual, +PBRS, and +PNA all sit
   at one feasibility (urban ~0.739, random ~0.281). No mechanism (stale CSI / recurrence / D_quorum-PBRS
   / residual repair / prune / PNA) lifts the DEPLOYABLE learned policy above the `local_hysteresis` anchor.
2. **The central oracle is a MODEST, NOT-SIGNIFICANT ceiling** — myopic-greedy beats the deployable tier
   by ~+0.06 (urban) / ~+0.03 (random) IN MEAN, but the **paired (myopic−anchor) 95% CI SPANS 0** (urban
   +0.061 [−0.019, +0.141]; random +0.028 [−0.031, +0.086]) and on **1/5 seeds the anchor BEATS myopic on
   both data** — so "ceiling" is in-mean / 4-of-5-seeds, NOT strict per-seed dominance or a significant
   separation. The oracle's small edge costs 432 evaluator calls/seed (NOT deployable). The central
   D_quorum-repair / prune help on the hard sub-cases (Q7/Q8) but are also central.
3. **Binding limit = feasibility-region learning** — the learned residual policy is clamped to the anchor
   manifold; the proxy machinery (D_quorum potential, PBRS) is optimum-preserving and correct but cannot
   make RL discover a topology the strong deployable heuristic misses, at N≤16.

## Honesty / scope
- 5 seeds × {urban, random} × held N∈{8,12,16}, 6 frames; the deployable-vs-central gap is modest with
  overlapping CIs — NOT a large separation, NOT a learned win, NOT a learned collapse. Stated as-is.
- The residual arms are CITED (Q9/Q11, established 5-seed) not re-trained — honest (the result is fixed at
  0.000 paired diff). The central references are correctly grouped (432 eval calls, never deployable).
- NOT extrapolated beyond N≤16; N≥24 remains the open frontier (cheaper exact-fault evaluator needed).

## Acceptance table (Contract v3 §15)
- Phase: **Q12 — consolidated multi-seed campaign**
- Status: **VALIDATED_NEGATIVE (learned == deployable anchor < central oracle, modest)** — the campaign headline.
- Implemented ✓ / Wired (campaign driver) ✓ / Active in this run ✓ (5 seeds × {urban, random})
- Test scale: 2 unit + suite 772/0; 5 seeds × {urban, random} × N{8,12,16} (deployable + central run; residual cited)
- Positive: clean grouped headline; deployable arms 0 eval calls; central budget reported; CIs honest.
- Negative: no learned arm beats the deployable anchor; the binding limit is feasibility-region learning.
- Conclusion scope: at N≤16, the full POMDP-QP-FAR deployable stack = the anchor < the central oracle
  (modest). NOT a claim at N≥24.
- Next action: **Q13 — docs / report / README close-out** (research-log SUMMARY + mechanism ledger;
  CURRENT_HEAD_STATUS; README/AGENTS alignment; claims-vs-evidence verify). Then the campaign is complete.

## Adversarial verification (multi-lens Workflow `ww7m0vggu`, 4 lenses) — overall MINOR, no blocker/major
- **GROUPING + EVAL-CALLS — PASS.** `eval_calls==0 IFF group==deployable` holds for all 6 arm/data cells;
  the deployable action path (local_threshold / local_hysteresis) reads ONLY local ef[psucc]+budgets+prev,
  NEVER the evaluator (traced, not trusted); central myopic = 432 counted action-time calls/seed
  (12 scenes × 6 frames × 6 candidates). The shared per-frame scoring metric (72/seed) is correctly
  excluded from the action-call budget for both. (Nit fixed: `group_arms` now called in `main()`.)
- **CI + DEPLOYABLE-VS-CENTRAL — PASS.** Every mean/CI reproduces from the raw per-seed data to <1e-4.
  Paired (myopic−anchor): urban [+0.014,+0.111,+0.097,−0.028,+0.111] mean +0.061 CI[−0.019,+0.141]; random
  [+0.056,+0.083,−0.042,+0.014,+0.028] mean +0.028 CI[−0.031,+0.086] — **both span 0** (4/5 seeds positive).
- **RESIDUAL==ANCHOR CITATION — PASS.** Q11's 5-seed paired diff exactly 0.000 (CI [0,0], retention 1.0,
  adversarially verified in wslezjbo1) genuinely establishes residual == anchor; citing rather than
  re-training is honest (re-running reproduces the anchor by construction). 5-seed attribute = Q11; Q9
  PART 2 = mechanism + single-seed pilot (clarified above).
- **NO OVER-CLAIM / SCOPE — MINOR.** None of the prohibited statements is made. MINOR wording guards
  (applied above): "ceiling" is in-mean / 4-of-5-seeds (on 1/5 seeds the anchor beats myopic on both data),
  NOT strict per-seed dominance; the gap is not statistically significant (paired CI spans 0); scope N≤16.

**Verdict: KEEP — the consolidated honest headline stands; the MINOR framing guards are applied.**
