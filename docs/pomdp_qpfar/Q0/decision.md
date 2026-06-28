# Q0 — decision (freeze the D13 negative as the POMDP-QP-FAR control baseline)

**Result: KEEP / FROZEN.** The D13 campaign result is pinned as the control. No code changed. The
freeze was independently re-derived from the raw `result_save/dynamic_d13_campaign.json` (not just
copied from `decision.md`) and is self-consistent.

## Independent re-derivation from the raw JSON (this round)
- Structure intact: `by_data` = {urban, random}, each grouped `learned_arms` / `deployable_policies` /
  `central_references` — **no group overlap** (Contract §10.1).
- Urban learned **baseline** held_feas mean = **0.19583** (report "0.196" ✓); **PNA** held_feas =
  **0.32083** (report "0.321" ✓, the highest learned).
- **PNA paired vs urban baseline**: mean **+0.125**, CI95 **[−0.2974, +0.5474]**, per-seed
  **[−0.43056, +0.36805, +0.29861, +0.36111, +0.02778]** — the **seed-0 collapse (−0.431)** is real and
  recorded; the CI spans 0 → trend, not a win (exact match to `D13/decision.md`).
- **Chance** is a net loss: held_feas 0.15 < 0.196, held_return −35.93 vs baseline −14.88, λ→4.457 (✓).
- Non-budget-neutral mechanisms self-report: Pareto `pareto_evaluator_calls` mean **144.0**; SCQ
  43.2/update (✓). COMA/PNA budget-neutral (0 extra) (✓).

## Frozen headline (the control, carried forward verbatim)
> On the corrected dynamic pipeline (real 4-RSU urban + single-RSU random, 5 seeds, N∈{8,12,16}, 6
> frames, hold_interval=4, γ=0.95, 30 updates, 25-ep BCSP warm-start), **no** mechanism (temporal /
> COMA / SCQ / chance / Pareto / PNA) gives a statistically significant held-feasibility gain (all six
> paired CIs span 0), and on both data sources **every learned arm < deployable heuristic < central
> myopic reference** (urban: learned 0.196 / best-learned PNA 0.321 < deployable 0.696·0.711 < central
> 0.793; random: learned 0.151 < deployable 0.364·0.369 ≲ central 0.374). Binding limit = **RL
> feasibility-region learning** (Axis B), with the task ~Markov under full CSI (Axis A).

## Q0 exit conditions — all met
1. learned < deployable < central reproducible from the JSON ✓
2. PNA seed-0 collapse recorded ✓
3. Scope fixed = N∈{8,12,16}, 6 frames, 5 seeds, urban+random ✓

## Acceptance table (Contract v3 §15)
- Phase: **Q0 — freeze D13 control baseline**
- Status: **VALIDATED_NEGATIVE (control frozen)**
- Implemented / wired / active: n/a (freeze of an existing, already-validated artifact)
- Test scale: N∈{8,12,16}, 6 frames; Seeds: 5/arm; Dataset: urban 4-RSU + random single-RSU
- Mechanisms active: none new; Mechanisms not tested: the entire POMDP-QP-FAR stack (Q1–Q12, NOT_IMPLEMENTED)
- Positive findings: none (this is the negative control)
- Negative findings: learned < deployable < central; all six mechanism A/B CIs span 0; PNA bimodal
- Conclusion scope: inside the D13 run only; NOT urban-at-scale, NOT N≥24
- Next action: **Q1 — stale/partial CSI observation model** (`csi_observation_model.py`),
  failing-test-first; stale CSI changes ONLY the actor observation, evaluator keeps true current CSI.
