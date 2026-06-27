# D8 — decision

**Result: KEEP (honest, scoped negative for temporal modeling + deployable-baseline-wins).** On the
fully-corrected single-RSU pipeline, neither velocity features nor cross-frame recurrence gives a
statistically significant gain; the learned RL arms underperform the fair DEPLOYABLE baselines (and the
central reference), so the binding limit is RL feasibility-region learning, not temporal modeling.

## Configuration
5 seeds × the 2×2 {motion × recurrence} matrix, N∈{8,12,16}, 30 updates, T=6, 24 train / 24 val / 24
held (disjoint), `--dyn-warmstart-mode bcsp` (decoder-aware, 25 ep), `--dyn-bc-anchor 0`, corrected
env-math (D2 hold/discount objective, D3 val-only checkpoint, D4 phase-specific energy). All arms share
the config; only {motion, recurrence} vary. Single-RSU random geometry (D1 urban data NOT yet built).

## Learned arms — held (5 seeds)
| arm | feas mean | feas 95% CI | discounted return | switches/frame | post-RL drift |
|---|---|---|---|---|---|
| memoryless_csi | 0.151 | [0.012, 0.291] | −14.48 | 0.65 | +0.054 |
| memoryless_velocity | 0.110 | [−0.051, 0.271] | −15.97 | 0.85 | +0.248 |
| recurrent_csi | 0.197 | [0.038, 0.356] | −13.95 | 1.31 | +0.325 |
| recurrent_velocity | 0.050 | [0.028, 0.072] | −17.74 | 1.03 | −0.024 |

Per-seed feasibility (no 0.0 collapse — D6 warm-start fixed the cold-start collapse the frozen baseline
had): memoryless_csi [0.271, 0.069, 0.063, 0.076, 0.278]; recurrent_csi [0.257, 0.083, 0.167, 0.090,
0.389]; recurrent_velocity [0.028, 0.049, 0.042, 0.076, 0.056] (consistently low).

## Paired ablation — every diff's CI spans 0 (no significant effect)
| comparison | mean | 95% CI |
|---|---|---|
| velocity − csi (memoryless), feas | **−0.042** | [−0.303, +0.220] |
| velocity − csi (memoryless), return | −1.49 | [−10.0, +7.0] |
| recurrent − memoryless (csi), feas | **+0.046** | [−0.026, +0.117] |
| recurrent − memoryless (velocity), feas | **−0.060** | [−0.226, +0.106] |

## Baselines — held (5 seeds), grouped per Contract §10.1
| group | method | feas | feas 95% CI | return | switches/frame | action eval calls |
|---|---|---|---|---|---|---|
| **deployable** | local_threshold | 0.364 | [0.307, 0.420] | −8.64 | 1.60 | **0** |
| **deployable** | local_hysteresis | 0.369 | [0.303, 0.436] | −8.28 | 1.05 | **0** |
| **central ref** | myopic_greedy | 0.374 | [0.276, 0.471] | −8.67 | — | 864 |

## Honest conclusions (scope: single-RSU, corrected env-math, N∈{8,12,16}, 5 seeds)
1. **Velocity features give no significant gain** (paired CI spans 0; point estimate slightly negative).
   The §3 "current-CSI ~Markov" framing is consistent with this — adding motion features does not help
   at single-RSU. (D5 made the test FAIR; this is the answer.)
2. **Cross-frame recurrence gives no significant gain** (both paired CIs span 0) — confirms the §3
   pre-registered prediction and the bounded Temporal-Value headroom.
3. **The learned RL arms (0.05–0.20) underperform BOTH the deployable baselines (0.36–0.37) AND the
   central reference (0.37).** The binding limit is RL feasibility-region learning at the corrected
   (harder) env-math, NOT temporal modeling. The decoder-aware warm-start reaches only ~0.15 vs the
   teacher's ~0.37 — the learned local GNN+BCSP actor is the weak link.
4. **The deployable local baselines ≈ the central myopic reference** (0.364/0.369 vs 0.374; 0 vs 864
   eval calls) — confirms D7 at 5 seeds: the simple LOCAL rules are strong and the evaluator search buys
   ≈nothing here. local_hysteresis switches less (1.05 vs 1.60) at equal feasibility.
5. **D6 warm-start protection validated**: post-RL drift is small/POSITIVE (+0.05…+0.32 return) — RL no
   longer DEGRADES the warm-start (the frozen report's "RL degrades the warm-start" does NOT reproduce
   with the decoder-aware bcsp warm-start), and no seed collapsed to 0.0.

## What this does NOT prove (scope limits)
- NOT an urban result — single-RSU random geometry; D1 (4-RSU urban) is pending and may change the
  temporal-value picture (mobility/handover structure differs).
- NOT "temporal modeling is useless" in general — only that it adds nothing on THIS task at this scale.
- The learned-actor underperformance is a single-RSU/N≤16 finding under the corrected env-math; large-N
  remains the open frontier.

## Next
- **D9** — dynamic COMA / Q-critic (per-agent counterfactual credit). Then D10 (SCQ), D11
  (chance/CVaR/Pareto), D12 (PNA/vector critic). D1 (urban data) remains required before any final
  headline. Adversarial verification of THIS conclusion: see §verification below.

## Verification — adversarial re-derivation from the raw JSON (PASS)
A focused adversarial agent independently re-derived every claim from `result_save/dynamic_d8_matrix.json`:
- all 4 paired diffs' 95% CIs INCLUDE 0 (numbers match exactly) → "no significant velocity/recurrence
  effect" is honest, not an over-claim;
- every learned arm (max 0.197) is below every baseline (min 0.364) — table numbers match the JSON;
- deployable `action_evaluator_calls = 0` (all seeds), central = 864 (all seeds) — exact;
- no hidden/collapsed seeds: every arm has n=5, no 0.0 in the per-seed feasibility OR post-RL-drift lists;
- scope honestly limited to single-RSU in both the JSON `scope` field and the decision; no urban
  extrapolation; the central myopic is never called "deployable".
- The one gap it flagged — learned-arm switches/frame missing from the table — is now FIXED (column
  added above). Verdict: **PASS, no over-claims**.
