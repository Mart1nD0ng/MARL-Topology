# D13 — decision (the full dynamic campaign)

**Result: KEEP the corrected pipeline + the baseline as the default; all D9–D12 mechanisms stay OPT-IN
(no headline gain). HONEST NEGATIVE, now on genuinely 4-RSU urban data.** Adversarial re-derivation from
the raw per-seed JSON (4 independent lenses) = **PASS, no refutation**.

## Headline (the single honest sentence)
> On the corrected dynamic pipeline with genuinely 4-RSU urban data (5 seeds, N∈{8,12,16}, 6 frames),
> no mechanism — temporal / COMA / SCQ / chance / Pareto / PNA — produces a statistically significant
> held-feasibility gain (all paired CIs span 0), and on both urban and random every learned arm falls
> below the zero-eval-call deployable heuristics — which trail the central myopic reference — so the
> binding limit is **RL feasibility-region learning**, not temporal structure, credit assignment,
> reliability shaping, or actor architecture.

## The campaign (5 seeds, N∈{8,12,16}, 30 updates, 25-ep bcsp warm-start; `result_save/dynamic_d13_campaign.json`)
Held per-frame feasibility (mean; per-seed + CI in the JSON):

| group | random | urban (real 4-RSU) |
|---|---|---|
| **learned** baseline (mlp+memoryless) | 0.151 | 0.196 |
| learned best mechanism | — | **pna 0.321** (largest, but bimodal) |
| **deployable** local_threshold / hysteresis (0 eval calls) | 0.364 / 0.369 | 0.696 / **0.711** |
| **central** myopic reference (864 eval calls) | 0.374 | **0.793** |

**Per-seed cherry-picked check (the strong form):** the best-of-7 learned arm beats the best deployable
on *no* urban seed (learned 0.56/0.44/0.41/0.50/0.15 vs deployable 0.76/0.57/0.67/0.81/0.75) and no random
seed. Learned RL is dominated by the simple local heuristics on both data sources.

## Mechanism A/Bs (paired vs the urban baseline, df=4, t.975=2.776 — all six CIs span 0)
| arm | the one variable | paired Δfeas | CI95 | budget |
|---|---|---|---|---|
| recurrent_velocity | `--motion-features` + recurrent | −0.092 | [−0.385, +0.202] | neutral |
| coma | `--counterfactual` | −0.039 | [−0.117, +0.039] | **budget-neutral** (0 extra) |
| scq | `--scq` | −0.008 | [−0.059, +0.043] | **NOT neutral: 43.2 calls/update** [12,15,101,6,82] |
| chance | `--chance` | −0.046 | [−0.127, +0.035] | neutral; **but λ→4.46, return −35.9 vs −14.9, drift −20.2** |
| pareto | `--pareto-archive` | +0.018 | [−0.013, +0.049] | **NOT neutral: 144 held eval calls** |
| pna | `--dynamic-actor-arch pna` | +0.125 | [−0.297, +0.547] | budget-neutral (0 extra) |

- **PNA** has the largest positive mean and the highest absolute feasibility (0.321), but its CI spans 0
  because of a **bimodal seed-0 collapse** (−0.431 vs +0.30/+0.37 on seeds 1–3). A *trend*, not a win;
  the honest call is more seeds, not a headline claim.
- **Chance** is reported as a net loss: it does NOT improve feasibility (0.150 < 0.196) and badly hurts
  return (the dual climbs and the penalty dominates) — the "no gain" is, if anything, understated.
- Every arm's distinguishing flag is genuinely active (single-variable, no silent no-op; confirmed in
  each `mechanism_activation.json`).

## Adversarial re-derivation (workflow `wfqnd72do`, 4 lenses, all CONFIRM, PASS)
1. **Data faithfulness**: all 35 urban manifests are genuinely 4-RSU / 9-building / 8-road / road-constrained
   with 5 distinct seed-keyed content hashes; **independently regenerated seed-0's hash byte-identically
   from source** and confirmed `NodeKind.RSU==4` in the actual scenes. Random arm correctly carries no
   manifest (single-RSU path). No relabeling, no over-claim (Contract §4.1/§4.2).
2. **A/B nullity**: every paired CI recomputed from raw per-seed JSON — exact match to the report; all six
   span 0; PNA's width traced to the seed-0 collapse.
3. **Central result**: learned < deployable (< / ≈ central) on every seed; deployables independently
   recomputed live (0 eval calls); central 864 calls; no learned/deployable/central grouping overlap.
4. **Budget + honesty**: SCQ/Pareto flag `budget_neutral=false` and report their extra calls; COMA/PNA
   genuinely budget-neutral; no dropped/hidden seed (all 40 runs, 5 seeds/arm present).

## Caveats recorded (none block PASS)
1. **"deployable ≈ central" is tight on random (Δ<0.01) but loose on urban (0.711 vs 0.793, ~0.08 gap).**
   State it as **learned < deployable < central (urban)** / learned < deployable ≈ central (random) — the
   central reference is the true ceiling on urban, the deployable sits below it.
2. **Scope:** single-RSU random vs 4-RSU urban grid, N∈{8,12,16}, 6 frames, hold_interval=4, γ=0.95,
   30 updates, 25-ep warm-start, 5 seeds. NOT urban-at-scale / NOT N≥24. Urban NLOS is harder than random;
   conclusions stay inside the run.
3. `pareto_evaluator_calls_held` lives in `mechanism_activation.json` (reliability block) — file placement,
   not hiding; it IS surfaced in the campaign JSON's pareto budget.

## Decision
- **Default production arm = the corrected baseline** (graph-MAPPO spine + MLP memoryless actor + bcsp
  warm-start + torch-free local decoder). All D9–D12 mechanisms remain **opt-in** (verified-correct, no
  headline gain at this scale) — consistent with the v2 static campaign's pattern (Phase 8–13).
- **Open frontier (deferred, owner's call):** RL feasibility-region learning at N≤16 is the bottleneck
  (learned ~0.2–0.32 vs deployable ~0.37 random / ~0.71 urban). PNA's positive-but-bimodal trend is the
  one lead worth more seeds. Large-N (N≥24) still needs the cheaper exact-fault evaluator (v2 open item).
- Next: **D14** — the docs/report/README reconciliation (research-log SUMMARY + mechanism ledger;
  CURRENT_HEAD_STATUS; correct the DYNAMIC_TASK_REPORT 4-RSU / "full Phase-8–11" claims to match what is
  now actually tested; AGENTS/README; remove any critic-free-identity residue). This is the final stage.
