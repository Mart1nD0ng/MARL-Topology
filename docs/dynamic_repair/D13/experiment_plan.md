# D13 — experiment plan (the full dynamic campaign)

**Goal:** the headline campaign on the corrected pipeline — the urban-vs-random contrast + the D9–D12
mechanism A/Bs — per-seed + CI + grouped baselines + per-arm budget, with honest scope.

## Driver: `scripts/diagnostics/dynamic_d13_campaign.py`
Each `(arm, seed)` is a `--dynamic` subprocess; the D7 deployable baselines + central reference are
evaluated per data source per seed. Single-variable arms (vs the urban baseline):

| arm | data | actor | arch | extra flag (the one variable) |
|---|---|---|---|---|
| baseline | urban | memoryless | mlp | — |
| baseline | random | memoryless | mlp | — (the urban-vs-random contrast point) |
| recurrent_velocity | urban | recurrent | mlp | `--motion-features` (the D8-best temporal) |
| coma | urban | memoryless | mlp | `--counterfactual --k-cf 4` |
| scq | urban | memoryless | mlp | `--counterfactual --k-cf 4 --scq …` |
| chance | urban | memoryless | mlp | `--chance --chance-delta 0.1 --chance-lr 0.2` |
| pareto | urban | memoryless | mlp | `--pareto-archive` |
| pna | urban | recurrent | pna | (arch swap) |

All arms share the recommended config: cold-start + decoder-aware **bcsp warm-start (25 ep)** +
`--normalize-adv`, dense reward, N∈{8,12,16}, 30 updates, 24/24/24 train/val/held, 6 frames.

## Headline run (background, ID `b8w4s172m`)
`--seeds 0 1 2 3 4 --updates 30 --dyn-train/val/held 24 --frames 6 --dyn-warmstart 25 --dyn-nodes 8 12 16`.
8 arms × 5 seeds = 40 subprocess runs + per-data baselines. → `result_save/dynamic_d13_campaign.json`.

## Report (grouped, Contract §10.1)
Per data source: `learned_arms` (per-seed → CI on held feasibility / return / switches / cvar +
budget: scq/pareto evaluator calls), `deployable_policies`, `central_references` — never mixed. Plus the
paired (mechanism − urban-baseline) feasibility diffs and the urban-vs-random baseline contrast.

## Tests / pilot (done)
- `tests/unit/test_d13_campaign.py` (3): arm spec covers urban+random + single-variable mechanisms; `_ci`
  reports per-seed + interval; `build_report` groups per-data without cross-data leak. **3/3 pass.**
- Tiny pilot (urban+random+coma, 1 seed, smoke params) exit 0; report grouping verified (central 36 eval
  calls separated from deployables' 0; paired diff present). **NOT a headline** (smoke params).

## Honest scope (binding)
Urban NLOS is genuinely harder than single-RSU random (the pilot already shows urban learned feas low vs
the deployable heuristics) — conclusions stay within the ≥5-seed headline run; failed seeds reported;
default-off mechanisms NOT written as "full model tested"; non-budget-neutral arms (scq/pareto) report
their extra evaluator calls.

## Status
Infrastructure built + pilot-validated + committed; **headline run in background**. The result analysis,
adversarial re-derivation from raw JSON, and `decision.md` are the next iteration (when the run completes).
