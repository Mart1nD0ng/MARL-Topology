# D2 — decision

**Result: KEEP.** The dynamic objective is now correctly specified and consistent across all four
measurement sites (training, eval=val/held, myopic reference, Temporal-Value-Test).

## What changed (one variable: the dynamic objective definition)
- `training/dynamic_rl.py::episode_rollout` — `reward = scene.hold_interval * base_r − reconfig`
  (was `base_r − reconfig`; `hold_interval` was ignored — gap #2).
- `training/dynamic_rl.py::dynamic_eval` — discounted, H-scaled episode return
  `ep_ret += γ^t · (H·base_r − reconfig)` (was undiscounted `base_r − reconfig` — gap #3). Keep-best
  now uses the discounted return.
- `scripts/diagnostics/dynamic_headline.py::_myopic_reference` — same discounted, H-scaled return.
- `mechanism_activation.json` now records `reward_definition`, `reward_uses_hold_interval=true`,
  `return_definition` (discounted; train==eval==myopic==TVT). Satisfies `test_reconfig_cost_scale_logged`.
- The Temporal-Value-Test (`two_timescale_env`) already H-scaled + discounted — the RL/eval sites were
  brought to MATCH it, so the contract's "train/TVT/val/held same objective" now holds for #2/#3 (the
  validation SPLIT itself is D3).

## Tests
- Failing-first (failed on HEAD `e450cb7`, pass after fix):
  `test_dynamic_reward_multiplies_base_by_hold_interval`, `test_temporal_value_and_rl_use_same_reward`,
  `test_dynamic_eval_uses_discounted_return`, `test_reward_decomposes_into_hold_times_base_minus_reconfig`.
- Unit suite: **654 passed / 0 failed**. Contract suite: **63 passed / 0 failed**.
- Real-shard `--dynamic` smoke (hold_interval=4, recurrent, 3 updates): **exit 0**; activation fields
  present; held feasibility 0.125, return −1.04 (in range, no collapse introduced).

## Cost / runtime
- No new training run beyond the tiny smoke (~seconds) + 2-min unit suite. No evaluator-budget change
  (reward scaling is arithmetic on the existing per-frame evaluator call).

## Honest analysis (scope)
- **Substantive, not cosmetic.** `H` multiplies the base objective but NOT the reconfig cost, so at
  H=4 the per-round objective is weighted 4× relative to switching cost — this changes the
  feasibility-vs-switching trade-off the policy optimizes (even with `--normalize-adv`, which rescales
  but preserves the within-episode relative structure). The critic target `G_t` and all reported
  returns are now on the correct objective; the frozen D4/D6 returns (computed on `base − reconfig`,
  undiscounted) are **superseded** for return comparisons.
- **Per-frame feasibility metric is unchanged in definition** (scale-invariant); but the *trained*
  policy now optimizes the corrected objective, so downstream feasibility may shift — to be measured
  in the post-D3 headline, not claimed here.
- **Not headline-eligible yet.** Per the plan, the multi-seed A/B headline on the corrected objective
  is deferred until after D3 (independent validation split) so keep-best is on a held-out val set, not
  train. D2 alone fixes口径; it does not re-decide recurrence vs memoryless.

## Failure modes watched
- No collapse/NaN introduced by the H scaling in the smoke. If a full multi-seed run shows new
  instability from the 4× objective weighting, the revise path is reward/H normalization — not
  reverting the objective.

## Next (D3)
Add the independent validation split (seed `*1000+333`): build `train`/`val`/`held`, select the
checkpoint by **val** discounted return only, keep held for final reporting. Failing-first tests:
`test_dynamic_train_val_held_seeds_disjoint`, `test_checkpoint_uses_val_not_train`,
`test_held_not_used_until_final`, `test_split_manifest_records_all_three`.
