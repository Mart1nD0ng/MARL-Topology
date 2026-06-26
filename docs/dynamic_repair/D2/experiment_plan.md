# D2 — experiment_plan

**Stage:** D2 — fix the dynamic reward + unify the discounted episode return (Plan §4; gaps #2, #3).

**Hypothesis:** The dynamic RL objective is currently mis-specified relative to the contract and the
Temporal-Value-Test: the RL reward omits the `hold_interval` factor (`reward = base − reconfig`) and
eval/keep-best use an *undiscounted* episode sum while training optimizes a *discounted* return.
Making the per-step reward `r_t = H·r_base,t − c_reconfig,t` and using ONE discounted episode return
`G_0 = Σ_t γ^t r_t` across training, the Temporal-Value-Test, eval (val/held), and the myopic
reference will (a) make all four the same problem, and (b) is a precondition for trusting any
recurrence/RL conclusion. Per-frame *feasibility* is scale-invariant, so it should not change; the
*return* numbers will (the frozen returns were on the wrong objective).

**Single change (one variable = the dynamic objective definition), applied at its 3 measurement sites:**
- `training/dynamic_rl.py::episode_rollout` — `reward = scene.hold_interval * base_r − reconfig`.
- `training/dynamic_rl.py::dynamic_eval` — discounted, H-scaled episode return
  `ep_ret += γ^t · (H·base_r − reconfig)` (keep-best then uses the discounted return).
- `scripts/diagnostics/dynamic_headline.py::_myopic_reference` — same discounted H-scaled return.
- The Temporal-Value-Test (`two_timescale_env`) already multiplies base by `hold_interval` and
  discounts (`step:104`, `_horizon_optimal_cost`), so the RL/eval sites are being brought to MATCH it.
- Log the objective in `mechanism_activation.json` (`reward_definition`, `reward_uses_hold_interval`,
  `return_definition`).

**Controlled variables:** data sampler, actor/critic architecture, BCSP sampler, decoder, PPO
hyperparams, seeds, splits — all unchanged. Only the reward/return口径 changes. No mechanism added.

**Dataset:** existing `sample_dynamic_scenes` (single-RSU mobility; D1 will replace the geometry later
— out of scope this round). Tests use the tiny controlled `_scene` fixture.

**Seeds / budget:** unit tests only this round + one real-shard `--dynamic` smoke (few updates). A
multi-seed re-run of the headline on the corrected objective is deferred to after D3 (validation
split) so the headline is built on the fully-corrected口径 — D2 alone is not headline-eligible.

**Required failing tests (must fail on HEAD, pass after the fix):**
- `test_dynamic_reward_multiplies_base_by_hold_interval` (H=4): `rec.reward == 4·base − reconfig`.
- `test_temporal_value_and_rl_use_same_reward`: `episode_rollout` reward == `TwoTimescaleTopologyEnv.step`
  reward frame-by-frame (cost_fn = −base) at H>1.
- `test_dynamic_eval_uses_discounted_return`: `dynamic_eval` episode_return == Σ γ^t (H·base − reconfig)
  recomputed from the per-frame traces (γ<1, H>1).
- Update `test_reward_decomposes_into_base_minus_reconfig` → assert the general `H·base − reconfig`.

**Expected mechanism activation:** `dynamic_task.reward_uses_hold_interval = true`,
`reward_definition = "hold_interval*base - reconfig"`, `return_definition` discounted; visible in the
smoke's `mechanism_activation.json`.

**Success criterion:** the 4 tests pass; the full unit suite stays green; the `--dynamic` smoke exits
0 with the new activation fields; per-frame feasibility on the smoke is unchanged vs HEAD at the same
seed (scale-invariance sanity); the return changes (objective corrected).

**Failure criterion:** if multiplying base by H destabilizes the smoke (NaN / all-collapse that was
absent before) — then flag it as an optimization-scale issue (revise: normalize the reward by H or
adjust advantage normalization), do NOT silently revert the objective.
