# D0 — decision

**Result:** D0 complete. The pre-repair dynamic result is frozen and the HEAD-vs-contract/plan gap
analysis is grounded in code.

**Artifacts produced:**
- `docs/CURRENT_DYNAMIC_REPAIR_STATUS.md` — all 10 flagged gaps verified against source (file:line),
  with contract refs + fix-stage mapping. **All 10 confirmed as real gaps.**
- `result_save/dynamic_baseline_frozen/README.md` (tracked) + archived JSONs (local) — frozen D4/D6
  numbers + scope labels + repro commands.
- `docs/dynamic_repair/D0/experiment_plan.md`, this `decision.md`.

**Key code-grounded findings (the load-bearing ones):**
- Gap #1: dynamic data = `_sample_scene`→`_node` = 1 RSU + (N−1) vehicles (random geometry), NOT the
  4-RSU urban grid (`_sample_urban_grid_scene` never called by the dynamic path).
- Gap #2: `dynamic_rl.episode_rollout` reward = `base − reconfig` (no `hold_interval`); the
  Temporal-Value-Test (`two_timescale_env.step`) multiplies base by `hold_interval` → different
  objectives.
- Gap #3: training discounted `G_t`, but `dynamic_eval`/keep-best use undiscounted episode sum.
- Gap #4: only train/held splits; keep-best on train; no validation split.
- Gap #5: `build_pbft_message_plan` not referenced by the production Stage-21 evaluator.
- Gap #7: `graph_payload` has zero motion features.

**Per-seed / CI / evaluator calls / wall-clock:** unchanged from the frozen report (no new run this
round); see `result_save/dynamic_baseline_frozen/README.md`.

**Failure modes recorded:** the frozen negative is a *diagnostic* mixing (a) a single seed-0
train→held drop measured under train-keep-best (so not a clean generalization gap per forbidden §13.7)
and (b) two cold-start optimization collapses.

**Scope:** D0 changes no code and revises no result. It only establishes the baseline + the gap map.

**Decision: KEEP.** Proceed to **D2** next (reward `hold_interval` + single discounted objective
across train/TVT/val/held) — the highest-priority code fix per Plan §17, and the prerequisite for any
honest recurrence/RL conclusion. (D1 urban data and D4 phase-accounting proceed in parallel but are
not on the critical path until the final headline.)

**Next hypothesis (D2):** "Making the dynamic reward `H·base − reconfig` and using ONE discounted
episode return for training, the Temporal Value Test, validation, and held eval will (a) make the TVT
and the RL objective the same problem, and (b) not, by itself, change the recurrence≈memoryless
verdict — but it is a precondition for trusting that verdict." First write failing tests:
`test_dynamic_reward_multiplies_base_by_hold_interval`, `test_temporal_value_and_rl_use_same_reward`,
`test_dynamic_eval_uses_discounted_return`, `test_keep_best_uses_validation_discounted_return`.
