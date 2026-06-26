# D3 — decision

**Result: KEEP.** An independent validation split is now wired into the dynamic arm; the checkpoint is
selected ONLY on the validation discounted return, and held is strictly final-reporting-only.

## What changed (one variable: the data split used for checkpoint selection)
- `train_decentralized_rl.py`: new `--dyn-val` arg (default 24; seed `*1000+333`); `parse_args(argv=None)`
  (testability, byte-identical for CLI); smoke sets `dyn_val=4`.
- `dynamic_frames.sample_dynamic_scenes`: new `tag` param → per-split id prefixes (`train_`/`val_`/`held_`)
  so the three `sequence_id` sets are literally disjoint (also fixes the N5 name-by-index collision).
- `dynamic_rl.run_dynamic_training`: builds train/val/held via `_mk`; the periodic keep-best eval runs
  on the **validation** split (`sel_scenes = val_scenes`); held eval runs once at the end after the
  best (val-selected) state is loaded. `build_split_manifest` writes `split_manifest.json`. Result +
  activation record `checkpoint_selection={split:"val", metric:"val_discounted_episode_return",
  held_used_for_checkpoint:false}` and the per-split seeds/counts.
- Pilot fallback: `--dyn-val 0` ⇒ keep-best falls back to train and the manifest records
  `pilot_only_no_validation=true` / `checkpoint_selection_split="train"` (Contract §3.4: a no-validation
  run is explicitly NOT headline-eligible).
- `dynamic_headline.py`: passes `--dyn-val` (downstream readiness for the post-D3 headline).

## Tests
- Failing-first (fail on HEAD `27ab147`, pass after): `test_dynamic_train_val_held_seeds_disjoint`
  (TypeError on old: no `tag`), `test_split_manifest_records_all_three` (ImportError on old: no helper),
  `test_run_dynamic_training_uses_val_split_for_checkpoint` (integration; SystemExit on old: no
  `--dyn-val`). Plus `test_split_manifest_pilot_fallback_when_no_validation` (verifier-requested gap).
- Dynamic-RL file: **13 passed**. Full unit suite: **657/0**. Contract: **63/0**. `--dynamic` smoke:
  **exit 0**, `split_manifest.json` = 3 disjoint splits, checkpoint on val, `held_used_for_checkpoint=false`.

## Adversarial verification (Ultracode, 4-lens Workflow `wy6zozm4w`, all PASS)
Independent re-derivation found **no blockers / no majors**:
1. **Leakage**: checkpoint update gated only on `val_score` (from `sel_scenes`=val); held evaluated
   only after `best_state` is loaded; `warmstart_held` is measurement-only and never feeds selection.
2. **Disjointness/honesty**: `splits_disjoint` is COMPUTED via `set.isdisjoint` on real ids (not
   hard-coded); seeds +1/+333/+777 + tags give genuinely disjoint splits.
3. **T=1 byte-identity / decentralization**: `parse_args(None)` identical for CLI; `--dyn-val` read only
   inside the `--dynamic` branch (which returns before the T=1 code); deployed decoder still
   `local_mutual_assemble`, no critic at inference; all edits in gate-exempt `training/` + diagnostics +
   argparse.
4. **Test adequacy**: the 3 D3 tests are real anti-pseudo-tests (no mocks; real evaluator in the
   integration test) and genuinely fail on pre-D3 code.
- Two nits fixed this round: stale "decoded eval on TRAIN scenes" comment → corrected; added the
  pilot-fallback (`dyn_val=0`) unit test.

## Cost / scope
- Evaluator budget ~neutral: the periodic eval moved train→val (same per-eval cost); +1 split's context
  build. No multi-seed headline this round — still gated on D5/D6/D7 per the plan order. D3 removes the
  train-keep-best confound (forbidden §13.7) and makes the dynamic experiment headline-eligible once the
  remaining口径/data/observation fixes land.

## Next (D4 — parallel track)
Wire the phase-specific PBFT message plan (`build_pbft_message_plan`) into the production Stage-21
evaluator (`stage21_objective_stack_evidence.evaluate`) so dynamic energy/latency come from real
pre-prepare/prepare/commit messages (gap #5). Failing-first: `test_stage21_uses_pbft_message_plan_when_corrected`,
`test_pre_prepare_not_all_pairs`, `test_clients_do_not_vote_in_stage21`,
`test_energy_breakdown_protocol_relay_control`, `test_latency_uses_phase_completion_from_plan`.
(Alternatively D5 velocity features per the compute-limited order — D4 chosen next as it is a deployed-
evaluator correctness fix that both static and dynamic headlines depend on.)
