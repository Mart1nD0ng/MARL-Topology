# D3 — experiment_plan

**Stage:** D3 — train/val/held three-split; checkpoint by validation only (Plan §5; gap #4).

**Hypothesis:** Checkpoint selection currently runs on the TRAIN set (`dynamic_eval(actor, train_scenes)`),
and there is no independent validation split — only train (seed`*1000+1`) and held (seed`*1000+777`).
Per Contract §3.4 / forbidden §13.7+§13.10 this makes the result pilot-only and lets a train→held drop
be mislabeled a "generalization gap". Adding an independent validation split (seed`*1000+333`) and
selecting the checkpoint by the **validation** discounted episode return (held used ONLY for final
reporting) makes the dynamic experiment headline-eligible and removes the train-keep-best confound.

**Single change (one variable = the data split used for checkpoint selection):**
- New `--dyn-val` trajectories at seed`*1000+333`, disjoint from train (`+1`) and held (`+777`).
- `sample_dynamic_scenes` gains a `tag` so the three splits have literally disjoint `sequence_id`s
  (also fixes the N5 nit: ids were name-by-index, identical strings across splits).
- `run_dynamic_training` runs the periodic keep-best eval on `val_scenes` (NOT `train_scenes`); the
  checkpoint = argmax val discounted episode return; held eval happens once at the end (final only).
- Write `split_manifest.json` (Contract §14): the three splits' seeds, counts, disjoint ids, the
  checkpoint-selection split, and `held_used_for_checkpoint=false`.
- `mechanism_activation.json` / `dynamic_result.json` record the three-way split + checkpoint source.

**Controlled variables:** reward/objective (D2, unchanged), sampler geometry, actor/critic, decoder,
PPO hyperparams, seeds offsets formula, warm-start. Only the checkpoint-selection split changes. No
new mechanism. Evaluator budget ~neutral: the periodic eval moves from train→val (same per-eval cost);
+1 split's context build.

**Dataset:** existing single-RSU `sample_dynamic_scenes` (D1 replaces geometry later — out of scope).

**Seeds / budget:** unit tests + 1 in-process integration test (tiny run) + 1 real-shard `--dynamic`
smoke. No multi-seed headline this round (still gated on D5/D6/D7 per the plan order; D3 just removes
the train-keep-best confound).

**Required failing tests (fail on HEAD `27ab147`, pass after):**
- `test_dynamic_train_val_held_seeds_disjoint` — three splits (tags train_/val_/held_) have disjoint
  `sequence_id`s and the geometry differs across splits at the same index (proves distinct seeds).
- `test_split_manifest_records_all_three` — `build_split_manifest(...)` returns all three splits with
  disjoint ids, `checkpoint_selection_split=="val"`, `held_used_for_checkpoint is False`.
- `test_run_dynamic_training_uses_val_split_for_checkpoint` (integration) — a tiny in-process
  `run_dynamic_training` writes `split_manifest.json` with three disjoint splits and selects the
  checkpoint on val (held not used for checkpoint). Covers the plan's `test_checkpoint_uses_val_not_train`
  + `test_held_not_used_until_final`.

**Expected activation:** `dynamic_task` records `splits={train,val,held}` sizes + seeds;
`checkpoint_selection={"split":"val","metric":"val_discounted_episode_return"}`;
`held_used_for_checkpoint=false`; `split_manifest.json` present.

**Success criterion:** the 3 tests pass; full unit suite + contract suite stay green; the `--dynamic`
smoke exits 0 and its `split_manifest.json` shows three disjoint splits with checkpoint on val.

**Failure criterion:** if the val eval cost makes the smoke pathological, reduce `--dyn-val` default /
eval frequency — do NOT fall back to train keep-best.
