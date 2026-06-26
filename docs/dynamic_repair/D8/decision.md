# D8 — decision (IN PROGRESS — headline running)

**Status: driver built + pilot-validated; the 5-seed headline is RUNNING in the background. The
scientific verdict (does velocity / recurrence help) is DEFERRED to the next iteration when the headline
completes — it must NOT be drawn from the pilot.**

## What changed (this stage = a result driver, not a code mechanism)
- New `scripts/diagnostics/dynamic_d8_matrix.py`: runs the 2×2 {motion × recurrence} matrix
  (memoryless/recurrent × current-CSI/velocity) via the `--dynamic` subprocess (all arms share the
  corrected D2/D3/D4 objective + the decoder-aware D6 warm-start `--dyn-warmstart-mode bcsp`), reads each
  run's held metrics + critic EV/KL, evaluates the D7 DEPLOYABLE baselines + the CENTRAL myopic-greedy
  reference on the SAME held set per seed, and emits a grouped report (learned_arms / deployable_policies
  / central_references) with per-seed + 95% CI + paired (velocity−csi) and (recurrent−memoryless) diffs.
- Pure helpers `D8_ARMS` (the 2×2) + `build_report` are unit-tested.

## Tests
- `test_d8_arm_matrix_is_2x2` (the 4 {motion × recurrence} combos), `test_d8_report_groups_arms_and_baselines`
  (groups learned arms / deployable / central separately, per-seed → CI; scope flags single-RSU).
- Unit suite **680/0**; contract **63/0**.

## Pilot (2 seeds, N=8, 8 updates, warm-start bcsp 12 — VALIDATION ONLY, NOT a headline)
The pilot ran end-to-end (exit 0) and the report is structurally correct:
- all 4 learned arms + 2 deployable baselines (local_threshold, local_hysteresis) + 1 central reference
  (myopic_greedy), grouped separately;
- the per-arm flags are correctly ACTIVE: `memoryless_csi` (motion=F, rec=F), `memoryless_velocity`
  (motion=T, rec=F), `recurrent_csi` (motion=F, rec=T), `recurrent_velocity` (motion=T, rec=T), all
  `warmstart_mode=bcsp`; deployable `action_evaluator_calls=0`.
- The pilot NUMBERS are pure noise (2 seeds, N=8, 8 updates → return CIs span ≈[−50, +42]) and are
  explicitly NOT interpreted — the pilot only proves the matrix wires the factors correctly.

## Headline run (launched, BACKGROUND)
`dynamic_d8_matrix.py --seeds 0 1 2 3 4 --dyn-nodes 8 12 16 --updates 30 --dyn-train 24 --dyn-val 24
--dyn-held 24 --frames 6 --dyn-warmstart 25 --dyn-warmstart-mode bcsp --dyn-bc-anchor 0.0` →
`result_save/dynamic_d8_matrix.json`. (The BC-anchor is OFF for the main 2×2 to isolate the
motion×recurrence factors; the anchor's effect is a separate D6 pilot question.)

## Next (next iteration)
Collect `result_save/dynamic_d8_matrix.json`, write the per-seed + CI tables (held discounted return,
per-frame feasibility, switches/frame, warm-start-alone return, post-RL drift, critic EV, actor KL),
draw the scoped conclusion (velocity vs current-CSI; recurrent vs memoryless — at matched features),
compare against the DEPLOYABLE baselines and the CENTRAL references (grouped), adversarially verify the
conclusion, and finalize this decision. **Scope: single-RSU random geometry — D1 urban data still
pending — so the conclusion does NOT extrapolate to urban.**
