# D8 — experiment_plan

**Stage:** D8 — retest recurrent-vs-memoryless across the 2×2 {motion × recurrence} matrix on the
fully-corrected pipeline (Plan §10). The FIRST comparative result stage (still single-RSU geometry until
D1 — the conclusion is scoped accordingly).

**Hypothesis (pre-registered):** With the corrected pipeline (D2 hold/discount objective, D3 validation
split, D4 phase-specific energy, D5 motion features, D6 decoder-aware warm-start + anti-drift anchor),
does temporal structure help? Two questions: (1) does adding VELOCITY features make a memoryless actor
match/beat a recurrent one (the §3 Markovness prediction)? (2) does cross-frame RECURRENCE add anything
over a memoryless actor with the same features? Pre-registered expectation: velocity ≥ current-CSI
(motion features give the memoryless actor the Markov signal), and recurrent ≈ memoryless at matched
features (the bounded temporal headroom). Honest: this is single-RSU random geometry, NOT urban.

**The 4 arms (2×2):**
- `memoryless_csi` — current-CSI only, memoryless (the historical baseline).
- `memoryless_velocity` — `--motion-features`, memoryless.
- `recurrent_csi` — current-CSI only, recurrent (cross-frame hidden).
- `recurrent_velocity` — `--motion-features`, recurrent.
All share the SAME corrected config (D2/D3/D4 always on) + the SAME decoder-aware warm-start
(`--dyn-warmstart-mode bcsp`) so the ONLY differences are {motion on/off} × {recurrent on/off} — a clean
2-factor ablation.

**Single change (this stage = a result, not a code mechanism):** a new driver
`scripts/diagnostics/dynamic_d8_matrix.py` that (a) runs the 4 arms × seeds via the `--dynamic`
subprocess, (b) reads each run's `dynamic_result.json` + `training_history.json` (held discounted return,
per-frame feasibility, switches/frame, warmstart-alone return, post-RL drift, critic EV, actor KL,
train/val/held), (c) evaluates the D7 DEPLOYABLE baselines (local_threshold, local_hysteresis) + the
CENTRAL myopic-greedy reference on the SAME held set per seed, and (d) emits a grouped report
(learned_arms / deployable_policies / central_references) with per-seed values + 95% CIs + paired
(velocity−csi) and (recurrent−memoryless) diffs.

**Controlled variables:** corrected pipeline always on; warm-start mode bcsp for all arms; same
seeds/splits/N/frames/updates across arms. Only {motion, recurrence} vary.

**Required failing tests (fail on `f6dbbf7`, pass after):**
- `test_d8_arm_matrix_is_2x2` — the driver's arm matrix is exactly the 4 {motion × recurrence} combos.
- `test_d8_report_groups_arms_and_baselines` — the report-builder groups learned_arms / deployable_policies
  / central_references separately, each with per-seed + CI, and computes the paired diffs.

**Plan:**
1. Driver + pure-helper tests (this turn).
2. **PILOT** (2 seeds, N=8, ~12 updates, warm-start bcsp 15) to validate the 4-arm matrix + baseline
   grouping run end-to-end and the arms differ correctly (motion/recurrence flags active). PILOT params
   are NOT a headline.
3. **Launch the real multi-seed headline in the BACKGROUND** (≥5 seeds, N∈{8,12,16}, 30 updates,
   warm-start bcsp 25). Collect + analyze + write the headline + per-seed/CI tables next iteration.

**Expected activation:** each run's `mechanism_activation.json` shows the arm's `cross_frame_recurrence`
+ `motion_features` + `warmstart_mode=bcsp`; the report groups the three baseline classes.

**Success criterion:** the 2 tests pass; the pilot runs end-to-end producing the grouped report (all 4
arms + 2 deployable + 1 central, per-seed + CI); the real run is launched in the background. The
SCIENTIFIC conclusion (does velocity/recurrence help) is drawn only from the ≥5-seed run, reported with
CI and scoped to single-RSU geometry — NOT from the pilot.

**Failure criterion:** if the pilot reveals the 4-arm matrix mis-wires a flag (e.g. motion not active in
an arm), fix before the headline. If arms collapse (training instability), report per-seed honestly and
diagnose (warm-start, lr) — do not hide failed seeds.
