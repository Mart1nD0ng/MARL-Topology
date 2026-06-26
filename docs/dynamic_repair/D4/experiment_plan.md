# D4 — experiment_plan

**Stage:** D4 — wire the phase-specific PBFT message plan into the production Stage-21 evaluator
(Plan §6; gap #5).

**Hypothesis:** `evaluate()` reuses ONE all-pairs record set for all three PBFT phases
(`phase_records = {pre_prepare: records, prepare: records, commit: records}`, line 357), so the energy
accounting (`_account_phase` sums `network_energy_j` over the FULL mesh ×3) over-counts the pre-prepare
round (which should be a primary→backups STAR, not all-pairs) and includes client links. Replacing the
**energy/latency accounting** record sets with the phase-specific plan (`build_pbft_message_plan`:
pre_prepare = primary→backups, prepare/commit = validator↔validator, clients never vote) makes the
energy/latency physically correct, opt-in and byte-identical when off.

**Key correctness constraint (do NOT break reliability):** the expected-initiator / fixed-set
reliability model averages over ALL possible primaries internally, so the **reliability matrices must
stay the full validator-validator set** (unchanged). D4 only changes the ENERGY/LATENCY accounting
record sets — reliability stays byte-identical always.

**Single change (one variable = phase-specific energy/latency accounting), opt-in:**
- `Stage21ObjectiveStackConfig`: new `phase_specific_accounting: bool = False` (default off →
  byte-identical).
- `evaluate()`: when on, build `acct_phase_records` by filtering the full `records` to the plan's
  directed pairs (pre_prepare = canonical view-0 primary `validators[0]` → backups; prepare/commit =
  validator↔validator); feed it to `account_pbft_protocol_latency_energy` + `_consensus_completion_latency`.
  The reliability `phase_records` / matrices are untouched. Add `metrics["energy_breakdown"]`
  (per-phase protocol energy + honest notes: relay folded into route energy; control/reconfig/
  view_change not modeled here / deferred) only when the flag is on.
- `PhysicsRegime`: `phase_specific_accounting: bool = False`; `build_stack_config` passes it through.
- `operating_point_regime`: set `phase_specific_accounting=True` so the dynamic/production path actually
  USES the corrected accounting (not an inert primitive — Contract §16.6).

**Modeling note (honest):** pre_prepare energy is accounted for the view-0 primary (`validators[0]`); a
single PBFT view has one primary. View-change / primary-rotation energy is the deferred `view_change`
term. (The reliability separately averages over primaries for Byzantine robustness — a documented
distinction between the reliability and energy models.)

**Controlled variables:** reliability computation, fault model, relay, MAC, decoder — all unchanged.
Only the energy/latency accounting record sets change. `data/` is a gate-scanned deployed path → keep
closed-form, no PBFT state machine, no banned literals.

**Required failing tests (fail on HEAD `5406749`, pass after):**
- `test_stage21_pre_prepare_is_star_not_all_pairs` — with the flag on, pre_prepare phase energy ≤
  prepare phase energy and pre_prepare message count = (n_val−1) (a star), strictly < the all-pairs
  count.
- `test_stage21_clients_do_not_vote` — with coverage-gated clients, no client id is a sender in the
  prepare/commit accounting record sets.
- `test_stage21_phase_specific_energy_below_all_pairs_x3` — corrected protocol energy < the legacy
  all-pairs-×3 energy (the over-count is removed).
- `test_stage21_energy_breakdown_present_when_corrected` — `metrics["energy_breakdown"]` exposes
  per-phase protocol energy (+ honest relay/control notes); absent when off.
- `test_stage21_phase_accounting_off_byte_identical` — flag off ⇒ metrics identical to HEAD (opt-in
  safety), and reliability identical with the flag on vs off (reliability untouched).

**Expected activation:** the operating-point regime sets `phase_specific_accounting=True`; the dynamic
`mechanism_activation.json` regime string + the evaluator's energy_breakdown reflect it.

**Success criterion:** the 5 tests pass; full unit + contract suites green (any test that pinned the OLD
all-pairs-×3 operating-point energy is updated to the corrected口径 ONLY if it genuinely pinned the bug —
scrutinized, never weakened); a real-shard impact measurement reports corrected-vs-legacy energy/latency
on operating-point scenes; reliability unchanged.

**Failure criterion:** if activating in the operating-point regime breaks a test for a reason OTHER than
a legitimately-corrected energy value, investigate — do not weaken the test. If the impact is
pathological, land opt-in but defer the operating-point activation with the measured rationale.
