# D4 — decision

**Result: KEEP** (pending final verification verdict — see §Verification).

The phase-specific PBFT message plan is wired into the production Stage-21 evaluator's energy/latency
accounting and ACTIVATED in the operating-point (production/dynamic) regime; the legacy all-pairs-×3
reuse remains the byte-identical default for every other caller.

## What changed (one variable: phase-specific energy/latency accounting)
- `Stage21ObjectiveStackConfig.phase_specific_accounting: bool = False` (opt-in).
- `evaluate()`: builds a SEPARATE `acct_phase_records` (phase-specific when on, else the legacy
  all-pairs-×3) for `account_pbft_protocol_latency_energy` + `_consensus_completion_latency`. The
  RELIABILITY matrices/`phase_records` are UNTOUCHED (the expected-initiator model averages over all
  primaries internally → its matrices must stay the full validator set). `_phase_specific_phase_records`
  filters the full directed records to each phase's `build_pbft_message_plan` pair set:
  pre_prepare = primary→backups star (canonical view-0 primary `validators[0]`), prepare/commit =
  validator↔validator, clients excluded. `metrics["energy_breakdown"]` (per-phase energy + message
  counts + honest term ledger) is added only when on.
- `PhysicsRegime.phase_specific_accounting` + `build_stack_config` passthrough; `operating_point_regime`
  sets it `True` → the dynamic/production path actually USES it (not an inert primitive, Contract §16.6).

## Modeling note (honest)
- pre_prepare energy is accounted for the view-0 primary (`validators[0]`); a PBFT view has one primary.
  View-change / primary-rotation energy is the deferred `view_change` term. (Reliability separately
  averages over all primaries for Byzantine robustness — a documented model distinction.)
- The `energy_breakdown` honestly labels `relay_folded_into_route_energy=True` (relay energy is already
  inside the per-link multi-hop route energy) and `control/reconfig/view_change=0.0` (NOT modeled here /
  the dynamic arm charges reconfiguration separately). D4 corrects the **protocol** term; it does **not**
  claim a complete S4.9 energy/latency optimization.

## Tests (failing-first; fail on `5406749`, pass after)
- `test_pre_prepare_is_star_not_all_pairs` (pre_prepare = n_val−1 < n_val(n_val−1)),
  `test_phase_specific_energy_below_all_pairs_x3`, `test_clients_do_not_vote` (clients excluded; prepare/
  commit = n_val(n_val−1)), `test_reliability_unchanged_by_flag`, `test_phase_accounting_off_byte_identical`,
  `test_operating_point_regime_activates_phase_specific_accounting`.
- Unit suite **664/0**; contract/gate suite **63/0** (deployed-path change trips no banned-literal gate);
  `--dynamic` smoke **exit 0** with the production regime activating the corrected accounting.

## Impact measurement (corrected vs legacy, operating-point scenes, full graph)
| scene | N_val | E_legacy | E_corrected | ΔE | reliability off→on |
|---|---|---|---|---|---|
| m0 | 12 | 0.1897 | 0.1302 | −31.4% | 0.0000→0.0000 |
| m1 | 7 | 0.0643 | 0.0274 | −57.5% | 0.0000→0.0000 |
| m2 | 11 | 0.1920 | 0.1019 | −46.9% | unchanged |
| m4 | 15 | 0.2867 | 0.1555 | −45.8% | unchanged |
| m5 | 16 | 0.3168 | 0.2136 | −32.6% | unchanged |

Corrected protocol energy is **31–58% lower** (the pre-prepare all-pairs over-count removed; clients
excluded). **Reliability is byte-identical on↔off** (matrices untouched). Latency unchanged on these
failing topologies (full-timeout regardless); the latency difference would appear on feasible topologies.
N=16 breakdown: pre_prepare 15 msgs (star) vs prepare/commit 240 each (16×15 validator vote).

## Scope / next
- This is a genuine deployed-evaluator correctness fix, ACTIVE in the production/dynamic regime, so the
  forthcoming D5–D8 measurements use corrected protocol energy (no rework). It does NOT model
  control/reconfig/view-change energy (deferred) — so it is NOT a "complete energy optimization."
- Next: **D5** (actor motion features: velocity/heading/relative-velocity/CSI-delta/CSI-age) — the
  plan's compute-limited critical path.

## Verification — found a real BLOCKER, fixed it, re-verified PASS
4-lens adversarial Workflow (`wwfl2hn3t`): lens 2 (plan-correctness, clients-don't-vote, reliability
separation) PASS; but lens 1 returned **BLOCKER** —

- **BLOCKER (fixed):** the `VectorizedStage21Evaluator` (the dynamic arm's DEFAULT fast path —
  `DynamicScene.vectorized=True`) had the same all-pairs-×3 reuse and silently IGNORED the flag, so D4
  was **inert in the dynamic reward** and the vectorized path would have diverged from the canonical
  (breaking its float-identity contract). Fix: applied the identical phase-specific accounting +
  energy_breakdown to `vectorized_objective_stack_evaluator.py` (imports `_phase_specific_phase_records`
  from the canonical module; reliability matrices still use the full `phase_records`). **Confirmed
  active in the dynamic path:** the dynamic context evaluator is `VectorizedStage21Evaluator` and now
  emits `energy_breakdown` (N=12 → pre_prepare=11 star, prepare=132 vote).
- **MAJOR (addressed):** added `test_vectorized_matches_canonical_with_phase_specific` (vectorized↔canonical
  float-identity with the flag on — pins the blocker fix) and `test_reliability_unchanged_by_flag_fixed_set`
  (reliability invariance under the production `fixed_set` fault model + per-primary vector).
- **nit (addressed):** documented the view-0 primary (`validators[0]`) modeling choice in the config docstring.
- **Re-verification (focused single-agent, all 5 claims PASS):** reliability untouched in the vectorized
  path; phase-specific logic mirrors canonical line-for-line; energy_breakdown byte-equivalent; off-path
  byte-identical; no third production evaluator bypasses the fix.

Final: unit **666/0**, contract **63/0**; 10 phase-accounting/vectorized tests pass. The verification
earned its keep — it caught the exact "正式 run 仍走简化 evaluator" trap the contract warns about.
