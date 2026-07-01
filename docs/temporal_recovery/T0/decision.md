# T0 — decision (freeze + audit)

**Disposition: KEEP (audit baseline established).** T0 changes no runtime path. It pins the four defects that
made the prior chain-1 fail as load-bearing regression tripwires and frames the T1 gate correctly.

## What T0 established
- The prior temporal/belief failure is **architectural**, localized to three code facts (Claim Cards + Matrix):
  (1) the belief target is the *absolute* current `p_t` → stale-echo is loss-optimal; (2) the residual decision
  logit is `tanh`-saturated → the GRU can't reach the acted logit; (3) recurrence is per-node while CSI is a
  per-edge attribute; (4) there is no edge-level recurrent state and no uncertainty head.
- The anchor decides **purely on psucc col 0** ([dynamic_baselines.py:59](src/marl_topology/training/dynamic_baselines.py:59)),
  so the T1 oracle can inject a recovered psucc into the anchor cleanly and the feasibility gap maps directly to
  psucc-recovery quality.
- The prior **anticipation** oracle (`logs/diagnose_anticipation_value.py`, +0.000) is a *forecasting* result
  and does NOT settle the *nowcasting* question T1 asks. They must stay distinct (Contract §5/§14).

## Load-bearing tests (fail on the fixed code → regression tripwires)
`tests/unit/test_temporal_recovery_T0_audit.py`:
- `test_belief_target_is_absolute_current_echo_optimal` — flips when T3 switches to a correction target.
- `test_residual_logit_saturates_and_gradient_vanishes` — flips when T2 removes the tanh saturation.
- `test_recurrence_is_per_node_and_no_edge_state` — flips when T4 adds edge-level recurrent state.

## Scope boundary
T0 proves nothing about whether recovery *works* — that is the T1 gate. No seeds, no A/B, no research
conclusion. It only establishes the falsifiable baseline.

## Next
**T1 (GATE):** the 4-arm nowcasting oracle (stale / current-belief / physics-recovery / true) + MSE landscape,
5-seed random+urban. If physics-recovery beats the stale floor (C−A>0) → recovery is realizable → proceed to
T2–T5. If not → per the task-1 fallback, pivot to improving the env's leak-safe temporal hidden features.
