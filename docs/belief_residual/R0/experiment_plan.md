# R0 — experiment plan (freeze Q14 + residual-trainer path audit)

## Hypothesis (the single thing R0 establishes)
The Q14 negative belongs to ONE specific path — the REINFORCE residual trainer with a fixed flip-penalty —
NOT to "residual learning" in general, and NOT to residual PPO / CTDE critic / CSI-belief, which are
NOT_PRESENT in that path. PPO exists only in the Graph-MAPPO trunk; per Contract v4 §7 that does not count as
residual PPO.

## Single change (audit only — no production code change)
Add executable audit evidence (load-bearing path tests) + the Mechanism-Path Matrix; freeze Q14 raw results
by reference. No `src/` change.

## Load-bearing path tests (Contract v4 §3) — `tests/unit/test_belief_residual_R0_audit.py`
1. `test_graph_mappo_defines_ppo_clip_and_approx_kl` — PPO machinery EXISTS in the trunk (epoch-0 ratio→1 ⇒
   approx_kl 0, loss −mean(A)).
2. `test_residual_trainer_source_is_reinforce_no_ppo_no_critic_no_csi` — the residual trainer source is
   REINFORCE + baseline + flip_pen, with NO ppo/approx_kl/critic/entropy/belief/L_CSI (only `clip_grad_norm_`).
3. `test_residual_update_does_not_call_ppo_clip_spy` — **the load-bearing spy / R3 tripwire**: run one real
   residual `reinforce_update` with `graph_mappo.ppo_clip_actor_loss` monkeypatched; assert it is never called.
   When R3 wires PPO into the residual trainer, this test FLIPS (must be updated) — proving the change is real.

## Exit condition (R0 passes iff)
- The three tests pass on HEAD (documenting the pre-fix reality).
- The "PPO exists in trunk" vs "PPO active in residual path" distinction is pinned in code + the matrix.
- The Q14 conclusion is re-scoped (Claim Card) so it cannot be misread as a residual-PPO / belief failure.

## Out of scope
Any fix (R1+). R0 changes no production code; it only audits and freezes.
