# T0 — experiment plan (freeze + temporal/activation audit)

## Hypothesis (audit stage, no mechanism change)
The prior campaign's chain-1 (temporal/belief) failed for **architectural** reasons, not because stale→current
CSI recovery is impossible. T0 pins those reasons in code as load-bearing regression tripwires, so T2–T4 have a
falsifiable target and the T1 gate is framed correctly.

## Controlled variable
None — T0 changes no runtime path (byte-identical model/train paths). It only adds a status doc + audit tests +
stage docs. The 4 defects are documented facts about HEAD.

## The 4 pinned defects (each becomes a load-bearing test)
1. **Direct-`p_t` belief target** — `belief_target_logits` returns the *absolute* true-current logit and
   `csi_belief_loss` compares the head output to it; for a stale≈current edge the loss-optimal output is
   `logit(stale)` → echoing the observation is optimal. Test: the target for a stale==current edge equals
   `logit(stale)` (echo-optimal), and the target is independent of the stale observation (absolute, not a delta).
2. **`tanh` saturation on the acted logit** — `forward` returns `z=z_max·tanh(raw/z_max)`; the map saturates to
   ±z_max and its gradient vanishes for |raw|≫z_max. Test: the actor's forward applies this exact map; a large
   raw rails the logit; d(logit)/d(raw)→0 at the rail.
3. **Per-node recurrence for per-edge dynamics** — `self.gru` is a `GRUCell`; hidden is `[N,H]` (one row per
   node) while CSI features are `[E,·]` (one row per edge). Test: `h.shape == (n_nodes, hidden)` and the only
   recurrent module is a single per-node `GRUCell`.
4. **No edge-level recurrent state** — tripwire asserting absence (flips in T4). Test: no `edge_gru`/edge hidden;
   exactly one recurrent cell and it is per-node.

## Steps
1. Write `docs/CURRENT_TEMPORAL_RECOVERY_STATUS.md` (gap + Mechanism-Path Matrix + T-plan). ✓
2. Write `tests/unit/test_temporal_recovery_T0_audit.py` (the 4 tests above). ✓
3. Run the new test file (must pass on HEAD) + the affected unit suite (must stay green).
4. Write T0 stage docs (this plan, matrix, claim cards, decision) + pre-stage the T1 plan.
5. Commit `T0` to `decentralized-marl-trunk` (no push).

## Definition of done (Contract v4 §15)
Claim Card written; Mechanism-Path Matrix updated; load-bearing audit tests pass; scope explicit; decision.md
written. No research conclusion at T0 (no seeds/A-B) — it is a freeze+audit stage.
