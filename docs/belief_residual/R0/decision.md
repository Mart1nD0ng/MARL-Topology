# R0 — decision (freeze Q14 + residual-trainer path audit)

**Result: KEEP / audit complete.** The Q14 negative is formally re-scoped to ONE path (REINFORCE residual +
fixed flip-penalty); residual PPO / CTDE critic / entropy / CSI-belief / evidence-gated edits / beneficial-edit
supervision are NOT_PRESENT in that path and were never tested. PPO exists only in the Graph-MAPPO trunk
(Contract v4 §7: that is not residual PPO). No production code changed; 3 load-bearing audit tests pin this.

## What was done
- **Mechanism-Path Matrix** (HEAD snapshot) written into `docs/CURRENT_BELIEF_RESIDUAL_STATUS.md` §1.
- **Audit artifact** `result_save/belief_residual/R0/mechanism_activation.json`.
- **Load-bearing path tests** `tests/unit/test_belief_residual_R0_audit.py` (3 pass, 10s):
  1. graph_mappo defines `ppo_clip_actor_loss` + `approx_kl` (epoch-0 ratio→1 ⇒ approx_kl 0, loss −mean(A)).
  2. residual trainer source is REINFORCE + baseline + flip_pen; NO ppo/approx_kl/critic/entropy/belief/L_CSI
     (only `clip_grad_norm_`).
  3. **spy / R3 tripwire**: a real residual `reinforce_update` never calls `ppo_clip_actor_loss` (0 calls).
- Q14 raw results frozen by citation (Claim Card §8 fields: source_commit a634928, source_artifact).

## Effect-on-decision metrics
N/A — R0 runs no policy; it audits the path. (Effect-on-decision metrics begin at R1 with the saturation
metrics: `recurrent_memoryless_logit_delta`, `recurrent_memoryless_action_delta`, `saturation_rate`.)

## Mechanism-Path Matrix delta (this stage)
No status change — R0 records the HEAD baseline. The residual path is REINFORCE-only; every target mechanism
(PPO/critic/entropy/belief/edit-supervision/gating/adaptive-KL) is NOT_PRESENT and scheduled R1–R7.

## Path-specific negative restatement (Contract v4 §5)
> "REINFORCE residual trainer WITHOUT PPO/critic/entropy/CSI-belief, with a fixed flip-penalty and a shared
> ±10-saturating logit head, fails to recover the stale-CSI feasibility loss under delay1 urban at N≤16
> (bimodal: 3/5 clamp, 2/5 collapse; recurrence behaviorally inert)."
This does NOT prove residual PPO or belief-augmented policy failure.

## Acceptance (Contract v4 §15)
- Claim Cards written ✓ (`claim_cards.yaml`) / Mechanism-Path Matrix updated ✓ / Load-bearing tests pass ✓
  (3/3) / Effect-on-decision N/A (audit) / activation artifact ✓ / decision ✓ / scope explicit ✓.
- Decision: **KEEP** — proceed to R1.
- Next: **R1** — feature standardization (all train frames) + logit-saturation fix (raw-logit L2 + separate
  small-range residual head) + saturation metrics. Exit when saturation drops and recurrent vs memoryless
  logits are no longer bit-identical (else pause control experiments, per Workflow R1).
