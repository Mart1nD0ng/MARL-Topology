# R1 Mechanism-Path Matrix delta (Contract v4 §2)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | before R1 | after R1 | path / evidence |
|---|---|---|---|
| separate small-range residual head (z_max∈[2,4]) | NOT_PRESENT (shared ±10 head) | **IMPLEMENTED_ONLY** | `BeliefResidualActor` (`belief_residual_actor.py`); not yet in a trainer (R3) |
| raw-logit L2 penalty | NOT_PRESENT | **CALLABLE** (used in the R1 pilot loss) | `residual_saturation.raw_logit_l2_penalty`; ACTIVE_IN_LOSS at R3 |
| saturation / degeneracy metrics | NOT_PRESENT | **CALLABLE** (ACTIVE_IN_EVAL in pilot) | `residual_saturation.saturation_metrics` + `recurrent_vs_memoryless_delta` |
| all-train-frame standardization | NOT_PRESENT (frame-0-only) | **CALLABLE** | `residual_saturation.feature_standardization_all_frames` |
| cross-frame GRU recurrence reaching the residual logit | inert (saturated) | **ACTIVE at logit level** (delta 0.0257>0); action-level still 0 | `BeliefResidualActor.encode`→`residual_raw` |

Unchanged this stage (still scheduled): residual PPO / CTDE critic / entropy (R3), CSI-belief auxiliary (R2),
beneficial-edit supervision (R4–R5), evidence-gated action (R6), adaptive anchor-KL (R7). The new head is
IMPLEMENTED_ONLY — it becomes ACTIVE_IN_LOSS only when R3 wires it into the residual PPO trainer. Per Contract
v4 §7 the R0 PPO-spy tripwire remains valid (the residual trainer still does not call `ppo_clip_actor_loss`).
