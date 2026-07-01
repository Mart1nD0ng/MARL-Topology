# T3 — Mechanism-Path Matrix (Contract v4 §2)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | code_path | status @ T3 | evidence |
|---|---|---|---|
| belief correction target (`stale_logit`, `belief_correction_target`) | `training/csi_belief.py` | ACTIVE_IN_LOSS | `belief_logit=stale_logit+head`; Huber(stale_logit+head, true); test_correction_target_makes_echo_nonoptimal |
| directional accuracy metric (`directional_accuracy`) | `training/csi_belief.py` | ACTIVE_IN_EVAL | held dir_acc ~0.92; test_directional_accuracy_rewards_correct_sign |
| move-weight magnitude emphasis (`_train(move_weight)`) | `t3_correction_belief_gen.py` | CALLABLE (sub-pilot, default 0) | backfires (MSE 0.10→0.35, n=1) |
| in-policy belief head (shares GRU) | `belief_residual_actor.belief` (R2) | ACTIVE_IN_LOSS | trained with the correction loss; residual_leak=0.1 (T2) |
| recovered-psucc → anchor (T1 injection) | `t1_oracle_recovery_gen._anchor_with_psucc` (reused) | ACTIVE_IN_EVAL | feas_gain spans 0 / negative |
| true current psucc = training-only label | `belief_target_logits` | ACTIVE_IN_LOSS (label only) | leak-free (R2 construction; correction adds stale_logit from stale obs) |

**Blast radius:** `csi_belief.py` gains 3 pure functions (no change to existing R2 functions). The generator is
new under `scripts/diagnostics/`. `belief()` and the actor are unchanged. `training/` is gate-exempt (CTDE).
