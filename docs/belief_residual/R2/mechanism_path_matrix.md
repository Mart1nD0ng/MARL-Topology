# R2 Mechanism-Path Matrix delta (Contract v4 §2)

| mechanism | before R2 | after R2 | path / evidence |
|---|---|---|---|
| CSI belief head (`belief_head`) | NOT_PRESENT | **IMPLEMENTED + ACTIVE_IN_LOSS** (belief trainer) | `BeliefResidualActor.belief`; grads reach it (spy) |
| `L_CSI` (weighted Huber, logit space) | NOT_PRESENT | **ACTIVE_IN_LOSS** (in `train_belief`) | `csi_belief.csi_belief_loss`; combined with PPO at R3 |
| true current psucc as training-only label | (Q2 diagnostic only) | **ACTIVE_IN_LOSS, leak-free** | `csi_belief.belief_target_logits` reads `scene.context(t)`; actor input = stale ef |
| GRU under supervised belief target | unsupervised | **supervised** (grads from L_CSI reach gru) | `test_belief_loss_enters_total_loss` |
| recurrence improving belief | (untested on policy actor) | **NO robust advantage** (noise-level vs memoryless) | `belief_multiseed.json` paired CI |

Unchanged / scheduled: residual PPO + CTDE critic + entropy (R3 — will add `L_CSI` to `L_PPO`); beneficial-edit
supervision (R4–R5); evidence-gated action (R6); adaptive anchor-KL (R7). The belief auxiliary is now
ACTIVE_IN_LOSS in the belief trainer; R3 carries it into the combined policy loss. The R0 PPO-spy tripwire
still holds (the residual trainer still does not call `ppo_clip_actor_loss`).
