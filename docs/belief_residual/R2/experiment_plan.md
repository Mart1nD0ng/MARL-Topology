# R2 — experiment plan (CSI belief prediction auxiliary)

## Hypothesis (single variable: a supervised belief head on the policy actor's GRU)
Q14/Q2 left the GRU's temporal signal unsupervised — it had no target teaching it to turn stale history into
a current-CSI estimate. If the policy actor (`BeliefResidualActor`) gets a **belief head** trained with a
supervised auxiliary loss `L_CSI` (predict the TRUE current link psucc from stale history; true CSI is a
training-only LABEL, never an actor input), then under stale CSI the **recurrent** actor will predict the
current CSI with lower MSE than the **memoryless** one — making the temporal module genuinely useful (the
precondition for belief-guided edits, R5–R6). This is on the POLICY actor's GRU, not a separate Q2 predictor.

## Single change (additive to the R1 module + a new belief module/trainer)
1. `belief_head` appended to `BeliefResidualActor` (after `residual_head`, so R1 residual-head init RNG is
   unchanged): `belief(ef, ei, h)` → per-edge predicted current-psucc **logit** from the GRU hidden + edge feats.
2. New `src/marl_topology/training/csi_belief.py`:
   - `belief_target_logits(scene, t, edge_ids)` — the TRUE current psucc as a logit target
     (`scene.context(t).link_records[eid].link_success_probability`, training-only; the actor's ef has the
     STALE value). Leak-free by the same construction Q2 verified.
   - `belief_weights(edge_ids, prev_topo, anchor, candidates=None, criticality=None)` — `w = 1 + α·1[prev] +
     β·1[anchor] + η·1[candidate] (+ γ·criticality)` (emphasize decision-relevant edges).
   - `csi_belief_loss(belief_logits, target_logits, weights)` — weighted Huber in logit space.
   - `belief_mse(belief_logits, target_logits)` — held metric (probability space).
3. New `scripts/diagnostics/csi_belief_train.py` — trains the POLICY actor's GRU + belief_head with `L_CSI`
   (recurrent vs memoryless), reports held belief MSE per CSI mode {current, delay1, delay2, partial}. This
   is the path where the belief loss ENTERS a training loss (not a diagnostic).

## Mechanism-Path Matrix delta
- belief head / `L_CSI`: NOT_PRESENT → ACTIVE_IN_LOSS (in the R2 belief trainer; combined with PPO at R3).
- the GRU under supervised belief target: unsupervised → supervised (gradients from `L_CSI` reach the GRU).

## Failing-first tests (`tests/unit/test_belief_residual_R2_belief.py`) — fail on HEAD (module/head absent)
- `test_belief_head_training_target_is_true_current_csi` — target == logit(true current psucc at frame t).
- `test_actor_input_does_not_include_true_current_csi` — under delay≥1 the actor ef[psucc] (input) ≠ the
  belief target (true current); the true CSI is ONLY the label (leak-free).
- `test_belief_loss_enters_total_loss` — **load-bearing spy**: after a training step, gradients from `L_CSI`
  reach BOTH `belief_head` AND the shared `gru` (the belief loss is in the backward path, not a diagnostic).
- `test_belief_loss_logged` — the trainer logs belief MSE per CSI mode.
- `test_belief_prediction_improves_over_epochs` — held belief MSE decreases over training.
- `test_recurrent_belief_beats_memoryless_under_delay1` — **the exit condition**: trained recurrent held
  belief MSE < memoryless under delay (the temporal module recovers current CSI).

## Exit condition (R2 passes iff)
Recurrent belief MSE significantly < memoryless under stale/partial AND `L_CSI` genuinely enters the training
loss (gradients reach the GRU). If recurrent does NOT beat memoryless on the POLICY actor, record a
path-specific negative (the temporal module cannot recover current CSI even when supervised) and pause.

## Out of scope
Combining `L_CSI` with `L_PPO` in the residual trainer (R3); using the belief for edits (R5–R6). R2 proves
the supervised belief head makes the policy actor's GRU recover current CSI.
