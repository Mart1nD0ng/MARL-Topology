# R1 — experiment plan (feature standardization + logit-saturation fix)

## Hypothesis (single variable: the residual logit head + its input scale)
Q14's "recurrence is behaviorally inert" was caused by the residual path reusing the shared **±10 tanh
activation head**, whose trained `raw`≈3000 saturated 83% of logits and annihilated the GRU's contribution —
NOT by recurrence being useless. If the residual path uses a **separate small-range head** (`z = z_max·tanh(
raw/z_max)`, `z_max∈[2,4]`) with **raw-logit L2** and **all-train-frame standardization**, then under stale CSI
the recurrent vs memoryless residual logits will **no longer be bit-identical** and the head will be
unsaturated. (Per Contract v4 §8/§10, no "recurrence ineffective" claim is allowed until this is fixed.)

## Single change (additive — does NOT modify the verified DynamicRecurrentActor / residual trainer)
1. New `src/marl_topology/models/belief_residual_actor.py::BeliefResidualActor` — the proven node-encoder +
   directional message passing + per-node GRUCell (cross-frame hidden), but with a **separate residual policy
   head** that outputs small-range logits `z_max·tanh(raw/z_max)` (default `z_max=3`) and **exposes `raw`**
   for the L2 penalty + saturation metrics. (Belief / repair / safety heads are placeholders for R2/R5.)
2. New `src/marl_topology/training/residual_saturation.py`:
   - `feature_standardization_all_frames(scenes)` — pool observations across ALL frames (not just frame 0).
   - `saturation_metrics(raw, logits, logit_scale)` — `raw_abs_mean`, `raw_abs_p95`, `frac_abs_raw_gt_30`,
     `frac_abs_logit_gt_9_5`, `frac_logit_near_rail`.
   - `recurrent_vs_memoryless_delta(logits_rec, logits_mem, action_rec, action_mem)` — the Effect-on-Decision
     deltas: `recurrent_memoryless_logit_delta`, `recurrent_memoryless_action_delta`.
   - `raw_logit_l2_penalty(raw)` = `mean(raw²)`.

## Mechanism-Path Matrix delta
- separate small-range residual head: NOT_PRESENT → IMPLEMENTED (BeliefResidualActor); ACTIVE_IN_LOSS at R3.
- raw-logit L2 / saturation metrics: NOT_PRESENT → CALLABLE (this stage), ACTIVE_IN_LOSS at R3.
- all-frame standardization: NOT_PRESENT → CALLABLE.

## Failing-first tests (`tests/unit/test_belief_residual_R1_saturation.py`) — fail on HEAD (modules absent)
- `test_standardization_uses_all_train_frames` — all-frame standardization pools n_scenes×n_frames obs and the
  csi_age std differs from the frame-0-only std under stale CSI (frame 0 age≈0; later frames age grows).
- `test_stale_feature_scale_not_exploding` — after all-frame standardization the standardized stale features
  are bounded (p99 |z| below a sane cap), unlike frame-0-only on later frames.
- `test_saturation_metrics_logged` — `saturation_metrics` returns all required keys with correct values on a
  synthetic railed-vs-unsaturated input.
- `test_raw_logit_l2_reduces_saturation_on_pilot` — a tiny optimization pushing logits up saturates without
  raw-L2 (`raw_abs_mean` large) but stays small WITH raw-L2.
- `test_recurrent_changes_logits_under_stale_csi` — **Effect-on-Decision (load-bearing)**: BeliefResidualActor
  on a stale scene, frame≥1 residual logits with carried hidden ≠ zero hidden (not bit-identical), head
  unsaturated (`frac_logit_near_rail`≈0), and the decoded flip pattern CAN differ.

## Exit condition (R1 passes iff)
Saturation drops (the new head's `frac_logit_near_rail`≈0 vs the old head's 0.83) AND recurrent ≠ memoryless
residual logits under stale CSI. Else: pause control experiments (Workflow R1).

## Out of scope
Wiring the head into a trainer / PPO (R3); the belief auxiliary (R2). R1 builds the unsaturated separate head
+ metrics + standardization and proves the temporal signal now passes the head.
