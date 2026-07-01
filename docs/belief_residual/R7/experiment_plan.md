# R7 — experiment plan: adaptive anchor-KL / safety constraint (replace the fixed flip penalty)

## One variable (Contract v4 §15)
**The anchor-pull: FIXED → ADAPTIVE.** R3's residual PPO used a FIXED anchor pull (`residual_prior=-1.0`
constant bias toward "don't flip"), and Q9/Q14 used a FIXED flip-count penalty — the sole anchor protection.
R7 replaces that with an ADAPTIVE anchor-KL coefficient driven by retention/validation. Everything else (the R3
PPO clip, per-edge ratios, target_kl new-vs-old trust region, CTDE critic, entropy, raw-L2) is unchanged.

## Mechanism
Add to the actor loss an **anchor-KL** term = the expected flip mass (deviation from the flip-nothing anchor):
`anchor_kl = mean_over_candidate_edges sigmoid(z_i)` (z = residual logit; sigmoid(z_i) = P(flip edge i); the
anchor policy is "flip nothing", so this is the KL-like deviation-from-anchor penalty). Actor loss:
`ppo_loss − entropy_coef·entropy + raw_l2·raw_pen + beta_anchor·anchor_kl`.
**Adaptive controller** (replaces the fixed penalty; the R7 variable):
- target_retention τ_ret (e.g. 0.95, anchor-edge retention on the eval decode).
- after each epoch: if retention < τ_ret → `beta_anchor *= 1.5` (TIGHTEN — pull back to the anchor); elif
  retention ≥ τ_ret AND validation return improved over the last epoch → `beta_anchor *= 0.7` (LOOSEN — allow
  exploration); clamp beta_anchor ∈ [beta_min, beta_max].
- PPO's target_kl (new-vs-old) stays as the stability trust region (R3, unchanged).

## Expected outcome (stated up front, per Contract v4 honesty — but TESTED, not assumed)
The R6 threshold sweep proved the optimal anchor-deviation is ZERO (no gate operating point beats the anchor).
So the adaptive controller is EXPECTED to tighten toward the anchor → edit_rate ~0, retention ~1.0, residual ==
anchor — a CONFIRMATORY negative ("even an adaptive/learned trust region lands at the anchor, not just the fixed
flip penalty"). R7 is run to CLOSE the "but did you try adaptive, not fixed?" objection (Contract v4: test the
ACTUAL mechanism, do not assume its outcome from R6). If — surprisingly — adaptive loosening finds a
retention-stable operating point with val-return > anchor, that would be a genuine positive (revise the R6
conclusion); the sweep makes this unlikely.

## Failing-first load-bearing tests (Contract v4 §3)
`tests/unit/test_belief_residual_R7_adaptive_kl.py` (fail on HEAD: the adaptive trainer / anchor-KL absent):
1. `test_anchor_kl_in_actor_loss` — the anchor-KL term is computed from sigmoid(z) over candidate edges AND
   enters the actor loss (grad to the actor's residual head is non-zero and increases with beta_anchor). Spy.
2. `test_beta_tightens_when_retention_drops` — feed a low-retention epoch → the controller INCREASES beta_anchor;
   feed a stable-high-retention + improving-return epoch → it DECREASES beta_anchor. (The adaptive law is
   load-bearing, not a constant.)
3. `test_adaptive_differs_from_fixed_only_in_anchor_pull` — the adaptive trainer shares the R3 PPO/critic/entropy
   path; the ONLY structural difference is the anchor-KL coefficient schedule (one-variable audit).
4. `test_adaptive_is_deployable_and_no_flip_penalty` — the eval decode is the same deployed MAP decode (0 eval);
   the fixed flip-count penalty is NOT the sole protection (the adaptive anchor-KL is present).

## Measurement (pilot → 5-seed A/B; Contract v4 §15)
Arms (same actor/rollout/scenes; only the anchor-pull differs):
- **R3-fixed** (residual_prior=−1.0, no adaptive KL) — the R3 baseline;
- **R7-adaptive** (adaptive anchor-KL, mild/zero fixed prior).
Per-arm: residual_feasibility vs anchor_feasibility, edit_rate, zero_edit_rate, retention, beta_anchor
trajectory, approx_kl/clip_fraction, EV, diverged. 5 seeds × {random, urban}; CI on (adaptive − fixed) feas and
on (adaptive residual − anchor) feas.

## Exit condition (Contract v4 §15)
- Adaptive residual feasibility CI **> anchor** on a regime → adaptive trust region BEATS the anchor → KEEP,
  revise R6 (a positive). (Unlikely per the R6 sweep.)
- Adaptive == anchor (edit_rate ~0, retention ~1, CI spans 0) → CONFIRMATORY negative: even an adaptive anchor
  trust region lands at the anchor; the fixed-penalty result was NOT the reason. KEEP the mechanism; the
  campaign's binding limit (deployable precision, R6) stands. → R8 (stale-CSI full pilot) or close-out.
- Adaptive UNSTABLE (retention collapses despite tightening) → REVISE the controller (report the failure).

## Hard constraints
Deployment decentralized (eval MAP decode, 0 eval); true C/E/L on the current real channel + closed-form quorum
tail; the critic/true-CSI are training-only; only anchor-relative LOCAL edits; failing-test-first; report whether
adaptive-KL changes the final edit_rate/retention (not just forward); ≥5 seeds + CI; commit to trunk, no push.
