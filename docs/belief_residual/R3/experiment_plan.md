# R3 — experiment plan (residual PPO + CTDE value critic)

## Hypothesis (single variable: the residual TRAINER — REINFORCE → PPO+critic)
Q14's clamp/collapse came from a high-variance REINFORCE trainer (scalar moving baseline, fixed flip penalty,
no PPO/critic/entropy). If the residual trainer is replaced by **genuine Residual PPO** (per-edge PPO clip +
adaptive KL early-stop) with a **CTDE value critic** (`A_t=G_t−V_φ`) + entropy + the R1 raw-L2 on the
unsaturated head, the bimodal clamp/collapse should reduce and the optimization metrics (ratio≈1 at epoch 0,
sane approx_kl/clip_fraction, non-trivial critic EV) should be healthy. This is the training-stability chain —
now the campaign's main lever (R1 temporal + R2 belief are confirmed non-load-bearing). NOTE (Contract v4 §5):
R3 is the TRAINER; beneficial-edit direction supervision is R4–R5. If PPO-residual still == anchor, R3 must
say whether that is a trainer problem or a missing-direction-signal problem.

## Single change (a NEW residual PPO trainer; the REINFORCE trainer is retained for the A/B)
`scripts/diagnostics/residual_ppo_train.py` — the canonical Residual PPO trainer:
1. rollout (no grad): per (scene, frame) `sample_residual` → store decisions, candidate_mask, **per-edge**
   `logp_old`, reward, anchor, and `value_old` from the critic.
2. PPO epochs: recompute per-edge `logp_new` for the SAME decisions; `ratio = exp(logp_new − logp_old)`;
   **call `graph_mappo.ppo_clip_actor_loss`** with the per-edge (per-agent) arrays + advantage repeated per
   candidate edge (NOT a joint per-frame ratio). Log `ratio`/`clip_fraction`/`approx_kl`; **`target_kl`
   early-stops** the inner epochs.
3. CTDE critic `ResidualValueCritic` (training-only; reads a global state summary incl. true-CSI summary —
   never used at deployment): `A_t=G_t−V_φ`; critic MSE; report `critic_parameter_delta`, `value_loss`, EV.
4. entropy bonus; raw-L2 (R1) ACTIVE; `L_CSI` ablatable (default off — R2 no-op, interface kept).
5. New `residual_logp_per_edge` (additive to `residual_action.py`) — the per-edge logp vector for the
   per-agent ratio (avoids the joint-ratio variance explosion the PPO docstring warns about).

## R0 tripwire resolution
The R0 spy `test_residual_update_does_not_call_ppo_clip_spy` (on the REINFORCE `residual_pbrs_train`) stays
TRUE (that trainer is the retained A/B baseline; it is still PPO-free). The "flip" is realized by the NEW
`test_residual_ppo_calls_ppo_clip_actor_loss` on `residual_ppo_train` (the residual PATH now has PPO).

## Mechanism-Path Matrix delta
- residual PPO clip (`ppo_clip_actor_loss`): NOT_PRESENT (residual path) → **ACTIVE_IN_LOSS** (residual_ppo_train).
- approx_kl / clip_fraction / target_kl: → logged + target_kl early-stop ACTIVE.
- CTDE value critic: NOT_PRESENT (residual path) → **ACTIVE_IN_LOSS** (critic_parameter_delta>0).
- entropy bonus: → ACTIVE_IN_LOSS. raw-L2: → ACTIVE_IN_LOSS. L_CSI: ablatable (default off).

## Failing-first tests (`tests/unit/test_belief_residual_R3_ppo.py`) — fail on HEAD
- `test_residual_ppo_calls_ppo_clip_actor_loss` — **load-bearing spy**: a residual PPO update CALLS
  `graph_mappo.ppo_clip_actor_loss` (≥1 call).
- `test_residual_ppo_logs_approx_kl` — the trainer returns `approx_kl`/`clip_fraction` in its metrics.
- `test_residual_ppo_epoch0_ratio_one` — at inner epoch 0 (logp_new==logp_old) ratio==1, approx_kl≈0.
- `test_target_kl_can_stop_inner_epochs` — a tiny `target_kl` stops the inner epochs early (fewer than max).
- `test_residual_critic_parameter_delta_positive` — the critic params change after an update (it trains).
- `test_entropy_bonus_present` — the entropy term is computed + enters the loss (entropy_coef changes the loss).
- `test_per_edge_logp_not_joint` — `residual_logp_per_edge` returns a per-edge vector summing to `residual_logp`.

## Exit condition (R3 passes iff)
Tests pass; the trainer genuinely calls PPO + critic (spy + critic_param_delta>0); epoch-0 ratio==1; target_kl
stops early; pilot shows healthy KL/clip and a critic EV that is non-negative or explained. The clamp/collapse
A/B (PPO vs REINFORCE) is REPORTED (reduced collapse is a positive; if PPO still == anchor, report it as a
missing-direction-signal finding deferred to R4–R5, NOT a PPO failure).

## Out of scope
Beneficial-edit supervision (R4–R5); evidence-gating (R6); adaptive anchor KL schedule (R7). R3 builds + proves
the PPO+critic trainer and reports the clamp/collapse A/B.
