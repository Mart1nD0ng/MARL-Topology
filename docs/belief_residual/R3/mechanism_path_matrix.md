# R3 Mechanism-Path Matrix delta (Contract v4 §2)

| mechanism | before R3 | after R3 | path / evidence |
|---|---|---|---|
| PPO clip (`ppo_clip_actor_loss`) on the residual path | NOT_PRESENT | **ACTIVE_IN_LOSS** (`residual_ppo_train`) | spy test `test_residual_ppo_calls_ppo_clip_actor_loss` (≥1 call) |
| approx_kl / clip_fraction | NOT_PRESENT (residual) | **logged** | `train_residual_ppo` returns them; epoch-0 ratio==1 ⇒ approx_kl 0 |
| target_kl early-stop | NOT_PRESENT | **ACTIVE** | `test_target_kl_can_stop_inner_epochs` |
| per-edge (per-agent) ratio | NOT_PRESENT | **ACTIVE** | `residual_logp_per_edge` (sums to joint, fed per-edge to PPO) |
| CTDE value critic | NOT_PRESENT (scalar moving baseline) | **ACTIVE_IN_LOSS** (EV ~0.56) | `ResidualValueCritic` (LayerNorm-bounded); `critic_parameter_delta>0` |
| entropy bonus | NOT_PRESENT (residual) | **ACTIVE_IN_LOSS** | `_entropy`; `test_entropy_bonus_present` |
| raw-logit L2 (R1) | CALLABLE | **ACTIVE_IN_LOSS** (residual PPO) | in `actor_loss` |
| L_CSI (R2) | ACTIVE_IN_LOSS (belief trainer) | ablatable (default OFF in R3) | R2 no-op; interface kept |
| residual trainer optimizer | REINFORCE only | **PPO + critic** (new canonical trainer) | `residual_ppo_train`; REINFORCE retained as A/B control |

R0 tripwire: the spy on `residual_pbrs_train` (REINFORCE) still passes (it remains the PPO-free A/B baseline);
the NEW `test_residual_ppo_calls_ppo_clip_actor_loss` is the realized flip on the residual PPO path. Per
Contract v4 §7, the trunk's PPO is NOT cited — `residual_ppo_train` calls `ppo_clip_actor_loss` itself.

Scheduled: beneficial-edit supervision (R4–R5 — the missing direction signal that R3 shows the trainer lacks),
evidence-gating (R6), adaptive anchor-KL (R7).
