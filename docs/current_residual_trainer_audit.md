# Residual trainer path audit (R0, Belief-Guided Residual PPO) — HEAD a634928

Named deliverable of Workflow R0. Verified by `tests/unit/test_belief_residual_R0_audit.py` (3 pass) +
`result_save/belief_residual/R0/mechanism_activation.json`. Full matrix: `docs/CURRENT_BELIEF_RESIDUAL_STATUS.md` §1.

## The audited residual path: `scripts/diagnostics/residual_pbrs_train.py` (Q9/Q11), reused by `stale_csi_residual_joint.py` (Q14)

| question (Workflow R0) | answer | evidence |
|---|---|---|
| residual trainer uses REINFORCE? | **YES** | `reinforce_update`: `loss = -Σ logp_t·(G_t − baseline) + anchor_reg·mean(#flips)` (L91) |
| residual trainer calls `ppo_clip_actor_loss`? | **NO** | source has no reference; spy test = 0 calls during a real update |
| residual trainer uses a centralized value critic? | **NO** | scalar moving-average baseline `0.9·b + 0.1·mean(G)` (L118); no critic, no `V(s_t)` |
| residual trainer logs KL? | **NO** | no `approx_kl` / `clip_fraction` / `target_kl` |
| residual trainer logs entropy? | **NO** | no entropy term; only `grad_norm` via `clip_grad_norm_` |
| CSI-belief auxiliary in the residual loss? | **NO** | no `belief_head` / `L_CSI` anywhere; CSI predictor is diagnostic-only (`csi_prediction_health.py`) |
| trust region | **fixed flip penalty** (`anchor_reg`) | penalizes ALL flips equally (no repair/safety/utility distinction) |
| logit head | **shared ±10 tanh** (saturates) | `DynamicRecurrentActor`: `logit = 10·tanh(raw/10)`; trained `raw`≈3000 → 83% railed (Q14) |

## The Graph-MAPPO trunk (a DIFFERENT path) — where PPO actually lives
`ppo_clip_actor_loss` (`graph_mappo.py:79`) + `approx_kl` (`:108`) + `clip_fraction` (`:102`) are
ACTIVE_IN_LOSS in the static/dynamic Graph-MAPPO trunk (called via `dynamic_rl.py` / `train_decentralized_rl.py`).
**Per Contract v4 §7 (No Substitution): this does NOT count as residual PPO.** The residual path must build
and prove PPO on its own (R3).

## Conclusion (the re-scope)
Q14's negative is path-specific: the REINFORCE residual trainer + fixed flip-penalty + shared saturating head
failed under delay1 urban. Residual PPO, a CTDE critic, a CSI-belief auxiliary, evidence-gated edits, and
beneficial-edit supervision are **NOT_PRESENT** in the residual path and were **never tested** — they are the
build targets of R1–R7. No "residual learning failed" claim is permitted until R1–R5 pass (Workflow §0).
