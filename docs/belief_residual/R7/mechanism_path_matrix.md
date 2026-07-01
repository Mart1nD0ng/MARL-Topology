# R7 Mechanism-Path Matrix delta (Contract v4 §2)

| mechanism | before R7 | after R7 (target) | path / evidence |
|---|---|---|---|
| anchor pull (protection against leaving the anchor) | FIXED (R3 `residual_prior=−1.0` constant; Q14 fixed flip-count penalty) | **ADAPTIVE anchor-KL (R7)** — `beta_anchor·mean_candidates σ(z)` in the actor loss, `beta_anchor` tightens on retention-drop / loosens on val-improve | new adaptive path in `residual_ppo_train.py`; spy the term is in the loss + the controller adjusts beta |
| adaptive controller (retention/val driven) | NOT_PRESENT | **ACTIVE_IN_LOSS (R7)** — beta_up 1.5 on retention<τ, beta_down 0.7 on stable+improving | `test_beta_tightens_when_retention_drops` |
| R3 PPO clip / per-edge ratio / target_kl / CTDE critic / entropy / raw-L2 | ACTIVE_IN_LOSS (R3) | unchanged (R7 changes ONLY the anchor pull) | one-variable audit test |
| fixed flip-count penalty as SOLE protection | (Q14) | **removed** — the adaptive anchor-KL is the protection | `test_adaptive_is_deployable_and_no_flip_penalty` |

**ACTIVATION SUMMARY (target).** R7's single new mechanism is the ADAPTIVE anchor-KL coefficient (vs R3's fixed
anchor pull). It answers the "did you try adaptive, not fixed?" objection left open by R3/R6. The R6 threshold
sweep already showed the optimal anchor-deviation is ZERO, so the EXPECTED outcome is that the adaptive
controller tightens to == anchor (edit_rate ~0, retention ~1) — a CONFIRMATORY negative — but per Contract v4
the actual mechanism is TESTED, not assumed. Deployment stays decentralized (MAP decode, 0 eval); the critic /
true-CSI are training-only; only anchor-relative LOCAL edits. Result (5-seed adaptive vs fixed vs anchor) →
`decision.md`.
