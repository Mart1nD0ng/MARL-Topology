# T5 — Mechanism-Path Matrix (Contract v4 §2)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | code_path | status @ T5 | evidence |
|---|---|---|---|
| heteroscedastic uncertainty head (`belief_logvar_head`, `belief_with_uncertainty`) | `models/belief_residual_actor.py` | ACTIVE_IN_LOSS (opt-in; campaign belief_uncertainty=True) | mu==belief(); logvar extra output; test |
| Gaussian NLL correction loss (`csi_correction_nll`) | `training/csi_belief.py` | ACTIVE_IN_LOSS | 0.5(exp(−logvar)se+logvar); heteroscedastic (test) |
| confidence gate + gated correction (`confidence_gate`, `gated_correction`) | `training/csi_belief.py` | ACTIVE_IN_EVAL | gate=sigmoid(−logvar); recovery = stale + gate·mu |
| uncertainty calibration diagnostic (`uncertainty_calibration`) | `training/csi_belief.py` | ACTIVE_IN_EVAL | corr(logvar, se); weak (0.06/0.18) |
| recovered-psucc → anchor (T1 injection) | `t1_oracle_recovery_gen._anchor_with_psucc` (reused) | ACTIVE_IN_EVAL | gated feas_gain spans 0, negative mean |
| belief_uncertainty=False default (back-compat) | `models/belief_residual_actor.py` | ACTIVE (default) | logvar head appended last; full state_dict preserved; prior suite byte-identical |

**Blast radius: zero at the default.** `belief_uncertainty=False` allocates no logvar head and does not change
`belief()`/`forward()`. `csi_belief.py` gains 4 pure functions (no change to existing functions).
`models/`/`training/` are gate-exempt (CTDE).
