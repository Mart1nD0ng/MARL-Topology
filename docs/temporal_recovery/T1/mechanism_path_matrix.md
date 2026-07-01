# T1 — Mechanism-Path Matrix (Contract v4 §2)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | code_path | status @ T1 | evidence |
|---|---|---|---|
| CSI-recovery → anchor injection (oracle) | `t1_oracle_recovery_gen._anchor_with_psucc` | ACTIVE_IN_EVAL | overwrites psucc col 0/1, runs `_anchor`, true evaluator scores; test `test_recovered_psucc_injects_into_anchor` |
| arm A floor (stale psucc) | `_eval_arms` | ACTIVE_IN_EVAL | = R8 stale_anchor; feasibility random 0.32 / urban 0.65 |
| arm D ceiling (true psucc) | `_eval_arms` + `_true_psucc` | ACTIVE_IN_EVAL | = R8 current_anchor; feasibility random 0.375 / urban 0.815; ceiling CI>0 |
| arm B direct predictor (absolute `p_t`) | `_train_predictor(mode=direct)` | ACTIVE_IN_LOSS + ACTIVE_IN_EVAL | MSE(sigmoid(net), true) training-only; B−A CI<0 (harmful) |
| arm C physics-residual predictor (correction) | `_train_predictor(mode=physics)` | ACTIVE_IN_LOSS + ACTIVE_IN_EVAL | MSE(clamp(stale+net), true); C−B CI>0; C−A spans 0 |
| leak-free features (stale CSI + CURRENT geometry) | `_edge_features` | ACTIVE_IN_EVAL | stale cols 0/2/3 + current distance col 5 + rel_vel/Δdist/age; NO true CSI (leak test) |
| true current psucc = training-only label | `_true_psucc` → `belief_target_logits` | ACTIVE_IN_LOSS (label only) | leak test: `belief_target_logits` called 0× in inference, 1× in label path |
| stale/partial CSI overlay | `CsiObservationModel(delay,1)` | ACTIVE_IN_EVAL | delay-1; evaluator on true channel (leak-free, R8 construction) |

**No deployed-path change** — T1 is a diagnostic generator under `scripts/diagnostics/` + a load-bearing test.
The anchor and evaluator are the existing deployable/true components; T1 only varies the psucc fed to the
anchor. The predictors are training-only reference probes (the deployed temporal module is built at T3–T5).
