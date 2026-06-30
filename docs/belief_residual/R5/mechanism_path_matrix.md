# R5 Mechanism-Path Matrix delta (Contract v4 §2)

| mechanism | before R5 | after R5 | path / evidence |
|---|---|---|---|
| edit_head (per-edge beneficial-edit logit) | NOT_PRESENT | **ACTIVE_IN_EVAL (R5)** — trained (L_edit BCE) AND the held top-k metric uses it | `BeliefResidualActor.edit_scores`; local features only; held precision − base = **+0.251 CI[+0.089,+0.413] random (CI>0)**, +0.184 [−0.005,+0.373] urban |
| repair_head (add/swap-in repair gain) | NOT_PRESENT | **ACTIVE_IN_LOSS (R5, L_repair Huber)**; held corr reported | supervised on R4 ΔD-reduction (adds); held repair_corr **+0.226 CI[+0.176,+0.277] random (CI>0)**, +0.178 [−0.085,+0.440] urban |
| safety_head (remove deletion risk) | NOT_PRESENT | **ACTIVE_IN_LOSS (R5, L_safety Huber)**; held corr reported | supervised on R4 ΔD-increase (removes); held safety_corr +0.105 random, +0.058 [+0.005,+0.111] urban |
| heads use LOCAL features only (deployable) | n/a | **enforced + verified** | `edit_scores(nf,ef,ei)` — no evaluator/true-CSI; R4 scores are labels only; untrained-head control at chance confirms the lift is from local-feature TRAINING, not the metric |

**ACTIVATION SUMMARY.** The heads are DEPLOYABLE (local-feature inputs; the deployed actor computes them with 0
evaluator calls) and now **DEMONSTRATED LEARNABLE**: trained on the R4 oracle-edit labels (training-only), they
rank held beneficial edits above the same-budget random baseline — decisively on random (CI entirely > 0, 3.2×
lift, untrained control at chance) and in mean on urban (4.5× lift, CI spans 0 by 0.005). This answers the
deployable-LEARNING question POSITIVELY: the R4 direction signal (R4 proved it EXISTS) is locally learnable.

**NEXT (R6).** Because the held ranking beats random, R6 promotes the heads from ACTIVE_IN_EVAL to the
candidate-gating path: the evidence-gated residual action samples only among edits passing the repair/safety
gate, zero candidates → anchor, budget-safe, local, 0 evaluator at deployment. R6 measures whether the learned
ranking converts into a deployed feasibility/return gain (the heads would then reach ACTIVE_IN_DEPLOY) — on BOTH
regimes, with the urban 95%-significance gap (n=5) carried forward as the open boundary. The R0 PPO-spy + R3
residual-PPO and the R1/R2 heads are unchanged.
