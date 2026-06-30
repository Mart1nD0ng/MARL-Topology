# R5 — experiment plan (repair / safety / edit heads) — the DEPLOYABLE-LEARNING crux

## Hypothesis (single variable: supervised heads on local features)
R4 proved a beneficial-edit signal EXISTS over the anchor (true-evaluator-scored). R5 asks the decisive
deployable question: can heads trained ONLY on LOCAL features (the deployed actor's `nf`/`ef` graph encoding +
anchor membership — NO evaluator, NO true CSI) predict which edits are beneficial, beating a random baseline on
held? If yes → a deployable direction signal exists → R6. **If the held top-k edit-hit-rate ≤ random, local
observations are insufficient to predict the central beneficial edits → STOP before PPO: a path-specific
deployable LEARNING gap (the campaign's honest final diagnosis at N≤16).**

## Single change (additive heads on the BeliefResidualActor; supervised on R4)
- Add `edit_head` (per-edge beneficial-edit logit), `repair_head` (add/swap-in repair gain), `safety_head`
  (remove/swap-out deletion risk) to `BeliefResidualActor` — each a small MLP on the SAME per-edge local
  features `[ef, h_u⊙h_v, |h_u−h_v|]` the residual head uses (graph-encoded local features only).
- New `src/marl_topology/training/edit_head_training.py`:
  - `build_edit_examples(scenes, T, margin)` — per frame: the obs features + anchor + per-candidate-edge
    targets `{positive (ΔJ>margin ∧ safe), repair_gain = −ΔD_quorum (adds), deletion_risk = ΔD_quorum
    (removes)}` from the R4 evaluator scoring (the TRAINING LABEL; the head input never sees it).
  - `train_edit_heads(train_scenes, T, ...)` — `L = L_edit (BCE on positive) + L_repair (Huber, adds) +
    L_safety (Huber, removes)`; held eval on a SEPARATE oracle-edit dataset (held never in training).
  - `topk_hit_rate(head_scores, labels, k)` + `random_baseline` (same-budget random edit selection).

## Mechanism-Path Matrix delta
- repair/safety/edit heads: NOT_PRESENT → ACTIVE_IN_LOSS (R5 supervised trainer). Heads use LOCAL features
  only (deployable); the R4 evaluator targets are labels, never head inputs.

## Failing-first tests (`tests/unit/test_belief_residual_R5_heads.py`) — fail on HEAD
- `test_heads_use_only_local_features` — the head forward takes `(nf, ef, ei)` (the deployed local obs) and
  has NO access to true CSI / the evaluator (audit: `edit_scores` signature + no evaluator import in the path).
- `test_repair_head_predicts_delta_D` / `test_safety_head_predicts_deletion_risk` — after training, the
  head predictions correlate with the R4 ΔD targets (held corr or MSE drop vs untrained).
- `test_edit_head_topk_hits_positive_edits` — on held, the edit head's top-k precision is defined + computed.
- `test_supervised_edit_improves_over_random` — held top-k precision (trained) ≥ random base-rate (the load-
  bearing exit metric; reported with the honest sign even if it FAILS).
- `test_heads_dataset_uses_train_only` — `build_edit_examples` / `train_edit_heads` take train scenes; held
  built separately for eval.

## Exit condition / decision rule (≥5 seeds, CI)
- held top-k edit-hit-rate (precision@k) **significantly > random base rate** (paired CI > 0) → KEEP → R6.
- held hit-rate ≤ random (CI spans/below 0) → **STOP**: path-specific NEGATIVE — local features cannot predict
  the central beneficial edits at N≤16. The belief-guided residual-PPO route is then an honest negative (all
  four chains: temporal ✗ / belief ✗ / trainer ✓-but-no-direction / direction-signal exists-but-not-locally-
  learnable). This is the campaign's final, precise diagnosis — NOT "the anchor wins".

## Out of scope
Evidence-gating deployment (R6); the residual PPO with the gated heads (R7–R8). R5 only tests local
learnability of the beneficial-edit signal.
