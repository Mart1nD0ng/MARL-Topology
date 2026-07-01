# R8 Mechanism-Path Matrix delta (Contract v4 §2)

| mechanism | before R8 | after R8 | path / evidence |
|---|---|---|---|
| stale-CSI observation overlay (delay-1) | **OFF in R1–R7** (current channel) | **ACTIVE in the DEPLOYED decision observation (R8)** — the deployed actor observes stale CSI (cols 0–3 = frame t−1; +[age, mask]) AT DECISION time; the evaluator stays on the TRUE current channel | `build_csi_scenes(..., CsiObservationModel(mode='delay', delay_frames=1))`; test: CSI cols diverge at t≥1, evaluator identical |
| evaluator on the TRUE current channel (leak-free) | invariant | **enforced + verified** | `context.evaluator` untouched by the overlay; test: reliability identical stale-vs-current |
| edit/repair/safety heads retrained on STALE features | trained on current features (R5) | **ACTIVE_IN_DEPLOY (R8)** — heads see stale ef; labels from the TRUE evaluator (training-only) | `train_edit_heads(stale_train, ...)`; deployed via `evidence_gated_residual` (0 eval) |
| evidence-gated action (R6) | ACTIVE_IN_DEPLOY on current channel (== anchor) | **reused under stale CSI** — same gate, 0 eval; the question is whether it repairs the degraded anchor | `evidence_gated_residual` on stale scenes |

**ACTIVATION SUMMARY.** R8 changes ONE variable vs R6 — the observation regime (current → stale delay-1) — and
retrains the heads for that regime. This is the campaign's PREMISE: R6/R7 showed no gain on the current channel
(near-optimal anchor, no room); R8 tests whether, when stale CSI DEGRADES the anchor (creating room), the
evidence-gated action repairs the drop. Leak-free by construction (actor observes stale CSI; the evaluator
scores on the true current channel — verified). Deployment stays decentralized (0 eval at decision). Result
(5-seed: current_anchor / stale_anchor / stale_gated; the (gated − stale_anchor) feasibility CI is the decision)
→ `decision.md`. A (gated − stale_anchor) CI > 0 would be the campaign's FIRST genuine deployable positive.
