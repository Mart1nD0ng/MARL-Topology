# T3 — decision (belief correction target; task 3.1/3.2)

**Disposition: KEEP the correction parametrization (marginally better structure) / HONEST NEGATIVE on
CSI-magnitude recovery + anchor conversion.** The correction target reparametrizes the belief so echo is the
zero-baseline and the head learns the SIGNED stale→current delta. It captures DIRECTION well and is marginally
better than the absolute (R2) target — but it does NOT beat the stale-echo floor nor improve the anchor. The
binding limit is MAGNITUDE/precision recovery (direction is learnable, magnitude is not), confirming T1 at the
belief level.

## Result (5 seeds × {random, urban}, delay-1, in-policy belief; `correction_belief_metrics.json`)

| metric | random | urban | reading |
|---|---|---|---|
| belief MSE (correction) vs floor | 0.09501 vs 0.09506 | 0.10263 vs 0.10298 | **correction ≈ floor** (within 0.0005) |
| `cor_beats_floor` | +5e-05 [1e-05,1e-4] | +3.5e-4 [−2e-4,+9e-4] | random CI>0 but **practically nil** (0.05% of floor); urban spans 0 |
| `abs_beats_floor` | −4.7e-4 [−1e-3,+6e-5] | −2e-4 [−5e-4,+8e-5] | absolute is at/below the floor (R2 negative) |
| `correction_vs_abs_mse` | +5.2e-4 [−0,+1e-3] | +5.4e-4 [+5e-5,+1e-3] | correction **marginally better** than absolute (sig urban) |
| `dir_acc` correction / absolute | 0.932 / 0.923 | 0.914 / 0.930 | **direction captured** by both (≫ chance 0.5) |
| `cor_feas_gain` (recovered→anchor) | −0.219 [−0.494,+0.057] | −0.094 [−0.355,+0.168] | spans 0, **mean negative** — no conversion |
| `abs_feas_gain` | **−0.1875 [−0.249,−0.126]** | −0.10 [−0.291,+0.091] | absolute **significantly hurts** the anchor (random) |

Plus a **move-weight sub-pilot** (1 seed, `move_weight=3` on `|correction_target|`): MSE 0.10→0.35, feas −0.25 —
aggressive magnitude emphasis **overshoots**. With shrink-to-echo at weight 0 and overshoot at weight 3, there is
no magnitude sweet spot → the magnitude is not robustly recoverable from the leak-free features.

## What T3 establishes (each a Claim Card)
1. **The correction parametrization is the better structure** (task 3.1): echo is the zero-baseline (no longer a
   global optimum), and correction MSE < absolute MSE (sig on urban). But the improvement is TINY — both sit at
   the stale-echo floor.
2. **Direction is recoverable; magnitude is not.** `dir_acc ≈ 0.92` (both arms) — the belief knows *which way*
   the channel moved (task 3.2, from the leak-free geometry). But the *magnitude* shrinks to echo (MSE at floor)
   or overshoots (move-weight backfires) → recovered psucc ≈ stale (or noisier).
3. **The belief-recovered psucc does NOT improve the anchor** — `feas_gain` spans 0 (correction) or is
   significantly negative (absolute, random). Feeding the anchor a learned psucc PERTURBS the ranking it sorts on
   without recovering the decision-critical magnitude → stale is a better anchor input. This is T1's lesson
   (MSE ⟂ decisions) confirmed for the in-policy belief.

## Honest scope / caveats (Contract v4 §5/§14)
- This is a NEGATIVE on the central T3 question ("does the correction target make the belief recover CSI and
  improve the anchor?") — NO. It reproduces R2's floor result under a better parametrization, and confirms T1's
  realizable-recovery limit (arm C, C−A spans 0) at the in-policy belief level.
- The `cor_beats_floor` random CI is technically >0 but the effect (5e-05) is negligible; NOT a recovery claim.
- Direction accuracy is measured on *moved* edges (|correction_target|>0.1 logit); it is high because geometry
  (current distance, velocity — leak-free) predicts the sign of the channel change well, but not its size.
- The magnitude limit is consistent across mechanisms (T1 standalone regressor, T3 in-policy belief), suggesting
  it is a feature/observability limit (the decision-critical fast-fading residual is not in the leak-free
  features), not just a node-GRU limit — but T4 (edge recurrence) and T5 (uncertainty) remain distinct untested
  levers the user requested.

## Effect on the campaign plan
- **KEEP** the correction primitives (`stale_logit`, `belief_correction_target`, `directional_accuracy`) — the
  better parametrization T4/T5 build on.
- **T4 (edge-level recurrence):** test whether the per-node GRU is a *magnitude* bottleneck (task 3.4) — the CSI
  dynamics are edge attributes; an edge-level recurrent state may capture magnitude the node state cannot.
- **T5 (uncertainty):** the magnitude is UNCERTAIN — predict its variance and shrink the correction by
  confidence, so the anchor is fed a calibrated psucc (only commit magnitude where confident). This directly
  targets the shrink-vs-overshoot failure T3 exposed.

## Verification (Ultracode adversarial Workflow `w569gv274`)
4-lens + synthesis. **Synthesis: PASS, 0 MAJOR — clear to commit.** leak-free **PASS** (belief input reads only
stale/geometry/GRU; the true current psucc enters ONLY as a training label + metric scorers; the `move_weight`
loss-reweight is a legitimate training-only use, inert in the 5-seed main); statistical-honesty **PASS** (the
negligible random `cor_beats_floor` +5e-05 is doubly flagged as "NOT a recovery claim"; all CIs match the
artifact); single-variable **PASS** (absolute vs correction differ only in `belief_logit = head` vs
`stale_logit + head`, same init/GRU/features/loss/epochs); interpretation **PASS** (direction-recoverable /
magnitude-not is supported; the move-weight backfire consistently caveated n=1). 1 MINOR applied: a leak-boundary
comment in `_held` marking the true-CSI/evaluator as metric-only.

## Next
**T4 — edge-level recurrent state:** add an edge-indexed recurrent hidden (the CSI dynamics are per-edge), test
whether it recovers magnitude the per-node GRU cannot. One variable vs the T3 correction belief.
