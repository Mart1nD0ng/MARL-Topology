# T1 — decision (oracle upper-bound gate)

**Disposition: KEEP / GATE PASS → proceed to the model/architecture redesign (T2–T5).** NOT the env-feature
fallback: the fallback ("improve env temporal hidden features") is only warranted if *perfect* recovery also
failed. It did not — perfect **(oracle/clairvoyant)** recovery converts significantly. The bottleneck is the
learned model's architecture, exactly what T2–T5 fix.

## Result (5 seeds × {random, urban}, delay-1 stale, true evaluator, 0-eval; `oracle_recovery_metrics.json`)

| metric | random | urban | reading |
|---|---|---|---|
| ceiling `D−A` (perfect recovery) | **+0.055 [+0.009,+0.101]** | **+0.165 [+0.118,+0.212]** | **CI>0 both** — perfect recovery converts; reproduces R8 exactly |
| realizable `C−A` (physics-residual) | −0.04 [−0.140,+0.060] | +0.025 [−0.041,+0.091] | spans 0 — crude geometry-MLP does not yet convert |
| model `B−A` (direct-neural) | **−0.25 [−0.477,−0.023]** | **−0.25 [−0.326,−0.174]** | **CI<0 both** — direct absolute-`p_t` prediction actively HARMS |
| arch `C−B` (correction vs direct) | **+0.21 [+0.036,+0.384]** | **+0.275 [+0.165,+0.385]** | **CI>0 both** — the correction/physics structure is essential |
| MSE echo / B / C | 0.102 / 0.085 / 0.093 | 0.109 / 0.081 / 0.085 | both predictors beat echo MSE, yet B (best MSE) has worst feasibility |

## What T1 establishes (each a Claim Card)
1. **The recoverable information exists and CONVERTS at the oracle level.** Feeding the anchor the *true*
   current psucc recovers the full stale drop (ceiling CI>0 both regimes, = R8). The oracle is validated (it
   reproduces R8's `stale_drop` sign and scale exactly), so the design is sound (task-1 "check oracle" ✓).
2. **The current architecture (direct absolute-`p_t`) is the failure — and it is actively harmful.** `B−A` is
   significantly negative on both regimes. This is *why* R2 echoed / failed. **Inferred mechanism** (not
   directly measured — see caveat): a direct predictor minimizes MSE by regressing toward the conditional
   mean, which compresses the psucc distribution and degrades the ranking the anchor sorts on near its
   thresholds; the evidence is the MSE-vs-feasibility inversion (B has the best MSE yet the worst feasibility),
   not a measured ranking correlation. Validates task 3.1 as a NEGATIVE control.
3. **The correction/physics structure is the right direction.** `C−B` is significantly positive on both
   regimes: predicting a correction on the stale value (echo = zero-baseline) instead of the absolute value
   flips the arm from significantly-harmful to neutral. Validates task 3.1 (correction target) + 4.2 (physical
   residual model).
4. **MSE is the wrong objective; decision/ranking is the right one.** B has the best MSE but the worst
   feasibility. T3–T5 must optimize the anchor-relevant ranking near the keep/add thresholds, not psucc MSE.

## Honest scope / caveats (Contract v4 §5/§14)
- The realizable arm `C−A` is a crude STANDALONE geometry-MLP and is NOT significant (spans 0). T1 does NOT
  prove a realizable predictor beats the stale anchor — it proves (a) perfect recovery would, (b) the current
  direct architecture is harmful, and (c) the correction structure is necessary. Whether the *in-policy*
  temporal module (T3 correction target + T4 edge recurrence + T5 uncertainty) converts the marginal `C−A`
  into a significant gain is the open question T3/T6 test.
- The gap between the oracle ceiling (+0.165 urban) and the realizable arm (+0.025) is a **decision-boundary
  precision** gap: geometry recovers the mean channel but not the decision-critical residual (fast fading) at
  the edges near the anchor's thresholds. This is the same binding limit the prior campaign found (R6), now
  localized to CSI recovery — a real risk the redesign must beat, not assume away.
- This is a CSI-recovery-feeding-the-anchor test, NOT a learned policy beating the anchor. Distinct from the
  prior anticipation (forecasting) oracle (+0.000).

## Effect on the campaign plan
- **T2 (activation):** proceed — cheap, clearly correct (the tanh saturation is orthogonal and pinned at T0).
- **T3 (belief target → correction):** proceed — DIRECTLY validated by T1 (`C−B` CI>0; `B−A` CI<0). This is now
  the highest-value stage. Its acceptance metric is decision/ranking (held top-k / anchor feasibility under
  recovered psucc), NOT MSE.
- **T4 (edge recurrence) / T5 (physics + uncertainty):** proceed — candidates to close the decision-boundary
  precision gap that keeps the realizable arm marginal.
- **Env temporal-hidden-feature fallback:** NOT triggered (perfect recovery succeeded). Held in reserve if
  T3/T6 in-policy recovery also stalls at the decision-boundary precision limit.

## Verification (Ultracode adversarial Workflow `w3fx2yp7j`)
4-lens refutation + synthesis. **Synthesis: PASS, 0 MAJOR.** leak-free **PASS** (true current psucc enters only
as a training label, 0 inference reads spy-verified; current distance is legitimately leak-free geometry);
oracle-validity **PASS** (clean single-variable psucc-only manipulation; ceiling reproduces R8 bit-for-bit);
statistical-honesty **PASS** (all 8 CIs match the artifact; no overclaim; `C−A` correctly reported spanning 0;
MSE ordering confirmed). The interpretation lens **degenerated to a placeholder** (emitted `lens:test /
findings:[a,b]` after 22 tool calls — a harness artifact, not a finding); its concern (causal overreach) is
covered by the statistician lens and by the self-applied softening of the "mechanism" claim above (marked
inferred). 3 MINOR fixes applied at commit: `_edge_features` docstring clarifies the motion features are
out-of-band/leak-free; `_anchor_with_psucc` comments the col-0-only anchor read; the headline marks
"perfect (oracle) recovery".

## Next
**T2 — activation redesign:** replace the `tanh` saturation on the recurrent→logit path so the recovered
temporal signal reaches the acted logit (the T0 saturation tripwire flips). One variable.
