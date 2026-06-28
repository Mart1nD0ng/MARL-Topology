# Q2 — decision (CSI-prediction + Temporal-Value health check under stale CSI)

**Result: KEEP / GATE PASS.** Stale CSI (Q1) creates a genuinely recoverable POMDP: predicting the
TRUE current channel from the stale observation, **temporal modeling beats the identity baseline AND
plain memoryless on all 3 seeds × all 3 stale modes**. This justifies a recurrent actor for the
residual RL stages (Q5+). Eval-only diagnostic — no production-path change. Adversarially verified
(independent agent, 6 claims, **PASS**, no leak / provably causal GRU / fair budget).

## The diagnostic
New `scripts/diagnostics/csi_prediction_health.py` (+ importable core). For dynamic scenes built with a
Q1 CSI model, extract per (edge, frame) `(obs_psucc, age, rel_vel, distance_delta, true_psucc)` — the
true current psucc is a **training-only target** (deployed actor never sees it; the predictor INPUT is
ONLY the stale obs + age + current velocity, structurally incapable of holding the target). Train tiny
predictors (MLP / causal GRU, hidden 16, matched epochs) on train scenes, MSE/Spearman on disjoint held.

## Result (3 seeds, train 8 / held 6, 5 frames, random data; held MSE of the true current psucc)
| CSI mode | identity (naive) | memoryless | best temporal | beats baseline (3 seeds) |
|---|---|---|---|---|
| current | **0.0000** | 0.0002 | 0.0002 | [False, False, False] |
| delay-1 | 0.1040 | 0.0914 | **0.0802** | [True, True, True] |
| delay-2 | 0.1054 | 0.0930 | **0.0868** | [True, True, True] |
| partial ρ=0.5 | 0.0524 | 0.0470 | **0.0456** | [True, True, True] |

- **current → identity MSE exactly 0** (obs == true): the POMDP is degenerate, nothing to recover, and
  `temporal_beats_baseline=False` correctly cannot be spun as a win. This is the sanity anchor.
- **delay/partial → identity MSE jumps to 0.05–0.10**: stale CSI made the current channel HIDDEN.
- **Temporal beats baseline on every seed and every stale mode**: even `memoryless+age` reduces the
  error (0.104→0.091 on delay-1), and adding **history/recurrence** reduces it further (→0.080, ~23%).
  Recurrence is the consistent lever; velocity-alone is inconsistent at this scale (helped delay-2/
  partial marginally, hurt in one rec+vel partial run) — recorded honestly, not over-claimed.

## Exit condition (Spec Q2) — MET
"recurrent OR velocity beats memoryless/identity at predicting the current CSI" → YES, sign-stable
across 3 seeds in all stale modes. The stale parameters are strong enough; the temporal machinery
(recurrent actor) is justified for Q5+.

## Adversarial verification (independent agent, 6 claims) — PASS, no issues
1. NO TARGET LEAK ✓ — input tensor has exactly 4 cols {obs,age,rel_vel,dist_delta}; target (true_psucc,
   index 4) cannot enter `x`; obs read from `context(src)` (src = edge_plan source), not `context(t)`.
2. NO FUTURE LEAK ✓ — `nn.GRU(batch_first=True)`, not bidirectional; perturbing the last frame leaves
   hidden states 0..t−1 exactly unchanged (causal).
3. FAIR BUDGET ✓ — all arms share train/held data, epochs (300), hidden (16), lr; only arch/feature-count
   differ (the intended variables).
4. HONEST VERDICT ✓ — `best_temporal < min(identity, memoryless) − 1e-6` on HELD MSE; train MSE unused;
   disjoint train/held seeds (×1000+1 vs ×1000+777).
5. EVAL-ONLY ✓ — no checkpoint, no reward feed, no production-path mutation; nothing in src/ imports it.
6. IDENTITY==0 NON-VACUOUS ✓ — under current, `edge_plan→(t,0,True)` so obs==true exactly.

## Honest scope (NOT a headline)
- This is the **prediction task in isolation** (a clean supervised problem with the true target). It
  proves the hidden current CSI is **recoverable from history** — the PRECONDITION for a recurrent
  actor — NOT that recurrence improves the POLICY/feasibility under RL (the actor has no true target,
  only the reward). That is Q12's question.
- Pilot scale (train 8, held 6, 5 frames, 3 seeds, random single-RSU). Sign-stable but modest magnitude
  (~13–23% MSE reduction); not a performance claim.

## Acceptance table (Contract v3 §15)
- Phase: **Q2 — CSI-prediction / Temporal-Value health check**
- Status: **VALIDATED_POSITIVE (gate met)** — diagnostic, not a policy result.
- Implemented ✓ / Wired (diagnostic script) ✓ / Active by default n/a / Active in this run ✓ (4 modes × 3 seeds)
- Test scale: train 8 / held 6 / 5 frames / 3 seeds; 5 new unit tests; full suite 720/0
- Dataset: `--dyn-data random`; Target: true current psucc (training-only); Predictors: identity / MLP /
  causal GRU (matched budget); no evaluator calls (prediction, not feasibility)
- Mechanisms active: stale-CSI recoverability check. Not tested: policy/feasibility under stale CSI (Q12).
- Positive findings: stale CSI is recoverable; recurrence reduces current-CSI error (3/3 seeds, all stale modes).
- Negative findings: velocity-alone inconsistent at pilot scale; under current CSI nothing to recover.
- Conclusion scope: the hidden current CSI is recoverable from history under stale CSI; recurrent actor
  justified for Q5+. NO claim about RL policy gains.
- Next action: **Q3 — quorum shortfall / D_quorum diagnostics** (per-phase/per-receiver expected
  shortfall, mean/max/CVaR), NOT yet wired into reward (gated on the Q4 alignment test).
