# T6 — decision (env temporal-structure diagnostic; task-1 fallback)

**Disposition: KEEP (diagnostic) — REFINES the campaign's binding limit.** T6 executes the task-1 env-feature
fallback as a *diagnostic* (no production env change). It overturns a simple "the env has no recoverable temporal
structure" story: the stale history DOES carry recoverable information beyond geometry, a dedicated deployable
predictor beats the echo on R², **yet none of it converts to an anchor-feasibility gain** (T1 arm C / T3–T5). So
the binding limit is the **deployable PRECISION at the anchor's decision boundary**, not the absence of
recoverable information — the same "deployable precision at N≤16" wall the project's prior four campaigns reached.

## Result (5 seeds; `env_temporal_structure_metrics.json`)

**Part A — real env R² decomposition** (held R² of the true current psucc):

| predictor | random | urban | meaning |
|---|---|---|---|
| geometry only (current dist, vel, Δdist) | 0.447 | 0.190 | the leak-free geometric part |
| echo (predict stale psucc) | 0.472 | 0.556 | the trivial temporal baseline |
| **geometry + stale** | **0.596** | **0.640** | the deployable predictor |
| **temporal_contribution** = geo+stale − geo | **+0.149 [0.127,0.172]** | **+0.450 [0.410,0.490]** | stale adds beyond geometry (CI>0) |
| geo+stale − echo | +0.124 | +0.084 | the deployable predictor beats echo on R² |

**Part B — synthetic AR sweep** (recoverability r2_pred of current psucc from stale, vs temporal autocorrelation ρ):

| ρ | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| r2_pred | −0.003 [−0.007,+0.001] | 0.086 [0.068,0.105] | 0.356 [0.328,0.385] | 0.806 [0.797,0.815] |

Recoverability rises monotonically with ρ: at ρ=0 (IID) nothing is recoverable; at ρ=0.9 most of the variance is.
The real env's R²~0.6 corresponds to an effective ρ≈0.75 (moderate-high).

## What T6 establishes (Claim Cards)
1. **The env is NOT structureless.** The stale history adds significant recoverable R² beyond geometry
   (temporal_contribution CI>0 both regimes), and the deployable geometry+stale predictor beats the echo on R²
   (by ~0.08–0.13). The current psucc is partially recoverable from the leak-free features.
2. **The recoverable R² does NOT convert to feasibility.** T1 arm C (same geometry+stale predictor) had its MSE
   beat echo but its anchor feasibility `C−A` span 0; T3–T5 land at the floor on the decision. So an ~0.1 R²
   advantage over echo is insufficient to flip the right edges at the anchor's keep/add thresholds → the binding
   limit is **decision-boundary precision**, not recoverability.
3. **Recoverability scales with temporal autocorrelation (method validation).** The synthetic sweep shows a
   predictor recovers a fraction of the variance ∝ ρ (0→0.81 as ρ 0→0.9), so the method is sound — it recovers
   temporally-autocorrelated structure when present. The true-CSI oracle (ρ=1, perfect precision) converts
   (+0.165, T1); realizable predictors (R² gain, imperfect precision) do not.

## Refined campaign diagnosis (supersedes the "magnitude not in the features" wording of T3–T5)
The earlier stages' "direction learnable, magnitude not recoverable" was slightly overstated. T6 shows the
magnitude is **partially recoverable** by a dedicated predictor (R² gain over echo), but neither the in-policy
belief (T3–T5, which under-extract it) nor a dedicated predictor (T1 arm C / T6) **converts** it to a deployed
feasibility gain. **The binding limit is the deployable PRECISION of the recovered channel at the anchor's
decision boundary** — realizable predictions are better than echo on average but not precise enough at the
decision-critical edges. This is the project's recurring "deployable precision at N≤16" wall, now localized to the
temporal/CSI-recovery axis.

## Honest scope / caveats (Contract v4 §5/§14)
- Part A R² is measured with a small MLP on pooled held edges; it is a *prediction* diagnostic, not a feasibility
  measurement — the feasibility non-conversion is established by T1 arm C + T3–T5 (cited), not re-run here.
- Part B is a *synthetic* AR channel (method validation), not the real env; it shows recoverability ∝ ρ but does
  not itself measure feasibility conversion (the precision bar for conversion is not measured synthetically).
- The task-1 env MODIFICATION (adding a leak-safe temporally-recoverable component to the production channel and
  re-running the oracle for feasibility) is NOT done here — it is a scoped follow-up (owner decision), now
  motivated by the diagnostic: raising the channel's decision-critical autocorrelation would raise R², but must
  clear the decision-boundary precision bar to convert.

## Verification (Ultracode adversarial Workflow `wtwf15cxb`)
4-lens + synthesis. **Synthesis: PASS, 0 MAJOR — safe to commit.** r2-correctness **PASS** (genuine train/held
split, train-only normalization; AR(1) independently verified stationary with lag-1 autocorr==ρ; R²
perfect=1/mean=0/wrong<0); leak-free **PASS** (features S/GEO carry no true-current quantity — col 5 distance is
not in csi_columns; true psucc is the target only); No-Silent-Citation **PASS** (feasibility non-conversion cited
from T1/T3–T5 with verified commit hashes, not re-presented); interpretation **PASS/MINOR** (all numbers match;
**honest in both directions** — claims neither that the method converts nor that the env is structureless; the
precision conclusion is supported by combining measured R² with cited non-conversion). MINOR applied: the
echo-beat bracket widened to ~0.08–0.13 (covers the random 0.124), a leak-free comment added to `_collect_real`.
(A noted non-blocking nit: `_fit_r2` uses the unbiased sample variance in the R² denominator — a standard
estimator that cancels in the reported differences; left as-is to avoid re-running.)

## Next
**T7 — docs close-out + 中文 analysis+data report:** consolidate T0–T6, state the plain deployable result (the
best temporal module ≈ the stale anchor, no gain), the refined binding-limit diagnosis, and the env-feature
follow-up; update the repo authority docs; deliver the Chinese summary; ask the owner about pushing T0–T7.
