# T5 — decision (uncertainty-gated correction; task 4.2/4.3)

**Disposition: KEEP the uncertainty mechanism (opt-in, correct) / HONEST NEGATIVE — confidence-gating does NOT
convert, because the uncertainty is too weakly calibrated to selectively suppress the harmful corrections.**
Predicting the correction's variance and shrinking by confidence (the one angle T1–T4 did not test) does not
make the recovered psucc a safe anchor input: it shrinks corrections roughly *uniformly* (killing the small MSE
benefit of the mean) without *selectively* protecting the anchor. Fourth mechanism confirming the feature limit.

## Result (5 seeds × {random, urban}, delay-1, in-policy heteroscedastic belief; `uncertainty_gated_metrics.json`)

| metric | random | urban | reading |
|---|---|---|---|
| `mean_beats_floor` (full correction) | +2.2e-3 [+6e-4,+3.8e-3] | +2.2e-3 [−4e-4,+4.7e-3] | mean slightly beats floor (applies the correction) |
| `gated_beats_floor` | +3e-5 [−1e-5,+8e-5] | +3e-5 [+1e-5,+5e-5] | gated at the floor (shrinks toward stale) |
| `mean_feas_gain` | −0.231 [−0.529,+0.066] | **−0.194 [−0.345,−0.043]** | mean HURTS the anchor (sig urban) |
| `gated_feas_gain` | −0.200 [−0.440,+0.040] | −0.188 [−0.383,+0.008] | gated spans 0 but **leans negative** — no "no-harm" |
| `gated_minus_mean_feas` | +0.031 [−0.036,+0.098] | +0.006 [−0.108,+0.121] | **spans 0 — gating NOT better than mean** |
| `calibration` (corr logvar↔error) | 0.063 [−0.037,+0.163] | 0.177 [+0.051,+0.303] | **weak** (uninformative on random) |
| `mean_gate` | 0.137 | 0.175 | mostly uncertain → near-uniform shrinkage |

## What T5 establishes (Claim Cards)
1. **Confidence-gating does not achieve "at worst == stale".** `gated_feas_gain` spans 0 with a negative mean
   (`gated_no_harm=False`) — gating does not remove the anchor harm; it still leans negative.
2. **Gating is not better than mean-only** (`gated_minus_mean_feas` spans 0, `gated_beats_mean=False`) — a
   marginal, non-significant +0.03/+0.006, and it *loses* the mean's small MSE benefit (`gated_vs_mean_mse` CI<0
   on random). Net: gating trades an MSE gain for no feasibility gain.
3. **The uncertainty is weakly calibrated** (calib 0.063 spans 0 random / 0.177 small urban) — the log-variance
   barely tracks the actual error, so the gate cannot *selectively* suppress the harmful corrections; it shrinks
   them roughly uniformly (gate ≈ 0.14–0.18). Calibrating the uncertainty requires knowing *where* the magnitude
   is unpredictable, which — like the magnitude itself — is not in the leak-free features.

## Honest scope / caveats (Contract v4 §5/§14)
- NEGATIVE on the T5 conversion: uncertainty gating does not rescue the recovery. The mechanism is correct and
  load-bearing (heteroscedastic NLL, gate = sigmoid(−logvar), test-verified) but the *signal* (calibrated
  uncertainty) is too weak to help.
- `belief_uncertainty=False` (default) is byte-identical (logvar head appended last; full state_dict preserved;
  prior suite green). `models/`/`training/` are gate-exempt (CTDE).
- The gate function (sigmoid(−logvar)) is a reasonable, standard choice; tuning it would not fix a calibration
  of ~0.06–0.18 (a well-calibrated uncertainty would score ≫ 0.5). The limit is the calibration, not the gate.

## Campaign state — the model-side levers tested here are exhausted

> Significance thresholds (as coded in the generator): `gated_no_harm` := `gated_feas_gain.lo ≥ −0.02` (false
> here); `gated_beats_mean` := `gated_minus_mean_feas.lo > 0` (false here) — so the small positive
> `gated_minus_mean` means (+0.03 / +0.006) are classified NON-significant.

Four model mechanisms have been tested against the magnitude-recovery question, all NEGATIVE on conversion:
- **T2** activation (leaky-tanh) — enabling only, no conversion.
- **T3** correction target — direction recoverable, magnitude not (at the floor).
- **T4** edge recurrence — locus is not the limit (≈ node ≈ floor; hurts).
- **T5** uncertainty — weakly calibrated; gating does not convert.

**The binding limit is definitively the FEATURES, not the architecture.** The T1 oracle proved perfect (true-CSI)
recovery converts (the headroom is real), but no realizable predictor — across four mechanisms — recovers the
decision-critical magnitude from the leak-free features (stale channel + current geometry). Direction is
recoverable (geometry); magnitude (fast-fading residual) and even its uncertainty are not.

## Verification (Ultracode adversarial Workflow `wxbgwz70w`)
4-lens + synthesis. **Synthesis: PASS, 0 MAJOR — clear to commit.** back-compat **PASS** (belief_uncertainty=False
byte-identical, full state_dict preserved, logvar head appended last); correctness **PASS** (NLL matches torch
gaussian_nll_loss to 1e-6, genuinely heteroscedastic; mu==belief() so mean-vs-gated is a clean single variable =
the gate; both rules feed the same anchor + true evaluator); leak-free **PASS** (mu/logvar inference reads only
stale/geometry/GRU; true CSI label/metric only); honest-scope **PASS** (numbers match; the small positive
`gated_minus_mean` reported non-significant; feature-limit scoped to THIS signal, env-fallback + T6 open). 3
MINOR applied: `confidence_gate` docstring notes the clamp; the "exhausted" header scoped to "tested here"; the
significance-threshold definitions added. (Note: "suite 841/0" is the **tests/unit** count I track campaign-wide;
the full `tests/` tree is 904/0, also green.)

## Next
Two remaining stages, no more model levers:
- **T6 — integrated A/B + the task-1 env-feature fallback:** (a) consolidate the best temporal module (T3
  correction) vs the stale anchor, 5-seed, to state the deployable result plainly; and (b) execute the task-1
  fallback — add a leak-safe, temporally-*recoverable* channel component (e.g. autocorrelated shadowing) to the
  env and re-probe whether, when the magnitude IS recoverable by construction, the temporal module converts.
  This distinguishes "the method is inadequate" from "the env has no recoverable temporal structure."
- **T7 — docs close-out + 中文 analysis+data report.**
