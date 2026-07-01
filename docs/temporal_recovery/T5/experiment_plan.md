# T5 — experiment plan (uncertainty-gated correction; task 4.2/4.3)

## Hypothesis (one variable: the recovery rule — mean vs confidence-gated)
T3/T4 showed the correction captures direction but not magnitude, and applying the noisy magnitude everywhere
HURTS the anchor. T5 trains a HETEROSCEDASTIC belief (Gaussian NLL): the head predicts a correction mean `mu`
AND a per-edge log-variance `logvar`. At inference, apply the correction ONLY where confident:
`recovered = sigmoid(stale_logit + confidence_gate(logvar)·mu)`, `confidence_gate = sigmoid(−logvar)`. The stale
observation is the physical baseline (task 4.2); the correction is a confidence-weighted residual (task 4.3).
Hypothesis: gating removes the magnitude-noise harm → the gated recovered psucc is at worst == stale (feas_gain
≈ 0), unlike the mean rule.

## Controlled variable
The recovery RULE on the SAME NLL-trained model: mean (`stale_logit + mu`) vs gated (`stale_logit +
gate(logvar)·mu`). Everything else identical (same model, features, residual_leak=0.1). `belief_uncertainty` is
opt-in (default off, byte-identical).

## Metrics (5 seeds × {random,urban}, delay-1, held; 95% CI)
- `{mean,gated}_beats_floor` = stale_echo_MSE − belief_MSE.
- `{mean,gated}_feas_gain` = feas(recovered → anchor) − feas(stale → anchor).
- `gated_minus_mean_feas` (does gating help vs mean?).
- `calibration` = corr(logvar, squared error of mu) (is the uncertainty meaningful?).
- `mean_gate` = mean confidence gate.

## Load-bearing tests
`test_temporal_recovery_T5_uncertainty.py`: uncertainty head added / mu == belief() / gate shrinks uncertain
corrections / NLL heteroscedastic (high logvar absorbs error) / calibration positive when logvar tracks error /
belief_uncertainty=False byte-identical (full state_dict).

## Decision rule
- gated achieves no-harm (feas_gain ≥ ~0) and/or beats mean (CI>0) → uncertainty gating converts → KEEP+adopt.
- gated spans/below 0 and ≈ mean → NEGATIVE (uncertainty too weak); model levers exhausted → T6 (integrated A/B
  + env-feature fallback).

## Definition of done
Failing-first tests; 5-seed CIs + raw artifact; scope explicit; decision.md; adversarial Workflow;
belief_uncertainty=False byte-identical (full prior suite green).
