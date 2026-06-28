# Q2 — CSI-prediction + Temporal-Value health check under stale CSI

## Hypothesis (the single thing this stage tests)
Stale/partial CSI (Q1) creates a genuine POMDP in which **history and/or velocity reduce the error of
estimating the TRUE current channel** from the lagged/held observation. If neither recurrence nor
velocity beats the naive "use the stale value as-is" (identity) predictor, the stale parameters are too
weak (or the observation still too rich) and the temporal machinery (recurrent actor, Q5+) is not yet
justified. This is a DIAGNOSTIC, not a policy/training change — eval-only, no checkpoint.

## Single change (one variable)
A new diagnostic `scripts/diagnostics/csi_prediction_health.py` (+ importable core). No production-path
change; no mechanism activation in the trunk.

## What is measured
For dynamic scenes built with a Q1 CSI model, extract per (edge, frame):
- `true_psucc(e,t)   = context(t).link_records[e].link_success_probability`  (training-only target)
- `obs_psucc(e,t)    = context(src).link_records[e]...`, `src,age = csi_model.edge_plan(e,t)`  (the actor input)
- `rel_vel(e,t)`, `distance_delta(e,t)` — CURRENT local motion (a node senses its own + neighbour
  velocity each frame; velocity is NOT staleified, which is exactly why it can help extrapolate).

Predictors of `true_psucc(e,t)` (train on train scenes, MSE/Spearman on disjoint HELD scenes):
- **identity** — predict `obs_psucc` (no training): the naive baseline.
- **memoryless** — MLP on `[obs_psucc, age]` (no velocity).
- **memoryless+velocity** — MLP on `[obs_psucc, age, rel_vel, distance_delta]`.
- **recurrent** — GRU over the observed sequence `[obs_psucc, age]_{0..t}` (no velocity).
- **recurrent+velocity** — GRU over `[obs_psucc, age, rel_vel, distance_delta]_{0..t}`.

CSI-mode sweep: `current` (sanity: identity MSE ≈ 0, nothing to learn), `delay-1`, `delay-2`,
`partial ρ=0.5`. Secondary: Temporal-Value Δ_H reported (it is computed on the TRUE channel, so it does
not change with the observation model — reported for context, not as the stale-CSI signal).

## Controlled variables
- Data `--dyn-data random` (cheap); motion_features available for velocity features; fixed seeds;
  train/held disjoint geometry; identical predictor capacity/epochs across arms (fair budget).

## Failing-first tests (fail on HEAD; the module/functions are new)
- `test_extract_pairs_observed_history_with_true_current` — under delay-1, `obs_psucc(e,t)==true(e,t-1)`,
  target `==true(e,t)`.
- `test_identity_mse_zero_under_current` — current mode → identity predictor MSE ≈ 0.
- `test_identity_mse_positive_under_delay` — delay-1 on a moving channel → identity MSE > 0.
- `test_velocity_linear_beats_identity_on_monotone_channel` — on a synthetic monotone-trend series, a
  velocity/slope-aware linear predictor has lower MSE than identity (the mechanism premise).

## Success criterion (Q2 passes iff)
1. Tests pass; the diagnostic runs real-shard and writes a JSON report with per-mode per-arm MSE +
   Spearman + the identity baseline, train/held.
2. The report cleanly shows: current → identity≈0 (POMDP is trivial); delay/partial → identity MSE>0.
3. A clear verdict on the EXIT CONDITION: does recurrent OR velocity beat memoryless/identity at
   predicting the true current CSI? (Either YES → temporal justified; or NO → stale params too weak,
   recorded honestly as a gate result, NOT spun as failure of recurrence in general.)

## Failure criterion
The extraction leaks the true current CSI into the predictor input; identity MSE not ≈0 under current
(extraction bug); arms compared at unequal capacity/budget; conclusions stated beyond the tested modes.

## Out of scope
Policy/feasibility under stale CSI (Q12); top-k critical-link metric (report if cheap, else note);
choosing the recurrent actor for RL (that follows from this gate + Q5+).
