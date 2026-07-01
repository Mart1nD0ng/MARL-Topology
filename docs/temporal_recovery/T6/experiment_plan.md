# T6 — experiment plan (env temporal-structure diagnostic; task-1 fallback)

## Question
The model-side levers (T2–T5) are exhausted. Task 1's fallback: investigate the ENV's temporal hidden features.
Does the STALE history carry recoverable information about the current channel BEYOND geometry, and how does
recoverability depend on the channel's temporal autocorrelation? (Diagnostic — no production env change.)

## Design
- **Part A (real env, R² decomposition):** fit small MLPs of the true current psucc and report held R²:
  `geo` [current distance, rel_vel, Δdist] / `temporal` [geo + stale psucc] / `echo` [predict stale]. The
  temporal_contribution = R²_temporal − R²_geo measures the recoverable temporal structure beyond geometry.
- **Part B (synthetic AR sweep, method validation):** an AR(1)-shadowing channel `psucc_t = sigmoid(σ·s_t)`,
  `s_t = ρ·s_{t-1} + √(1−ρ²)·ε`; a predictor of current-from-stale reports r2_pred vs ρ ∈ {0,0.3,0.6,0.9}.

## Metrics (5 seeds; 95% CI)
- Part A: R²_geo, R²_temporal, temporal_contribution (CI), R²_echo.
- Part B: r2_pred (CI) per ρ (absolute recoverability), recovery_over_echo.

## Load-bearing tests
`test_temporal_recovery_T6_env_structure.py`: `_fit_r2` recovers a linear signal (R²>0.9) and rejects noise
(R²<0.2); `_synth_ar` recoverability rises with ρ (r2_pred rho=0 <0.15, rho=0.9 ≫).

## Decision rule
- temporal_contribution CI>0 → the env has recoverable temporal structure; interpret against the (cited)
  feasibility non-conversion (T1 arm C / T3–T5) → localize the binding limit.
- Part B monotonic in ρ → the method recovers structure when present (validation).

## Definition of done
Failing-first tests; 5-seed CIs + raw artifact; scope explicit (prediction-level diagnostic; feasibility
non-conversion cited, not re-run; env modification deferred); decision.md; adversarial Workflow.
