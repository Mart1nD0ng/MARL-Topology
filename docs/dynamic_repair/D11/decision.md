# D11 — decision

**Result: KEEP (chance/CVaR/Pareto wired + verified + active; preference is a verified primitive for D12;
A/B deferred to D13).** The dynamic task now genuinely optimizes RELIABILITY (frequency + tail) and
selects checkpoints on the energy-latency Pareto front from VALIDATION metrics — not just mean feasibility.

## What changed (one variable: episode-level reliability constraints, opt-in)
- New `training/dynamic_reliability.py` (thin EPISODE-level wrappers over the verified static Phase-10
  primitives): `episode_chance_dual_step`, `episode_cvar_shortfall` (+ `cvar_bruteforce` oracle),
  `dynamic_pareto_select`, `preference_weighted_objective`.
- **Chance** (`--chance`): the sign-flexible dual `λ_chance` ascends when the per-frame failure rate
  `Pr(C_t<τ)` exceeds `δ` and FALLS (to ≥0) when met; the reward gains `−λ_chance·1[C_t<τ]`, applied in
  BOTH `episode_rollout` AND `dynamic_eval` (train==eval objective, D2/D3 preserved). The dual is updated
  from the RECORDED feasibility flags — **budget-neutral** (no extra evaluator call).
- **CVaR**: `held_cvar_shortfall` = the tail mean of the per-frame reliability shortfalls (over the
  recorded consensus margins — budget-neutral) is reported as a risk metric.
- **Pareto** (`--pareto-archive`): the val archive is seeded ONLY at VAL checkpoints with
  `{split:val, reliability_violation, energy, latency}`; the final checkpoint is `dynamic_pareto_select`
  (reliability-risk → min violation → energy-latency non-dominated → hypervolume), NEVER raw feasibility.
  The energy/latency come from an EXTRA evaluator call at val/held — **NOT budget-neutral**:
  `pareto_budget_neutral=False`, `pareto_evaluator_calls` reported.
- **Preference**: `preference_weighted_objective(base − ω_E·Ẽ − ω_L·L̃)` is a VERIFIED primitive (a
  preference over the energy-latency trade-off changes the reward-best topology); the preference-CONDITIONED
  policy is D12 (PNA actor), so D11 ships the primitive, not the training integration.
- Default off → λ_chance=0 (no penalty), evaluate=None (no extra eval, keep-best-on-val) → trained-policy
  behavior byte-identical. CVaR + consensus-margin are now always REPORTED as free metrics (report-only,
  no behavior/objective change).

## Tests (failing-first; fail on `bc09019` with ImportError, pass after)
- `test_dynamic_chance_dual_up_down` (sign-flexible: rises violated, falls met, never negative),
  `test_dynamic_cvar_metric_matches_bruteforce` (Rockafellar == tail-mean oracle), `test_dynamic_pareto_archive_uses_val`
  (val-only; min-violation/non-dominated; rejects held/train), `test_dynamic_preference_changes_actions`
  (preference flips the reward-best topology).
- Dynamic-RL + reliability **29/29**; unit suite **692/0**; contract **63/0**; `--dynamic --chance
  --pareto-archive` smoke exit 0 (λ_chance/residual + CVaR + Pareto-selected checkpoint + 6
  pareto_evaluator_calls reported).

## Adversarial verification (focused single agent, 4 claims, PASS, no gaps)
1. **Chance**: sign-flexible (residual = frac_below − δ), budget-neutral (recorded flags), penalty in
   train AND eval (objective-consistent).
2. **CVaR**: Rockafellar minimum, matches the oracle, over the recorded margins (no extra eval).
3. **Pareto**: val-only (raises on held/train), reliability/non-dominated not raw feasibility, extra eval
   reported (`pareto_budget_neutral=False`), loads the Pareto-selected val state.
4. **Byte-identity off**: behavior byte-identical when off; CVaR/margin are report-only free metrics;
   training-only (not in the deployed actor).

## Scope / next
- D11 makes the dynamic task optimize reliability/energy/latency (not just feasibility). It does NOT claim
  these mechanisms improve the headline — the A/B (chance residual / CVaR / Pareto hypervolume, per-seed/CI)
  is part of D13, consistent with the static Phase-10 pattern and the D8 binding-limit finding.
- Next: **D12** (PNA / recurrent PNA dynamic actor + the preference-conditioned policy that consumes the
  D11 preference primitive).
