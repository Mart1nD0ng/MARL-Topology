# Q0 — Freeze the D13 negative as the POMDP-QP-FAR control baseline

## Hypothesis / purpose
This is not an experiment but a **freeze**: the D0–D14 dynamic campaign's negative result is the
control against which every POMDP-QP-FAR arm (Q5–Q12) is measured. Q0 pins the artifacts and the
scope so no later stage silently overwrites or re-scopes them.

## Single change
None (documentation freeze). No `src`/test changes.

## Controlled variables (the frozen scope — Spec/Workflow Q0 exit conditions)
- Data: `dynamic_urban_4rsu` (real 4-RSU urban grid, `--dyn-data urban`) + `dynamic_random_geometry`
  (single-RSU random, `--dyn-data random`).
- N ∈ {8, 12, 16}; 6 frames; hold_interval = 4; γ = 0.95; 30 updates; 25-episode BCSP warm-start.
- 5 seeds per arm; held per-frame feasibility is the primary metric.

## Artifacts to freeze (must exist + be internally consistent)
- `result_save/dynamic_d13_campaign.json` (30.5 KB) — raw per-seed + CI + grouping.
- `docs/dynamic_repair/D13/decision.md`, `docs/dynamic_repair/D13/experiment_plan.md`.
- The per-arm `mechanism_activation.json` referenced by D13.

## Success criterion (Q0 passes iff)
1. The central conclusion **learned < deployable < central** is reproducible from the JSON.
2. The **PNA seed-0 collapse** is recorded (so PNA's +0.125 is a trend, not a win).
3. The D13 scope is fixed (N≤16, 6 frames, 5 seeds) and carried forward verbatim.

## Failure criterion
Any of: JSON missing/corrupt; grouping overlap (learned/deployable/central mixed); a paired CI in the
report that does not match the JSON; the scope silently widened.
