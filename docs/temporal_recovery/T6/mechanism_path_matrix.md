# T6 — Mechanism-Path Matrix (Contract v4 §2)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | code_path | status @ T6 | evidence |
|---|---|---|---|
| real-env R² decomposition (geo / geo+stale / echo) | `t6_env_temporal_structure_gen._real_part` | ACTIVE_IN_EVAL (diagnostic) | temporal_contribution CI>0; test_fit_r2 |
| synthetic AR-recoverability sweep | `t6_env_temporal_structure_gen._synth_ar` | ACTIVE_IN_EVAL (diagnostic) | r2_pred ∝ ρ; test_synth_ar |
| leak-free feature extraction (stale + current geometry) | `_collect_real` | ACTIVE_IN_EVAL | features = ef col 0 (stale) + col 5/velocity (current geometry); true psucc = label only |
| feasibility non-conversion | (cited) T1 arm C / T3–T5 | ACTIVE (cited, No-Silent-Citation) | C−A spans 0; T3–T5 at floor — not re-run at T6 |
| production channel model | env | NOT_MODIFIED | T6 is a diagnostic; no env change (task-1 modification deferred) |

**Blast radius: zero.** T6 is a new diagnostic generator under `scripts/diagnostics/` + a load-bearing test. No
`src/` change, no production env change. `scripts/diagnostics/` is not a deployed path.

**No-Silent-Citation (Contract v4 §8):** the feasibility non-conversion is cited from T1 (arm C `C−A` spans 0,
commit 3b47eec) and T3–T5 (at the floor, commits 8a147ac / 999922e / b6d4ffd), not re-measured at T6; T6 adds the
R²-level recoverability decomposition + the synthetic autocorrelation sweep.
