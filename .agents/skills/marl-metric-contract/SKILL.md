---
name: marl-metric-contract
description: Define and review MARL-Topology metric governance, registration, and CSV/summary discipline.
---

# MARL Metric Contract

## When To Use

- When adding or changing consensus success, latency, energy, topology diagnostics, or any new metric.
- Before comparing policies or reporting evaluation.

## When Not To Use

- For visual-only changes.
- For reward changes without protocol context.

## Procedure

1. Read `docs/METRIC_CONTRACT.md`.
2. Check whether the needed quantity fits an existing minimal concept.
3. If not, register the metric before use.
4. Declare name, definition, range/unit, level, used_for, formula source, dependencies, and tests.
5. Check CSV and summary fields against the registry.

## Output Format

```text
Metric:
Definition:
Range/unit:
Level:
Used for:
Formula source:
Dependencies:
CSV fields:
Tests:
Legacy mapping:
V5 inheritance check:
```

## Quality Gates

- Every output metric maps to a registered name.
- Reward and metric names are not mixed.
- Diagnostics are not presented as primary objectives.
- Derived reliability names from v5 are rejected until registered with tests.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` and `docs/V5_DO_NOT_LEARN_BLINDLY.md` before accepting any v5-derived metric name.
- Keep the active minimal concepts: `consensus_success`, `consensus_success_probability`, `latency`, `energy`, and `topology_diagnostics`.
- Do not reintroduce `P_eff`, hard/soft/legacy modes, log/geometric/arithmetic summaries, or old CSV fields as defaults.
- If a derived metric becomes necessary, require name, definition, range/unit, level, used_for, formula source, dependencies, tests, and a clear reason the minimal concepts are insufficient.

## Failure Modes

- Same metric with multiple names.
- Same name with multiple meanings.
- Private metric reimplementation inside scripts.
