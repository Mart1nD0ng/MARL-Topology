---
name: cybernetic-harness-design
description: Design or review tests, rubrics, tasks, and evidence loops for MARL-Topology work.
---

# Cybernetic Harness Design

## When To Use

- Before adding harness tasks, rubrics, validation scripts, or acceptance templates.
- When a task lacks measurable evidence.
- When contract tests need to protect future migration.

## When Not To Use

- For ordinary source edits already covered by existing tests.
- For training experiments without reward, metric, and evaluation contracts.

## Procedure

1. Identify the behavior to observe.
2. Choose sensors: unit, contract, regression, metric snapshot, or rubric.
3. Define required artifacts and negative checks.
4. Add validation that fails loudly on missing evidence.
5. Run harness validation.

## Output Format

```text
Harness target:
Observed behavior:
Sensors:
Required artifacts:
Negative checks:
Acceptance threshold:
Validation commands:
Residual risks:
V5 inheritance check:
Required negative test:
```

## Quality Gates

- A task must have required fields, evidence, and negative checks.
- A rubric must avoid scoring unverifiable intent.
- Scripts must be minimal and runnable without training.
- Harness tasks near v5 lessons must include a negative check for blind inheritance.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` before adding or revising task/rubric coverage.
- Add negative checks for unregistered metrics, old reward copying, phase-script promotion, full-mask-as-oracle claims, fixed `0.5` deployment defaults, COMA/Q defaults, and actor global-info leakage when those risks are in scope.
- Do not let keyword scoring stand in for semantic evidence; require file references, tests, or explicit manual sensors.
- Harness validation may prove structure only. Treat semantic claims as requiring an additional contract or review sensor.

## Failure Modes

- Rubric rewards vague prose.
- Task has no regression check.
- Harness claims final truth from keyword scoring alone.
