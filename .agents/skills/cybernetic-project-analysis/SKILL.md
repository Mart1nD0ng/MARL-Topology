---
name: cybernetic-project-analysis
description: Analyze unfamiliar or high-impact MARL-Topology work using an engineering cybernetics control model before edits.
---

# Cybernetic Project Analysis

## When To Use

- Before large design, migration, refactor, or architecture decisions.
- When the controlled object or acceptance signal is unclear.
- When v5 legacy semantics may affect the new clean-core design.

## When Not To Use

- For trivial file presence checks.
- For isolated typo fixes with obvious verification.

## Procedure

1. Name controlled object, boundary, and desired state.
2. List state variables, sensors, actuators, disturbances, and coupling.
3. Capture baseline evidence.
4. Propose the smallest safe actuator.
5. Define acceptance and regression checks.

## Output Format

```text
Controlled object:
Desired state:
Baseline:
State variables:
Sensors:
Actuators:
Disturbances:
Coupling map:
Plan:
Acceptance:
Verification:
Residual risks:
V5 inheritance check:
Forbidden defaults avoided:
Lesson status:
```

## Quality Gates

- Includes baseline and post-change evidence plan.
- Separates verified facts from assumptions.
- Names at least one non-target regression risk.
- Checks `docs/SKILL_CALIBRATION.md` when v5 evidence, migration, metrics, reward, actor, oracle, or training is in scope.

## V5 Anti-Inheritance Calibration

- Treat v5 as a read-only experience library, not a design template.
- Map every v5-inspired claim to `docs/V5_FAILURE_LESSONS.md` with `seed`, `confirmed`, `contradicted`, or `unresolved` status.
- Reject old metric names, old reward logic, phase-script structure, full-mask optimality, fixed `0.5` deployment thresholds, and default COMA/Q architecture unless a clean-project contract and test explicitly admit them.
- If a task asks for architecture or training before oracle, metric, reward, and Dec-POMDP gates exist, return a bounded precondition plan instead of architecture advice.

## Failure Modes

- Treating a model or script as the whole controlled object.
- Expanding scope without a sensor.
- Copying v5 semantics without ledger review.
