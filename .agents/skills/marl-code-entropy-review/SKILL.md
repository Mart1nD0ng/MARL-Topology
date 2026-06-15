---
name: marl-code-entropy-review
description: Audit code entropy, phase-script growth, duplicate semantics, and migration risk.
---

# MARL Code Entropy Review

## When To Use

- Before migrating legacy code.
- When scripts grow into long-term functionality.
- When duplicate metric or reward semantics appear.

## When Not To Use

- For small tests or documentation updates.
- To block urgent fixes that already have clear contracts and tests.

## Procedure

1. Inventory modules, scripts, configs, and duplicate names.
2. Classify code as core module, adapter, harness, script, or legacy-only.
3. Identify semantic aliases and uncontrolled coupling.
4. Propose bounded refactor or rejection.
5. Add contract or regression tests before moving behavior into `src/`.

## Output Format

```text
Scope:
Entropy signals:
Duplicate semantics:
Phase-script risks:
Migration candidates:
Recommended actuators:
Required tests:
V5 inheritance check:
Forbidden defaults avoided:
Residual risks:
```

## Quality Gates

- Durable functionality belongs in `src/marl_topology/`.
- Phase scripts stay orchestration-only.
- Migration ledger is updated before code movement.
- Skill recommendations preserve `docs/SKILL_CALIBRATION.md` and Stage 0.2 metric governance.

## V5 Anti-Inheritance Calibration

- Treat v5 phase scripts, reward modules, metric aliases, model classes, and result directories as entropy signals until a clean contract and test justify reuse.
- Flag duplicate names for consensus success, derived reliability, timeout, quorum, latency, energy, reward, and topology diagnostics.
- Reject durable behavior in `scripts/phase*` and any new project script that privately implements a core metric, reward, oracle, or actor rule.
- Do not classify old reward, fixed threshold deployment, full-mask optimality, COMA/Q critics, or global actor tensors as migration candidates.

## Failure Modes

- Mechanical copy from v5.
- Refactor changes behavior without baseline.
- Entropy audit becomes an unbounded rewrite.
