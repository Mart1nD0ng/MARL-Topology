---
name: marl-protocol-reliability
description: Design and review PBFT quorum, timeout, and consensus reliability semantics.
---

# MARL Protocol Reliability

## When To Use

- When defining PBFT success, quorum, timeout, `consensus_success`, `consensus_success_probability`, or a registered derived reliability metric.
- Before implementing protocol or consensus reliability code.

## When Not To Use

- To choose actor architecture.
- To add standalone timeout or quorum rewards.

## Procedure

1. Start from `docs/PROTOCOL_CONTRACT.md`.
2. Declare committee size, fault tolerance, and quorum rule.
3. Define strict success and deadline conditions.
4. Map protocol success to registered metrics.
5. Add boundary tests for quorum and deadline.

## Output Format

```text
Protocol variant:
n:
f:
Quorum:
Deadline:
Success condition:
consensus_success:
consensus_success_probability:
Derived metrics if registered:
Tests:
Metric registry impact:
V5 inheritance check:
Risks:
```

## Quality Gates

- `n >= 3f + 1` assumption is explicit or variant is named.
- Evaluation use is declared in metric registration.
- Any training surrogate is registered and labeled training-only.
- Timeout, quorum, consensus event/probability, and reward remain separate unless a contract explicitly links them.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` and lessons `L001_metric_governance_first` and `L002_effective_success_alias_risk`.
- Do not import v5 `P_succ`/`P_eff` names or hard/soft/legacy mode taxonomy as defaults.
- If a derived reliability metric is needed, register it as a new metric and state why `consensus_success` or `consensus_success_probability` is insufficient.
- Treat deadline and quorum as protocol conditions, not standalone reward terms or hidden metric factors.

## Failure Modes

- Timeout becomes a reward term.
- Quorum count is reported as a consensus metric.
- Protocol code hides physical-link assumptions.
