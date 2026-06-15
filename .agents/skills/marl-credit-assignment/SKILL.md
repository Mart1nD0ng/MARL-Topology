---
name: marl-credit-assignment
description: Review MARL credit-assignment choices such as COMA, difference rewards, direct edge-delta critics, and centralized critics.
---

# MARL Credit Assignment

## When To Use

- When choosing or reviewing COMA, counterfactual baselines, direct edge-delta critics, GNN critics, or auxiliary value heads.
- When global team reward learning is unstable.

## When Not To Use

- To add arbitrary local rewards.
- To bypass Dec-POMDP actor constraints.

## Procedure

1. Define team objective and constraint metrics.
2. Identify which credit path is used.
3. State critic information and actor information separately.
4. Define fidelity checks against oracle or counterfactual evaluation.
5. Add ablation and failure-mode tests.

## Output Format

```text
Team objective:
Credit mechanism:
Actor information:
Critic information:
Counterfactual source:
Fidelity checks:
Ablations:
Failure modes:
Metric registry impact:
V5 inheritance check:
Forbidden defaults avoided:
```

## Quality Gates

- Credit signal aligns with registered evaluation metrics.
- Critic-only information is not deployed in actor.
- Direct edge-delta claims have counterfactual evidence.
- COMA/Q failure or success is not interpreted without calibration and ranking evidence.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` and lesson `L007_credit_calibration_not_impossibility`.
- Do not inherit v5 COMA, Q critic, edge-delta critic, or actor/critic checkpoint loading as default architecture.
- Separate sign accuracy, ranking, calibration, rare-safety recall, oracle edge-hit rate, and magnitude error before making credit claims.
- Do not introduce local rewards that fight the registered global reliability constraint and latency/energy objectives.

## Failure Modes

- Local reward fights global constraint.
- COMA baseline uses invalid action availability.
- Edge-delta critic learns a proxy unrelated to registered consensus, latency, or energy metrics.
