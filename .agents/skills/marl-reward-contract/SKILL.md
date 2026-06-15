---
name: marl-reward-contract
description: Review or design reward contracts where reliability is a constraint and latency/energy are objectives.
---

# MARL Reward Contract

## When To Use

- Before reward implementation, reward ablation, or reward review.
- Before any training task.

## When Not To Use

- To copy old v5 reward logic.
- To change metrics without metric-contract review.

## Procedure

1. Read `docs/REWARD_CONTRACT.md` and `docs/METRIC_CONTRACT.md`.
2. Define reliability threshold `tau`.
3. State the registered consensus metric and any registered training surrogate.
4. Define latency and energy objectives.
5. List excluded proxies and reward-hacking tests.

## Output Format

```text
Task:
Reliability constraint:
Registered training surrogate:
Evaluation metric:
Latency objective:
Energy objective:
Diagnostics:
Excluded reward terms:
Reward-hacking tests:
Acceptance:
Metric registry impact:
V5 inheritance check:
Forbidden defaults avoided:
```

## Quality Gates

- Reliability plateaus after the registered consensus threshold is satisfied unless explicitly justified.
- Timeout and quorum are not standalone reward terms.
- Improvements are reported with registered metrics, not shaped reward alone.
- Old reward formulas or weights are never accepted as a starting point.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md`, `docs/V5_DO_NOT_LEARN_BLINDLY.md`, and lessons `L004_reliability_plateau_before_resource_objective` and `L013`-`L015` before reward advice.
- Do not copy v5 reward terms, weights, sentinel penalties, timeout rewards, quorum rewards, full-mask normalization, or edge-count objectives.
- Debug hypotheses marked `contradicted` must not become reward design rules.
- Reward design must report registered evaluation metrics separately from any training surrogate or shaped reward.

## Failure Modes

- Optimizing edge count as primary objective.
- Reward scale tuned from intuition only.
- Local individual reward competes with team metric.
