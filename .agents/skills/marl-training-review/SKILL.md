---
name: marl-training-review
description: Review training plans, diagnostics, and evidence before running MARL-Topology experiments.
---

# MARL Training Review

## When To Use

- Before training runs, hyperparameter sweeps, or evaluation claims.
- When training instability, reward hacking, or seed variance is suspected.

## When Not To Use

- During this scaffold-only phase unless planning future work.
- Without metric, reward, and evaluation contracts.

## Procedure

1. Confirm reward, metric, protocol, and Dec-POMDP contracts exist.
2. Define seeds, scenarios, baselines, and hard evaluation.
3. List diagnostics: losses, entropy, constraint violations, consensus metrics, latency, and energy.
4. Define stop conditions and artifact locations.
5. Plan regression against oracle or deterministic baselines.

## Output Format

```text
Training purpose:
Contracts checked:
Baselines:
Seeds:
Diagnostics:
Registered evaluation:
Stop conditions:
Artifacts:
V5 inheritance check:
Forbidden defaults avoided:
Residual risks:
Verdict:
```

## Quality Gates

- No training without registered evaluation path.
- Reports include variance or repeated-seed plan.
- Shaped reward is not the primary success evidence.
- Training proposals are blocked until oracle, baseline, metric, reward, and Dec-POMDP gates exist.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` and lessons `L006_fixed_threshold_deployment_risk`, `L010_training_after_oracle_baseline_contracts`, and `L011_debug_hypotheses_need_status`.
- Do not start training to resolve missing contracts, missing oracle evidence, reward ambiguity, metric ambiguity, or actor leakage questions.
- Fixed `0.5` threshold deployment, full-mask reproduction, single-seed success, and shaped-reward improvement are diagnostics, not final evidence.
- Debug claims must name data window, seed coverage, and lesson status before becoming training decisions.

## Failure Modes

- Training starts before contracts.
- One seed is treated as final evidence.
- Training surrogate improvement hides registered evaluation failure.
