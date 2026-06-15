---
name: marl-physics-design
description: Design 3D V2X physical-layer contracts and sanity checks before simulator implementation.
---

# MARL Physics Design

## When To Use

- When designing building geometry, LoS/NLoS, path loss, SINR, interference, latency, or energy interfaces.
- Before implementing any 3D physics module.

## When Not To Use

- To tune reward weights.
- To run training.
- To copy v5 channel code without tests.

## Procedure

1. Start from `docs/PHYSICS_CONTRACT.md`.
2. Declare units, inputs, outputs, and seed behavior.
3. Define deterministic sanity scenes.
4. Specify contract tests for field names and physical monotonicity.
5. Map outputs to link and protocol metrics.

## Output Format

```text
Physics component:
Inputs:
Outputs:
Units:
Assumptions:
Sanity scenes:
Contract tests:
Couplings:
Metric registry impact:
V5 inheritance check:
Residual risks:
```

## Quality Gates

- 3D distance and LoS/NLoS are testable.
- SINR names signal, interference, and noise terms.
- Latency and energy state omitted components.
- Physics regimes are named before cross-regime policy or reward claims.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` and lesson `L012_physics_can_mask_learning_claims` before using v5 physics evidence.
- Do not copy v5 channel, fading, interference, BLER, or resource-mask code into the clean skeleton.
- Start with deterministic sanity scenes and registered link outputs; add stochastic realism only after range/unit and monotonic checks pass.
- Full-mask performance under one physics regime is not evidence of policy quality or feasibility in another regime.

## Failure Modes

- 2D shortcut hidden in a 3D interface.
- Random shadowing without seed control.
- Link success probability used as reward without metric contract.
