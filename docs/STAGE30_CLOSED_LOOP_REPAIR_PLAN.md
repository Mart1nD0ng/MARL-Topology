# Stage 30 Closed-Loop Repair Plan

Stage 29 blocked scale-up because critic repair worked, but reward/objective alignment, projection friction, reliability margin, and data scale remained unresolved.

## Control Model

- Controlled object: pre-scale repair loop for the active policy-gradient stack.
- Desired state: either large-scale readiness certificate or a blocker review with owner decision required.
- State variables: component health, reliability margin, reward/objective rank alignment, projection rejection, data scale, critic health, and loop iteration count.
- Sensors: Stage 26 diagnostics, Stage 27 critic report, Stage 28 repaired-critic pilot, Stage 29 decision packet, Stage 30 tests, and harness validation.
- Actuators: diagnostic repair code, report generation, readiness gate, root-cause matrix, and owner-gated next-stage recommendation.
- Disturbances: small duplicated data, stochastic policy-gradient outcomes, reward/projection coupling, and pressure to promote small-pilot evidence into scale-up.

## Iteration Protocol

Each iteration diagnoses, chooses the dominant blocker by priority, applies the smallest executable repair or repair candidate, validates with report evidence, updates the root-cause matrix, then either continues or stops.

## Entropy Control

Stage 30 uses internal iteration ids instead of new stage numbers. Durable behavior lives in `src/marl_topology/evaluation/stage30_repair_diagnostics.py`; generated docs are the human review surface.

## Boundary

- large-scale training remains blocked
- final tau selection remains blocked
- reward-weight sweeps remain blocked
- sampler switching remains blocked
- recurrent policy work remains blocked
- owner decision is required for Stage 31
