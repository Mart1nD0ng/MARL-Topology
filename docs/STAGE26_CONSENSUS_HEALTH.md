# Stage 26 Consensus Health

## Cybernetic Diagnostic

- Controlled object: `consensus_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `PASS`
- score: `85.0`
- diagnosis labels: `consensus_signal_healthy`
- blocks scale-up: `False`
- likely contribution: medium when expected-initiator averaging exposes weak primary spread
- confidence: `medium`

## Key Metrics

- `consensus_success_probability`: `{"count": 60, "max": 1.0, "mean": 0.38000002304257924, "min": 0.0, "std": 0.4826316564638133}`
- `tau_feasible_rate`: `0.36666666666666664`
- `violation_rate`: `0.6333333333333333`
- `per_primary_reliability_variance`: `0.23437498036152785`
- `min_primary_reliability`: `0.0`
- `max_primary_reliability`: `1.0`
- `primary_spread`: `{"count": 60, "max": 1.0, "mean": 0.01666672427307332, "min": 0.0, "std": 0.12801908829885933}`
- `phase_pre_prepare_delivery`: `{"count": 60, "max": 1.0, "mean": 0.48277777777777786, "min": 0.0, "std": 0.42077933280233754}`
- `phase_prepare_delivery`: `{"count": 60, "max": 1.0, "mean": 0.48277777777777786, "min": 0.0, "std": 0.42077933280233754}`
- `phase_commit_delivery`: `{"count": 60, "max": 1.0, "mean": 0.48277777777777786, "min": 0.0, "std": 0.42077933280233754}`
- `quorum_tail_sensitivity`: `0.01666672427307332`
- `fault_filter_effect`: `remove_largest_fault_filter_active_in_stage21_stack`

## Evidence

- Stage 4 expected-initiator PBFT evaluations

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
