# Stage 26 Reward Objective Health

## Cybernetic Diagnostic

- Controlled object: `reward_objective_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `WARN`
- score: `65.0`
- diagnosis labels: `reward_objective_mismatch`
- blocks scale-up: `True`
- likely contribution: medium because reward worsened while latency and energy improved
- confidence: `high`

## Key Metrics

- `reward`: `{"count": 912, "max": -3.443459986970389, "mean": -35.93143478995562, "min": -83.51483843137554, "std": 38.74091833821607}`
- `reward_component_means_std`: `{"energy_component": {"count": 912, "max": 2.0, "mean": 1.484283182830982, "min": 0.0, "std": 0.6264860919516259}, "latency_component": {"count": 912, "max": 2.0, "mean": 1.5853095018614554, "min": 0.0, "std": 0.5345070325955771}, "reliability_violation_component": {"count": 912, "max": 0.81, "mean": 0.3286184210526324, "min": 0.0, "std": 0.39773213900501647}}`
- `reward_variance`: `1500.858753688326`
- `reward_feasible_vs_infeasible_gap`: `78.89214706716577`
- `reward_latency_correlation`: `0.8368566503071432`
- `reward_energy_correlation`: `0.969586449615794`
- `reward_consensus_correlation`: `0.9999298457571467`
- `objective_rank_reward_rank_spearman`: `-0.8591934743000385`
- `dominated_topology_reward_error_count`: `0`
- `empty_graph_reward_rank`: `543`
- `full_graph_reward_rank`: `19`
- `sparse_feasible_reward_rank`: `792`
- `stage25_reward_delta`: `-2.466148376475111`

## Evidence

- Stage 25 reward surface analysis
- Stage 25 eval aggregate deltas

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
