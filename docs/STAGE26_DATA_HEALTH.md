# Stage 26 Data Health

## Cybernetic Diagnostic

- Controlled object: `data_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `WARN`
- score: `60.0`
- diagnosis labels: `sufficient_for_small_pilot_only, data_too_small, data_too_duplicate`
- blocks scale-up: `True`
- likely contribution: medium-high because the formal pilot reused 10 unique contexts into 24 slots
- confidence: `high`

## Key Metrics

- `num_train_scenarios`: `16`
- `num_eval_scenarios`: `8`
- `num_unique_source_contexts`: `10`
- `feasible_ratio_train`: `0.625`
- `feasible_ratio_eval`: `1.0`
- `near_threshold_ratio`: `0.7`
- `hard_infeasible_ratio`: `0.3`
- `sparse_feasible_count`: `7`
- `duplicate_context_rate`: `0.5833333333333333`
- `actor_signature_contradiction_rate`: `0.015037593984962405`
- `edge_target_high_mid_low_distribution`: `{"high": 12, "low": 107, "mid": 23}`
- `edge_delta_positive_negative_balance`: `{"zero": 71}`
- `edge_delta_action_distribution`: `{"edge_delta": 71}`
- `temporal_sequence_count`: `1`
- `actor_feature_missingness`: `0.0`
- `train_eval_source_overlap_count`: `0`

## Evidence

- Stage 25 train/eval split
- Stage 21/22 actor and critic target views

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
