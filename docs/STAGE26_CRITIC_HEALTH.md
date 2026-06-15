# Stage 26 Critic Health

## Cybernetic Diagnostic

- Controlled object: `critic_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `FAIL`
- score: `35.0`
- diagnosis labels: `critic_unused_or_weak, critic_high_bias, critic_value_scale_mismatch, critic_underfit`
- blocks scale-up: `True`
- likely contribution: high because explained variance is near zero and value scale mismatch is large
- confidence: `high`

## Key Metrics

- `value_loss`: `{"count": 50, "max": 240147.59375, "mean": 218482.9696875, "min": 185676.828125, "std": 11462.853385459792}`
- `explained_variance`: `{"count": 50, "max": 0.0009551234543323517, "mean": 0.00012423977255821228, "min": -0.00141964852809906, "std": 0.00041022137818060093}`
- `value_prediction`: `{"count": 50, "max": -0.1534189977683127, "mean": -2.8842623532284053, "min": -6.755303502082825, "std": 1.7577996264478903}`
- `return`: `{"count": 50, "max": -318.93017578125, "mean": -351.80689514160156, "min": -372.0033874511719, "std": 13.176843155006035}`
- `value_return_correlation`: `-0.09110991184763563`
- `value_bias`: `342.39235267917314`
- `advantage_std`: `{"count": 50, "max": 1.0, "mean": 0.9999999964237213, "min": 0.9999999403953552, "std": 1.4155318840787316e-08}`
- `advantage_std_with_critic`: `{"count": 50, "max": 1.0, "mean": 0.9999999964237213, "min": 0.9999999403953552, "std": 1.4155318840787316e-08}`
- `advantage_std_batch_mean_baseline`: `None`
- `critic_grad_norm`: `None`
- `combined_grad_norm`: `{"count": 50, "max": 6763.841176986694, "mean": 4452.037420751452, "min": 2483.344937801361, "std": 1183.684082196155}`
- `critic_parameter_delta`: `{"count": 3, "max": 35.02615589182824, "mean": 25.048465790384096, "min": 12.831761317886048, "std": 9.198104030769894}`
- `edge_delta_rank_metrics`: `None`
- `feasibility_accuracy`: `None`

## Evidence

- Stage 25 value-loss/explained-variance metrics
- critic prediction vs return pairs

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
