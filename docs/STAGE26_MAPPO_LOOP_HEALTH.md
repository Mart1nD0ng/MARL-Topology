# Stage 26 Mappo Loop Health

## Cybernetic Diagnostic

- Controlled object: `mappo_loop_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `WARN`
- score: `62.0`
- diagnosis labels: `unstable_seed, critic_not_helpful`
- blocks scale-up: `True`
- likely contribution: medium-high because one seed stopped and critic signal was weak despite safe KL/entropy
- confidence: `high`

## Key Metrics

- `approx_kl`: `{"count": 50, "max": 0.004846026189625263, "mean": 0.001094675436615944, "min": 8.65226611495018e-05, "std": 0.0009545715725640384}`
- `clip_fraction`: `{"count": 50, "max": 0.013671875, "mean": 0.0014453125, "min": 0.0, "std": 0.0028386909842225256}`
- `entropy_trend`: `{"delta": 0.02643933892250061, "first": 4.758865863084793, "last": 4.785305202007294}`
- `policy_loss`: `{"count": 50, "max": 6.614008452743292e-05, "mean": -0.0016843427758431063, "min": -0.0071448159869760275, "std": 0.001453063531668014}`
- `value_loss`: `{"count": 50, "max": 240147.59375, "mean": 218482.9696875, "min": 185676.828125, "std": 11462.853385459792}`
- `actor_grad_norm`: `None`
- `critic_grad_norm`: `None`
- `combined_grad_norm`: `{"count": 50, "max": 6763.841176986694, "mean": 4452.037420751452, "min": 2483.344937801361, "std": 1183.684082196155}`
- `ratio_mean`: `None`
- `ratio_max`: `None`
- `train_eval_gap`: `-0.3671875`
- `seed_success_count`: `2`
- `seed_stop_reason_count`: `{"tau_feasible_rate_degraded": 1}`
- `update_effect_size`: `{"energy_delta": -0.00019750000000000236, "latency_delta": -8.90672257260498e-05, "tau_feasible_rate_delta": -0.03125}`

## Evidence

- Stage 25 update and eval metrics

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
