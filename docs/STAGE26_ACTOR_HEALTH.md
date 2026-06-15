# Stage 26 Actor Health

## Cybernetic Diagnostic

- Controlled object: `actor_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `WARN`
- score: `70.0`
- diagnosis labels: `actor_assembler_mismatch`
- blocks scale-up: `True`
- likely contribution: medium; actor moved but did not reduce projection friction
- confidence: `medium`

## Key Metrics

- `score_mean_std_min_max`: `{"count": 63, "max": -0.6027878495091099, "mean": -0.6998999270469114, "min": -0.8090321694811186, "std": 0.053311787473204954}`
- `logit_mean_std`: `{"count": 63, "max": 0.3206777138728781, "mean": 0.2205988465564762, "min": 0.09923849542024499, "std": 0.049640923553545775}`
- `accepted_score_mean`: `None`
- `rejected_score_mean`: `None`
- `accepted_rejected_score_gap`: `None`
- `score_target_correlation`: `None`
- `score_teacher_rank_correlation`: `None`
- `actor_parameter_delta`: `{"count": 3, "max": 4.96758042554211, "mean": 2.3874167700078033, "min": 0.009309637813203153, "std": 2.0292416366842656}`
- `edge_score_entropy_proxy`: `0.2205988465564762`
- `selected_edge_count_shift`: `-0.013020833333333037`
- `empty_full_graph_tendency`: `{"empty_graph_rate": 0.0, "full_graph_rate": 0.0}`
- `actor_feature_missingness`: `0.0`
- `target_utility_distribution`: `{"count": 142, "max": 0.93, "mean": 0.3070422535211263, "min": 0.06, "std": 0.24455705498761904}`

## Evidence

- Stage 25 actor score summaries
- Stage 21/22 actor targets

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
