# Stage 26 Assembler Health

## Cybernetic Diagnostic

- Controlled object: `assembler_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `WARN`
- score: `68.0`
- diagnosis labels: `tx_budget_bottleneck`
- blocks scale-up: `True`
- likely contribution: medium because top proposal rejection worsened slightly and tx budget rejections are present
- confidence: `high`

## Key Metrics

- `top_proposal_rejection_rate`: `{"count": 63, "max": 0.06380208333333337, "mean": 0.04679232804232805, "min": 0.03255208333333332, "std": 0.006369899821493487}`
- `above_threshold_rejection_rate`: `{"count": 63, "max": 0.0, "mean": 0.0, "min": 0.0, "std": 0.0}`
- `rejection_reason_distribution`: `{"tx_budget_exceeded": 2036}`
- `high_score_rejected_count`: `None`
- `accepted_low_score_count`: `None`
- `selected_edge_count`: `{"count": 63, "max": 2.8828125, "mean": 2.6612103174603177, "min": 2.55859375, "std": 0.10462882583510792}`
- `empty_graph_rate`: `{"count": 63, "max": 0.0, "mean": 0.0, "min": 0.0, "std": 0.0}`
- `full_graph_rate`: `{"count": 63, "max": 0.0, "mean": 0.0, "min": 0.0, "std": 0.0}`
- `tx_budget_exceeded_rate`: `0.1407632743362832`
- `rx_capacity_exceeded_rate`: `0.0`
- `conflict_rejection_rate`: `0.0`
- `projection_limit_rate`: `0.1407632743362832`
- `stage25_projection_rejection_delta`: `0.004340277777777783`

## Evidence

- Stage 25 projection diagnostics

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
