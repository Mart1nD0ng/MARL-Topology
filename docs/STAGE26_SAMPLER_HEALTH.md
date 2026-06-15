# Stage 26 Sampler Health

## Cybernetic Diagnostic

- Controlled object: `sampler_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `WARN`
- score: `68.0`
- diagnosis labels: `sampler_projection_mismatch`
- blocks scale-up: `True`
- likely contribution: medium-low; sampler remains valid but projected proposals are not fully aligned
- confidence: `medium`

## Key Metrics

- `proposal_count`: `{"count": 63, "max": 2.8828125, "mean": 2.6612103174603177, "min": 2.55859375, "std": 0.10462882583510792}`
- `sampler_logprob`: `{"count": 63, "max": -4.686213720589876, "mean": -4.789416475517172, "min": -4.869017906486988, "std": 0.03971923259007486}`
- `entropy`: `{"count": 63, "max": 4.841787384822965, "mean": 4.789891684220897, "min": 4.684226233512163, "std": 0.035499438639730985}`
- `unique_proposal_rate`: `None`
- `proposal_overlap_rate`: `None`
- `selected_from_proposed_rate`: `0.953207671957672`
- `candidate_mask_violation_count`: `0`
- `top_k_capacity_binding_rate`: `0.0`
- `active_sampler_id`: `physical_plackett_luce_top_k_sampler`

## Evidence

- Stage 25 sampler logprob and entropy metrics

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
