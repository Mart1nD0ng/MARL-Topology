# Stage 26 Communication Health

## Cybernetic Diagnostic

- Controlled object: `communication_health` in the active Stage 25 stack.
- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.
- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.
- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.
- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.

## Status

- status: `PASS`
- score: `85.0`
- diagnosis labels: `communication_signal_usable`
- blocks scale-up: `False`
- likely contribution: medium if link probabilities are saturated or flat
- confidence: `medium`

## Key Metrics

- `link_success_probability`: `{"count": 71, "max": 1.0, "mean": 0.9303736177813934, "min": 0.0, "std": 0.2530363613313749}`
- `sinr_db`: `{"count": 71, "max": 52.14325972379691, "mean": 36.38032861660262, "min": -12.574496486411135, "std": 17.50873785355118}`
- `deadline_delivery_probability`: `{"count": 71, "max": 1.0, "mean": 0.9325020845208734, "min": 0.0, "std": 0.24622067052538488}`
- `required_reliability_met_rate`: `1.0`
- `required_transmission_time_capped_rate`: `0.0`
- `expected_attempts`: `{"count": 71, "max": 4.0, "mean": 1.2066681598640934, "min": 1.0, "std": 0.7516624321995388}`
- `scheduled_latency`: `{"count": 912, "max": 0.007800354966071213, "mean": 0.0006654409690580185, "min": 0.0, "std": 0.0010135490846951078}`
- `successful_latency`: `{"count": 912, "max": 0.00240018450500151, "mean": 0.0003890662887631674, "min": 0.0, "std": 0.00048325527715468314}`
- `energy`: `{"count": 983, "max": 0.0014320554804791094, "mean": 0.0001510326153861173, "min": 0.0, "std": 0.00019183059480130902}`
- `p2p_latency_energy_correlation`: `0.9637252396710405`
- `reliability_latency_energy_coupling_sanity`: `{"latency_energy_correlation": 0.9637252396710405, "probability_energy_correlation": -0.9630728777983698, "probability_latency_correlation": -0.9990610254913702}`

## Evidence

- Stage 3 finite-blocklength link recomputation
- Stage 21 network records

## Residual Risk

This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.
