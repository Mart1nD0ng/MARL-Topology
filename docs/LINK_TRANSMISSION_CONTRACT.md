# Link Transmission Contract

## Responsibility

Compute point-to-point communication reliability, latency, and energy from one
shared finite-blocklength transmission model. Stage 3.6 makes link reliability
a dependent quantity of transmission duration, bandwidth, payload, and SINR;
expected latency and expected energy then depend on deadline-bounded
retransmission attempts.

## Inputs

- channel record with `sinr_db`, `distance_3d_m`, `bandwidth_hz`, and
  `tx_power_dbm`
- `payload_bits`
- `bandwidth_hz`
- fixed transmission duration or inverse target-reliability mode
- optional `target_reliability`
- optional `deadline_s`
- processing delay and explicit queueing placeholder
- tx, rx-circuit, and processing power
- propagation distance in meters

## Outputs

- `propagation_delay_s`
- `transmission_delay_s`
- `processing_delay_s`
- `queueing_delay_s`
- `p2p_latency_s`
- `tx_energy_j`
- `rx_energy_j`
- `processing_energy_j`
- `p2p_energy_j`
- `packet_error_probability`
- `packet_success_probability`
- `target_reliability`
- `required_transmission_time_s`
- `required_reliability_met`
- `required_transmission_time_capped`
- `success_probability_at_required_time`
- `required_time_search_max_s`
- `attempt_duration_s`
- `max_attempts_within_deadline`
- `deadline_delivery_probability`
- `expected_attempts`
- `expected_latency_s`
- `expected_energy_j`
- `finite_blocklength_regime_id`

These outputs are link-layer transmission records. They are not PBFT consensus
success, not application deadline satisfaction, not reward, and not topology
oracle status.

They are not PBFT consensus success.

## Units

- payload: bits
- bandwidth: Hz
- delays and latency: seconds
- power: W
- energy: joules
- distance: meters
- probabilities: `[0, 1]`

## Assumptions

- Channel records provide `sinr_db`; they do not provide packet success.
- `transmission_delay_s` is the finite-blocklength block duration used for
  packet reliability.
- `attempt_duration_s` includes propagation, transmission, processing, and
  queueing components for deadline accounting.
- `p2p_latency_s` and `p2p_energy_j` are expected quantities when deadline
  retransmission is active.
- Queueing starts as an explicit placeholder, not a hidden stochastic model.

## Active Regime

`urlcc_finite_blocklength_v1`

The regime name intentionally preserves the owner-specified identifier. It is
the active Stage 3 link-reliability regime.

## Core Formula

For fixed-duration evaluation with bandwidth `B`, transmission duration `T`,
SINR `gamma`, and payload `L` bits:

```text
n = B * T
C(gamma) = log2(1 + gamma)
V(gamma) = (1 - (1 + gamma)^(-2)) * (log2(e))^2
epsilon ~= Q((n*C - L + 0.5*log2(n)) / sqrt(n*V))
packet_success_probability = 1 - epsilon
```

The implementation also supports a simplified mode without the
`0.5*log2(n)` correction through configuration.

Inverse reliability mode solves the minimal `required_transmission_time_s` that
reaches `target_reliability` under the same formula. Stage 4.8 makes the search
cap observable:

- `required_reliability_met`: true only when the solved duration reaches the
  target reliability.
- `required_transmission_time_capped`: true when the search returns
  `required_time_search_max_s` because the target is unreachable within the
  configured bound.
- `success_probability_at_required_time`: the finite-blocklength success
  probability at the returned duration.
- `required_time_search_max_s`: the configured maximum search time.

An unreachable target must not be silently reported as a successful inverse
solution.

Deadline retransmission uses:

```text
K = floor(deadline_s / attempt_duration_s)
q_deadline = 1 - (1 - p_attempt)^K
expected_attempts = (1 - (1 - p_attempt)^K) / p_attempt
expected_latency_s = expected_attempts * attempt_duration_s
expected_energy_j = expected_attempts * attempt_energy_j
```

If `K = 0`, deadline delivery probability is zero. If `p_attempt = 0`,
`expected_attempts = K`. All probabilities are clipped to `[0, 1]`, and no
NaN or infinity is allowed.

## Stage 3.6 Implementation Status

Implemented interfaces:

- `LinkTransmissionConfig`
- `LinkTransmissionRecord`
- `evaluate_link_transmission`
- `finite_blocklength_packet_error_probability`
- `finite_blocklength_packet_success_probability`
- `required_transmission_time_for_reliability`
- `deadline_retransmission_probability`
- `dbm_to_watt`
- `LinkTransmissionFixture`

Implemented behavior:

- fixed-duration packet reliability from SINR, bandwidth, payload, and
  transmission duration;
- inverse target-reliability transmission-time solving;
- explicit capped inverse-reliability diagnostics;
- deadline-bounded retransmission probability;
- expected attempt count;
- expected latency and expected energy from the same attempt model;
- unselected or inactive transmissions represented with
  `transmission_attempted = false` and zero point-to-point latency and energy.

Implemented fixtures:

- `payload_small`
- `payload_large`
- `bandwidth_narrow`
- `bandwidth_wide`
- `unselected_edge`

## Tests

- Success probability increases with SINR.
- Success probability increases with bandwidth.
- Success probability increases with transmission duration.
- Required transmission time increases with payload.
- Required transmission time decreases with SINR.
- Unreachable inverse-reliability targets expose capped diagnostics.
- Deadline delivery probability increases with max attempts.
- Expected energy increases with expected attempts.
- Latency and energy components are nonnegative.
- Packet success probability remains a link field, not a consensus metric.

## Omitted Components

- MAC backoff;
- HARQ combining;
- adaptive modulation/coding policy;
- sleep/idle energy;
- congestion control;
- application wait time;
- PBFT consensus semantics.

## Failure Modes

- Counting energy for unselected or inactive edges.
- Treating a channel SINR surrogate as packet success.
- Hiding queueing assumptions.
- Mixing per-attempt and expected energy without field names.
- Treating link latency as application deadline satisfaction.
- Reporting packet success as topology or consensus success.

## GOAL_SKELETON Coupling

This contract expands `LinkModel` and `LatencyEnergy` into a shared
communication-layer transmission model. Network communication may aggregate
`deadline_delivery_probability`, `expected_latency_s`, and `expected_energy_j`,
but Stage 4 must still map communication delivery into protocol-level
consensus reliability through a protocol contract.

## Deferred To Stage 4

No PBFT quorum success, consensus reliability metric, consensus reward, or
application-layer deadline objective is defined here.
