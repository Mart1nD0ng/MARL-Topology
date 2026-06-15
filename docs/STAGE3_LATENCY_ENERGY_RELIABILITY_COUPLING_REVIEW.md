# Stage 3 Latency / Energy / Reliability Coupling Review

## Controlled Object

The controlled object is the Stage 3 communication stack:

- `src/marl_topology/channel/model.py`
- `src/marl_topology/link/transmission.py`
- `src/marl_topology/network/communication.py`
- the Stage 3 communication contracts and fixtures

## Desired State

Stage 3 must expose a communication model where link reliability, latency, and
energy come from one declared transmission regime. The active regime is:

`urlcc_finite_blocklength_v1`

The channel layer produces geometry-conditioned SINR. The link layer converts
SINR, bandwidth, payload, duration, and deadline into packet reliability,
deadline delivery probability, expected latency, and expected energy.

## Current Coupling After Stage 3.6

### Channel

```text
geometry + path loss + interference -> sinr_db
```

The active channel model is `stage3_channel_v1_fspl_sinr`. It does not compute
packet success.

### Link Reliability

For a finite-blocklength transmission with bandwidth `B`, block duration `T`,
SINR `gamma`, and payload `L`:

```text
n = B * T
C(gamma) = log2(1 + gamma)
V(gamma) = (1 - (1 + gamma)^(-2)) * (log2(e))^2
epsilon ~= Q((n*C - L + 0.5*log2(n)) / sqrt(n*V))
packet_success_probability = 1 - epsilon
```

This makes packet success a dependent quantity of duration, bandwidth, payload,
and SINR.

### Inverse Reliability

When `use_inverse_reliability = true`, the link model solves the minimal
`required_transmission_time_s` needed to reach `target_reliability`. This mode
is deterministic and bounded; it does not use sampling or learned prediction.
Stage 4.8 adds explicit capped-search diagnostics through
`required_reliability_met`, `required_transmission_time_capped`,
`success_probability_at_required_time`, and `required_time_search_max_s`.

### Deadline Retransmission

For deadline `D`, per-attempt duration `t_attempt`, and per-attempt success
`p_attempt`:

```text
K = floor(D / t_attempt)
q_deadline = 1 - (1 - p_attempt)^K
expected_attempts = q_deadline / p_attempt
expected_latency_s = expected_attempts * t_attempt
expected_energy_j = expected_attempts * attempt_energy_j
```

If `K = 0`, `q_deadline = 0`. If `p_attempt = 0`, `expected_attempts = K`.
All probabilities are clipped to `[0, 1]`; NaN and infinity are invalid.

### Network Aggregation

Stage 3.4 now treats per-hop delivery as deadline-conditioned link delivery:

```text
p2p_delivery_probability = deadline_delivery_probability
route delivery = product(per-hop p2p_delivery_probability)
network_scheduled_latency_s = sum route or max broadcast expected latency
network_successful_delivery_latency_s = scheduled latency if delivery probability > 0 else 0
network_energy_j = sum(per-hop expected energy)
```

Disconnected records with no selected route still report zero latency and
energy as a sentinel. Failed scheduled records do not: they preserve positive
scheduled latency and energy so Stage 4 protocol accounting can observe phase
occupancy even when delivery probability is zero.

## Boundary

Stage 3 records are communication records only:

- `packet_success_probability` is an attempt-level link field.
- `deadline_delivery_probability` is a link-layer delivery field.
- `network_delivery_probability` is a communication-layer delivery field.
- None of these is `consensus_success_probability`.
- Communication delivery is not `consensus_success_probability`.
- No reward, PBFT quorum success, training, actor/critic, or v5 migration is
  introduced by this coupling model.

## Required Regression Checks

- Success probability increases with SINR.
- Success probability increases with bandwidth.
- Success probability increases with transmission duration.
- Required transmission time increases with payload.
- Required transmission time decreases with SINR.
- Deadline delivery probability increases with max attempts.
- Expected energy increases with expected attempts.
- Communication delivery fields are not exported as consensus metrics.

## V5 Reference

Read-only v5 evidence remains useful only as a warning that deadline,
reliability, quorum, and reward names must not collapse into ambiguous aliases.
The clean project keeps the new Stage 3 coupling in a named regime and does not
inherit v5 metric names, reward shaping, phase scripts, or training interfaces.

## Acceptance

- The old active SINR-only success surrogate is removed from active source
  paths.
- Stage 3 reliability, latency, and energy are coupled through
  `urlcc_finite_blocklength_v1`.
- Stage 4 consumes communication delivery only through declared PBFT message
  matrices and protocol contracts.
