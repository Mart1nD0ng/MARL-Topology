# Channel Model Contract

## Responsibility

Convert geometry and active-transmission context into channel records for packet
communication. Stage 3.6 makes the channel layer a SINR producer only; packet
success is not computed by the channel layer.

## Inputs

- geometry visibility record with `distance_3d_m`, `los_state`, and blockers
- carrier frequency in Hz
- bandwidth in Hz for noise calculation
- tx power in dBm
- tx/rx antenna gains in dB
- noise figure in dB
- active same-resource transmitters for interference
- optional deterministic or seeded shadowing configuration

## Outputs

- `channel_model_id`
- `path_loss_db`
- `los_penalty_db`
- `shadowing_db`
- `tx_power_dbm`
- `rx_power_dbm`
- `noise_power_dbm`
- `interference_power_dbm`
- `sinr_db`
- seed or deterministic-shadowing marker when applicable

These outputs are channel records. They are not packet success, consensus
success, PBFT reliability, reward, or topology oracle status.

## Units

- frequency: Hz
- bandwidth: Hz
- powers: dBm
- gains, losses, and SINR: dB

## Assumptions

- LoS/NLoS state comes from the geometry contract.
- Path loss is monotonic nondecreasing with distance for a fixed regime.
- NLoS penalty makes channel quality no better than matching LoS unless a
  future model explicitly justifies otherwise.
- Shadowing is disabled or seeded for deterministic tests.
- Interference includes active same-resource transmitters only.

## Omitted Components

- fast fading;
- Doppler;
- MIMO and beamforming;
- scheduler adaptation;
- HARQ;
- empirical calibration;
- correlated shadowing unless later declared.

## Tests

- Farther distance increases path loss under fixed LoS state.
- NLoS is worse than LoS for the same distance and power.
- Increasing tx power increases rx power and SINR.
- Increasing same-resource interference lowers SINR.
- Orthogonal resource transmissions do not contribute to interference.
- Higher SINR improves finite-blocklength success probability only after the
  link-transmission layer consumes the channel record.
- Shadowing is deterministic when seeded.

## Stage 3.2 Implementation Status

Stage 3.6 keeps this implementation status as a SINR-only channel layer.

Implemented interfaces:

- `ChannelModelConfig`
- `ActiveTransmission`
- `ChannelRecord`
- `free_space_path_loss_db`
- `noise_power_dbm`
- `evaluate_channel`

Active channel regime:

`stage3_channel_v1_fspl_sinr`

Implemented deterministic behavior:

- free-space path loss using declared distance, carrier frequency, and a
  documented minimum-distance floor;
- LoS/NLoS penalty from geometry `los_state`;
- optional seeded shadowing, with unseeded shadowing rejected;
- tx and rx antenna gains in dB;
- thermal noise from bandwidth and noise figure;
- same-resource active-transmitter interference;
- orthogonal-resource exclusion from interference;
- SINR from signal over noise plus interference.

Packet success is not computed by the channel layer.

packet success is not computed by the channel layer. The previous active
SINR-only success surrogate has been removed from the active code path and is
not a valid Stage 3 regime.

Implemented fixtures:

- `free_space_close`
- `free_space_far`
- `blocked_by_building`
- `two_transmitters_interference`
- `orthogonal_channel_no_interference`

Layer boundary:

`sinr_db` is a channel-record field. It is not packet success, consensus
success, PBFT reliability, reward, topology oracle status, or a training
objective.

It is not consensus success, not PBFT reliability, and not reward.

Deferred from the channel layer:

- finite-blocklength packet reliability;
- deadline retransmission;
- link-layer payload transmission latency;
- tx/rx/processing energy;
- queueing delay;
- network routing or broadcast delivery;
- PBFT/application-layer reliability.

## Failure Modes

- Mixing W, dBm, and dB without explicit conversion.
- Treating channel `sinr_db` as application consensus reliability.
- Reintroducing a private SINR-only packet-success surrogate.
- Unseeded randomness in regression tests.
- Interference set includes inactive or orthogonal transmitters.
- Hidden clipping creates nonmonotonic behavior without documentation.

## GOAL_SKELETON Coupling

This contract feeds link transmission with geometry-conditioned SINR. The
finite-blocklength link model is the next layer that couples reliability,
latency, and energy.

## Deferred To Stage 4

No PBFT quorum success, consensus reliability metric, consensus reward, or
application-layer deadline objective is defined here.
