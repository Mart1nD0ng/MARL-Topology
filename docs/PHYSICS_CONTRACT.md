# Physics Contract

This document defines the future 3D physical-layer interface. It is a contract only; the scaffold does not implement the full simulator.

## Required Inputs

Building geometry:

- 3D axis-aligned or mesh-backed building volumes.
- Building height, footprint, material class, and scenario id.

Road and node geometry:

- Vehicle positions in 3D.
- RSU and base-station positions in 3D.
- Road lanes and allowed mobility surfaces.

## Required Computations

3D distance:

- Euclidean distance between transmitter and receiver positions.
- Must preserve units in meters.

Ray-box LoS/NLoS:

- Ray from transmitter to receiver is tested against building volumes.
- Result must distinguish clear LoS, blocked NLoS, and optional partial obstruction classes.

Path loss:

- Uses declared frequency, distance, LoS/NLoS state, and model version.
- Must expose units and clipping rules.

Shadowing:

- Random or deterministic shadowing component with seed control.
- Must declare distribution and correlation assumptions.

SINR:

- Signal power divided by interference plus noise.
- Must declare transmit power, antenna gains, bandwidth, noise figure, and interference set.

Interference:

- Must define which concurrent transmitters contribute.
- Must expose whether scheduling, channel reuse, or PBFT traffic phase controls interference.

Link success probability:

- Derived from SINR, modulation/coding or abstract link model, and packet size.
- Must declare whether it is empirical, analytic, or surrogate.

Latency:

- Includes propagation, transmission, queueing, processing, and protocol wait components when available.
- Any omitted component must be logged as omitted.

Energy:

- Includes transmit energy and optional receive, idle, and processing energy.
- Must declare device class and power model.

## Interface Expectations

Future physics modules must return structured records with:

- `tx_id`
- `rx_id`
- `scenario_id`
- `distance_3d_m`
- `los_state`
- `path_loss_db`
- `shadowing_db`
- `sinr_db`
- `link_success_probability`
- `latency_s`
- `energy_j`

## Acceptance

- Unit tests cover distance, LoS/NLoS, and deterministic seed behavior.
- Contract tests verify units and field names.
- Regression tests include simple city scenes with known blocked and unblocked links.

## Stage 2.5 Fixture Boundary

Stage 2.5 scenario fixtures deliberately use only the named regime `stage2_deterministic_distance`.

Allowed at Stage 2.5:

- 3D node coordinates in meters;
- deterministic candidate-distance thresholds;
- deterministic distance-based link success, latency, and energy;
- explicit fixture expectations for oracle status and full-graph baseline success.

Deferred:

- ray-box LoS/NLoS;
- path loss;
- shadowing;
- SINR;
- interference;
- mobility;
- stochastic channel sampling.

Cross-fixture baseline comparisons must report the physics regime and must not claim policy quality across future physics regimes.

## Stage 2.7 Link Regime Review

The active link regime is `stage2_deterministic_distance`, reviewed in `docs/LINK_MODEL_REGIME_REVIEW.md` and represented in `src/marl_topology/link/regime.py`.

Current formulas:

- `link_success_probability = exp(-distance_3d_m / reference_distance_m)`
- `latency_s = base_latency_s + distance_3d_m / speed_of_light_mps`
- `energy_j = fixed_tx_energy_j + energy_per_m_j * distance_3d_m`

Current outputs:

- `distance_3d_m`
- `link_success_probability`
- `latency_s`
- `energy_j`
- `physics_regime`

The regime deliberately omits LoS/NLoS, path loss, shadowing, SINR, interference, packet-size effects, queueing, processing delay, receive energy, and mobility.

Any future richer physical layer must enter as a new named regime or a reviewed revision of this regime, with field, unit, monotonicity, deterministic seed, and regression tests before it is used for reward or training claims.

## Stage 3.2 / Stage 3.6 Channel Regime

The active channel regime is `stage3_channel_v1_fspl_sinr`, represented in
`src/marl_topology/channel/model.py`.

Current channel outputs:

- `distance_3d_m`
- `los_state`
- `blocker_ids`
- `base_path_loss_db`
- `los_penalty_db`
- `shadowing_db`
- `path_loss_db`
- `tx_power_dbm`
- `rx_power_dbm`
- `noise_power_dbm`
- `interference_power_dbm`
- `interference_tx_ids`
- `sinr_db`
- `visibility_regime`

Current assumptions:

- geometry visibility comes from Stage 3.1 records;
- shadowing is disabled by default or deterministic when seeded;
- interference is limited to active same-resource transmitters;
- packet success is not computed in the channel layer.

Deferred:

- finite-blocklength packet reliability;
- point-to-point latency and energy;
- network delivery aggregation;
- application-layer consensus semantics;
- empirical calibration, fading, Doppler, MIMO, HARQ, and scheduler adaptation.

## Stage 3.3 / Stage 3.6 Link Transmission Regime

The active link transmission regime is `urlcc_finite_blocklength_v1`,
represented in `src/marl_topology/link/transmission.py`.

Current link transmission outputs:

- `payload_bits`
- `bandwidth_hz`
- `effective_rate_bps`
- `propagation_delay_s`
- `transmission_delay_s`
- `processing_delay_s`
- `queueing_delay_s`
- `p2p_latency_s`
- `tx_power_w`
- `rx_circuit_power_w`
- `processing_power_w`
- `tx_energy_j`
- `rx_energy_j`
- `processing_energy_j`
- `p2p_energy_j`
- `packet_error_probability`
- `packet_success_probability`
- `target_reliability`
- `required_transmission_time_s`
- `attempt_duration_s`
- `max_attempts_within_deadline`
- `deadline_delivery_probability`
- `expected_attempts`
- `expected_latency_s`
- `expected_energy_j`
- `finite_blocklength_regime_id`
- `transmission_attempted`

Current assumptions:

- channel records come from Stage 3.2;
- packet success probability follows the finite-blocklength normal
  approximation using SINR, bandwidth, payload, and transmission duration;
- inverse reliability mode solves the minimum required transmission time for a
  target reliability;
- deadline retransmission uses `K = floor(deadline_s / attempt_duration_s)`;
- queueing delay is an explicit placeholder, defaulting to zero;
- tx energy uses the channel transmit power and transmission duration;
- receive and processing energy use declared circuit/processing powers;
- expected latency and expected energy are coupled through expected attempts;
- unselected or inactive transmissions consume no link transmission energy.

Deferred:

- network route or broadcast aggregation;
- MAC backoff, HARQ combining, and adaptive coding;
- application-layer deadline and consensus semantics.

## Stage 3.4 Network Communication Regime

The active network communication regime is `stage3_network_communication_v1`,
represented in `src/marl_topology/network/communication.py`.

Current network communication outputs:

- `selected_edge_ids`
- `active_transmission_ids`
- `interference_group_ids`
- `reachable_node_ids`
- `route_node_ids`
- `route_edge_ids`
- `hop_records`
- `network_delivery_probability`
- `network_latency_s`
- `network_energy_j`
- `hop_count`
- `is_full_graph_baseline`
- `is_oracle`

Current assumptions:

- selected edges are eligible communication links, not guaranteed delivery;
- route primitive uses a deterministic shortest selected-edge path;
- broadcast primitive activates selected edges in the source-reachable
  component;
- same-resource background transmissions may interfere;
- orthogonal resources are excluded from interference;
- latency and energy aggregation rules are declared on the network record.

Deferred:

- PBFT message scheduling and quorum semantics;
- application-level deadline satisfaction;
- routing convergence, retransmission policy, congestion control, and learned
  topology policies.
