# Network Layer Contract

## Responsibility

Evaluate communication over a selected topology using active transmissions,
resource assignments, interference groups, and simple route or broadcast
primitives.

## Inputs

- candidate communication graph
- selected edge ids
- active transmission records
- channel or resource assignment per active transmission
- point-to-point link transmission records
- source and target ids or broadcast source id
- simple route or broadcast primitive
- optional maximum hop count or route selection rule

## Outputs

- `selected_edge_ids`
- `active_transmission_ids`
- `interference_group_ids`
- `reachable_node_ids`
- per-hop delivery probability records
- `network_delivery_probability`
- `network_latency_s`
- `network_scheduled_latency_s`
- `network_successful_delivery_latency_s`
- `network_energy_j`
- route or broadcast trace
- network-layer diagnostic record

These outputs are network communication records. They are not PBFT quorum
success, consensus reliability, or reward.

## Units

- latency: seconds
- energy: joules
- probabilities: `[0, 1]`
- hop count: count
- ids: stable strings

## Assumptions

- Selected edges define eligible communication links for the network primitive.
- Unselected edges are inactive and must not generate transmit energy.
- Same-resource active transmissions may interfere according to the channel
  contract.
- Orthogonal resources do not contribute to the same interference group.
- Multi-hop latency and energy aggregation must declare sum, max, or another
  explicit rule.
- `network_scheduled_latency_s` records scheduled route or broadcast occupancy
  from attempted communication even when delivery probability is zero.
- `network_successful_delivery_latency_s` records latency only when the network
  delivery probability is positive.
- `network_latency_s` is a compatibility alias for
  `network_successful_delivery_latency_s`; protocol accounting must use
  scheduled latency when it needs phase occupancy.
- Full graph remains a baseline topology, not an oracle or resource optimum.

## Omitted Components

- PBFT message schedule;
- application consensus;
- routing protocol convergence;
- congestion control;
- packet retransmission strategy;
- mobility during a packet;
- learned topology policy.

## Tests

- Unselected edge produces no transmit energy.
- Same-resource active links can lower SINR through interference.
- Orthogonal resource assignment removes interference contribution.
- Disconnected topology lowers delivery probability or reachability.
- Multi-hop path aggregates latency and energy with declared rules.
- Redundant selected topology can increase resource use without being labeled
  oracle.

## Stage 3.4 Implementation Status

Implemented interfaces:

- `NetworkCommunicationConfig`
- `NetworkTransmissionSpec`
- `NetworkHopRecord`
- `NetworkCommunicationRecord`
- `evaluate_network_communication`
- `NetworkCommunicationFixture`

Active network communication regime:

`stage3_network_communication_v1`

Implemented deterministic behavior:

- selected edge ids define eligible communication links;
- route primitive chooses a deterministic shortest selected-edge path to one
  target;
- broadcast primitive activates selected edges in the source-reachable
  component;
- resource assignments are explicit per selected edge, defaulting to
  `resource_0`;
- background active transmissions can interfere when they reuse the same
  resource;
- orthogonal-resource background transmissions are excluded from interference;
- reachable nodes are computed from the selected topology;
- per-hop packet delivery, latency, and energy come from Stage 3.3 link
  transmission records;
- route latency aggregates by sum, broadcast latency aggregates by max, and
  energy aggregates by sum for scheduled communication;
- Stage 4.8 distinguishes scheduled latency from successful-delivery latency:
  failed scheduled messages can have positive `network_scheduled_latency_s` and
  `network_energy_j` while `network_successful_delivery_latency_s` remains
  zero;
- network delivery probability is the product of active per-hop delivery
  probabilities in the current v1 primitive.

Implemented fixtures:

- `multi_hop_delivery`
- `disconnected_reachability`
- `two_transmitters_interference_network`
- `orthogonal_channel_no_interference_network`
- `resource_redundant_topology`

Layer boundary:

`network_delivery_probability`, `network_scheduled_latency_s`,
`network_successful_delivery_latency_s`, `network_latency_s`, and
`network_energy_j` are network communication record fields. They are not PBFT
quorum success, not consensus reliability, not reward, not topology oracle
status, and not training objectives.

Boundary shorthand: not PBFT quorum success, not consensus reliability, not
reward.

Full graph can be marked as `is_full_graph_baseline = true`, but Stage 3.4
records must keep `is_oracle = false`.

Deferred from Stage 3.4:

- PBFT message schedule;
- application consensus semantics;
- application-level deadline satisfaction;
- retransmission policy;
- routing protocol convergence;
- learned topology policy.

## Failure Modes

- Confusing network delivery with application consensus.
- Treating selected edge as guaranteed packet success.
- Hidden resource assignment changes interference.
- Full graph labeled as feasibility oracle.
- Network latency reused as application deadline satisfaction without Stage 4
  contract.

## GOAL_SKELETON Coupling

This contract connects selected topology semantics to future `TopologyEvaluator`
and keeps communication delivery separate from `ConsensusSuccess`.

## Deferred To Stage 4

PBFT quorum success, consensus reliability, application-level deadline
objectives, and consensus reward are intentionally deferred to Stage 4.
