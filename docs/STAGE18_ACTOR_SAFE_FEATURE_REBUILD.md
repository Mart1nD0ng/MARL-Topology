# Stage 18 Actor-Safe Feature Rebuild

## Controlled Object

The controlled object is the actor input view used by future edge scorers.
Stage 18 extends the actor-safe view from one agent observation to one
agent-edge observation with local topology, projection, resource, message, and
link-estimate context. It does not add global topology, oracle labels,
consensus success probability, latency, energy, reward surrogate, future
outcome, or critic-only targets to actor input.

## Feature Schema

Schema id: `stage18_actor_safe_local_feature_schema_v1`.

Each feature is declared with `source` and `actor_safe = true`.

## Local Topology History

| Feature | Source | Actor Safe |
| --- | --- | --- |
| `edge_active_prev` | `local_history` | true |
| `edge_active_current_local` | `local_projection_feedback` | true |
| `local_outgoing_degree_prev` | `local_history` | true |
| `local_incoming_degree_estimate` | `local_projection_feedback` | true |
| `local_selected_edge_count` | `local_projection_feedback` | true |
| `last_action_proposed` | `local_history` | true |
| `last_action_accepted` | `local_history` | true |
| `last_projection_rejection_reason` | `local_projection_feedback` | true |
| `recent_rejection_reason_histogram` | `local_projection_feedback` | true |
| `previous_selected_neighbor_summary` | `local_history` | true |

These fields are limited to this agent's incident edges, local history, and
local projection feedback. They do not expose the full selected topology.

## Local Resource And Conflict Context

| Feature | Source | Actor Safe |
| --- | --- | --- |
| `tx_budget_used` | `local_projection_feedback` | true |
| `tx_budget_remaining` | `local_projection_feedback` | true |
| `rx_capacity_estimate_for_neighbor` | `local_message` | true |
| `channel_slot_available` | `local_link_estimate` | true |
| `local_channel_slot_occupancy` | `local_projection_feedback` | true |
| `local_conflict_group_occupancy` | `local_projection_feedback` | true |
| `local_interference_estimate` | `local_link_estimate` | true |
| `recent_local_resource_rejection_count` | `local_projection_feedback` | true |
| `neighbor_role_summary` | `local_message` | true |

These are local estimates or declared resource summaries. They are not global
objective values and are not oracle repair signals.

## Local Communication Estimates

| Feature | Source | Actor Safe |
| --- | --- | --- |
| `estimated_link_success_probability` | `local_link_estimate` | true |
| `estimated_deadline_delivery_probability` | `local_link_estimate` | true |
| `estimated_p2p_latency` | `local_link_estimate` | true |
| `estimated_p2p_energy` | `local_link_estimate` | true |
| `estimated_sinr` | `local_link_estimate` | true |
| `estimated_los_nlos` | `local_link_estimate` | true |
| `estimated_required_transmission_time` | `local_link_estimate` | true |
| `estimated_retransmission_attempts` | `local_link_estimate` | true |

These are Stage 3 local link estimates or local proxies derived from link
estimates. Stage 4 `consensus_success_probability` remains forbidden.

## Local Message And History

| Feature | Source | Actor Safe |
| --- | --- | --- |
| `recent_message_success_rate_local` | `local_message` | true |
| `recent_retry_count_local` | `local_history` | true |
| `recent_neighbor_response_summary` | `local_message` | true |
| `local_history_embedding_fields` | `local_history` | true |
| `time_since_last_successful_local_delivery` | `local_history` | true |

These fields summarize past local information only. Future delivery outcomes
remain forbidden.

## Forbidden Actor Fields

The Stage 18 actor-safe feature validator rejects global topology, selected
edge lists, oracle labels, centralized critic state, consensus metrics,
latency, energy, reward surrogate fields, edge-delta targets, future outcomes,
and global objectives.

## Implementation

Durable logic lives in:

`src/marl_topology/data/actor_feature_rebuild.py`

The builder emits edge-level actor-safe rows and validates each row before it
enters Stage 18 evidence.

