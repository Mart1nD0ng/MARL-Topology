# Stage 27 Critic Architecture Repair

Implemented candidates:

- `enriched_centralized_mlp_value_critic_v1`: enriched pre-action feature vector, normalized value head, and auxiliary feasibility/consensus/latency/energy heads.
- `centralized_message_passing_graph_value_critic_v1`: centralized training-only graph critic with node encoder, edge encoder, two message-passing layers, graph pooling, normalized value head, and auxiliary heads.

The graph critic was implemented safely and passed the structure-sensitive tests. It is not a Transformer and is not exposed to the actor.
