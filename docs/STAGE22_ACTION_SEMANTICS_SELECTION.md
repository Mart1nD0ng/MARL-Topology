# Stage 22 - Action Semantics Selection

Selected active semantics: `undirected_physical_link_v1`.

Archived losing semantics: `directed_outgoing_v1`.

## Decision Rule

The selection key used final-stack evidence only:

1. full-GNN projected tau-feasible rate;
2. lower latency and energy among feasible projected topologies;
3. lower high-score projection rejection;
4. cleaner alignment between actor action and evaluator topology;
5. lower code entropy.

Stage 20 actor metrics were historical diagnostics only.

## Why Option B Won

`undirected_physical_link_v1` aligns the first topology action with the
physical-edge evaluator. Endpoint proposals such as `i->j` and `j->i` are
aggregated into one physical edge by `max_endpoint_score`, preventing duplicate
directional activation.

The selected path reached full-GNN projected tau-feasible rate `0.7`.
`directed_outgoing_v1` reached `0.0` after directed assembler projection and
directed PBFT message-matrix evaluation.

## Cleanup State

The active registry returns only `undirected_physical_link_v1`.
`directed_outgoing_v1` remains only as archived Stage 22 A/B evidence, not an
active deployment path.

policy-gradient was not run in Stage 22.
