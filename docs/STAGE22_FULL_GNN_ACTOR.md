# Stage 22 - Full GNN Actor

Stage 22 replaced the active toy edge-set aggregation GNN with a local
message-passing GNN.

## Active Model

Active model id: `local_message_passing_gnn_edge_scorer_v2`.

Archived toy id: `local_gnn_edge_scorer_v1_archived_toy_not_active`.

The public `LocalGNNEdgeScorer` import is preserved, but the implementation is
now message-passing.

## Architecture

The active GNN uses:

- local ego candidate graph scope only;
- edge encoder;
- ego-node and neighbor-node encoders;
- at least two message-passing layers;
- edge-to-node message aggregation;
- node-to-edge updates;
- permutation-invariant mean aggregation within local graph groups;
- padded mask support;
- edge-score outputs only.

It does not output final topology and does not consume global topology,
objective metrics, oracle labels, critic outputs, future outcomes, reward
surrogates, COMA inputs, or Transformer components.

## Result

The full GNN was trained in the Stage 22 A/B trial for both
`directed_outgoing_v1` and `undirected_physical_link_v1`. Under the selected
physical-link semantics it reached projected tau-feasible rate `0.7`.

policy-gradient was not run in Stage 22.
