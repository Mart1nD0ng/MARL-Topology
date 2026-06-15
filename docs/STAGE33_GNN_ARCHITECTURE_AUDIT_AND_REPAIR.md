# Stage 33 GNN Architecture Audit and Repair

## Controlled Object

The controlled object is the local actor architecture registry and full local message-passing GNN component behavior.

## Baseline Audit

Current v2 GNN:

- uses local ego-graph actor-safe rows;
- uses edge encoders and node-message aggregation;
- supports variable local graph size and masks;
- preserves local-group permutation invariance;
- does not consume global topology, oracle labels, reward values, or critic outputs.

Stage33 kept v2 only as archived/diagnostic after closeout.

## Implemented Variants

Variant A: `local_message_passing_gnn_edge_scorer_v3_residual_norm`

- residual connections;
- per-layer LayerNorm;
- configurable message-passing depth;
- edge-to-node and node-to-edge updates;
- better Xavier initialization;
- mask support;
- active Stage33 production registry entry.

Variant B: `local_role_resource_aware_gnn_edge_scorer_v3`

- explicit role/resource encoder and gate from actor-safe local features;
- edge update plus node update over local ego graph;
- no global objective or oracle fields;
- inactive/diagnostic after Stage33 because it did not pass.

## Tests

`tests/unit/test_stage33_gnn_architecture_variants.py` checks:

- output changes when local graph grouping changes;
- node/edge feature changes affect output;
- role/resource columns affect the role-aware variant;
- permutation invariance within grouped local graphs;
- mask support;
- forbidden actor field rejection.

## Run Result

No v3 variant passed the Stage33 training gate. The active residual-norm v3 collapsed in all five seeds for all three fixed optimization configs. The role/resource v3 also collapsed in all five seeds under the selected warmup config.

The metric winner was archived v2, not the active production candidate. Therefore no production GNN was promoted.

## Low-Entropy Closeout

Exactly one active Stage33 production GNN remains in the registry: `local_message_passing_gnn_edge_scorer_v3_residual_norm`.

That registry entry does not mean the actor artifact passed. It means the active direction is GNN v3 for the next owner-approved repair. Stage33 FAIL prevents claiming a production actor artifact.

## Residual Risk

The active v3 architecture may need a smaller update surface, sampler/logprob scale repair, or value normalization repair before any further architecture expansion. Stage33 did not authorize more variants.
