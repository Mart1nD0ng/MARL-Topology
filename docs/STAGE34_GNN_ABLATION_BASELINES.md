# Stage 34 GNN Ablation Baselines

Implementation: `src/marl_topology/models/gnn_ablation_registry.py`.

Exactly seven ablations are registered:

1. `v2_reference`
2. `v2_plus_residual`
3. `v2_plus_norm`
4. `v2_plus_role_features`
5. `v2_plus_resource_features`
6. `v2_plus_depth_2_3`
7. `v3_current`

All are local message-passing actors and output edge scores only. None uses global topology, critic outputs, oracle labels, recurrent state, or a production MLP fallback.

Before full Stage34 selection, active Stage34 production count is zero. This avoids promoting a failed variant prematurely.
