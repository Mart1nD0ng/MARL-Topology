# Stage 33 GNN vs MLP Diagnostic Comparison

## Controlled Object

The controlled object is fair comparison across active/diagnostic actors under the same official MAPPO adapter, dataset, sampler, assembler, evaluator, reward surrogate, and seeds.

## Compared Models

- MLP diagnostic baseline: `local_mlp_edge_scorer_v2_model_package`
- active v2 diagnostic GNN: `local_message_passing_gnn_edge_scorer_v2`
- Variant A active production candidate: `local_message_passing_gnn_edge_scorer_v3_residual_norm`
- Variant B diagnostic candidate: `local_role_resource_aware_gnn_edge_scorer_v3`

MLP remains diagnostic only. It cannot be selected as production actor.

## Result

All models failed the reliability/collapse surface in this bounded Stage33 run. The active v3 residual-norm GNN collapsed in every seed for every fixed config. The archived v2 GNN had the best surrogate among GNNs but still collapsed in 4 of 5 seeds and is not active production.

The MLP diagnostic baseline also collapsed in 4 of 5 seeds. Therefore the comparison does not show an MLP production advantage; it shows that the official-loop optimization surface is unstable under this bounded protocol.

## Gate Outcome

The production actor gate failed because:

- selected GNN was not the active Stage33 production actor;
- selected collapse rate was 0.8, above the 0.2 gate.

No MLP fallback was selected.

## Residual Risk

Because all actors had eval tau mean 0.0, the bounded comparison cannot rank graph reasoning quality. It can only reject production promotion and identify the optimization/architecture blocker.
