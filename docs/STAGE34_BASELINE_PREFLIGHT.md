# Stage 34 Baseline Preflight

## Read Summary

Stage32 showed unstable GNN training and did not beat the MLP baseline. Stage32a repaired the loss/backward path and used node counts 4..8, but still found GNN seed instability. Stage33 moved production training to the official adapter, added v3 variants and graph-structure data, but used only seven scenarios and one update, so its failure is not a final architecture conclusion.

Official training path for Stage34 remains:

- `marl_topology.training.mappo.stage25_pilot._stage25_loop_config`
- `marl_topology.training.mappo.stage28_repaired_critic_pilot._collect_repaired_rollout_batch`
- `marl_topology.training.mappo.advantages.compute_gae_returns`
- `marl_topology.training.mappo.losses.clipped_policy_value_loss`
- `marl_topology.training.production_mappo_adapter.Stage33ProductionMappoAdapter`

Stage32/32a custom loop remains inactive and guarded as historical reproducibility only.

## Code-Doc Consistency

Stage32a code and docs used 4..8 nodes. Stage34 does not claim that range. Stage34 declares node_count_choices `(6, 7, 8, 9, 10)` in its dataset contract. The default diagnostic script does not materialize the full dataset, so actual materialized node counts remain pending owner-approved full execution.

## Control Decision

The repaired baseline contract is implemented, but empirical baseline repair is blocked until the owner authorizes full materialization/training or changes the protocol.
