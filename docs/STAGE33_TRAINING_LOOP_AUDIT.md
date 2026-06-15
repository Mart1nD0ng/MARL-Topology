# Stage 33 Training Loop Audit

## Controlled Object

The controlled object is the production training loop boundary: official MAPPO modules, Stage32/32a custom code, scripts that can start training, and documentation that indicates the recommended active path.

## Audit Result

Official loop:

- `marl_topology.training.mappo.trainer.Stage24LoopConfig`
- `marl_topology.training.mappo.stage25_pilot._stage25_loop_config`
- `marl_topology.training.mappo.advantages.compute_gae_returns`
- `marl_topology.training.mappo.losses.clipped_policy_value_loss`
- `marl_topology.training.mappo.stage28_repaired_critic_pilot._collect_repaired_rollout_batch`
- `marl_topology.training.mappo.stage28_repaired_critic_pilot._loss_for_repaired_indices`

Deprecated loop:

- `marl_topology.training.stage32_production_training.train_production`
- `scripts/train/stage32_production_training_run.py`

Production Stage33 caller:

- `scripts/train/stage33_gnn_stability_repair_training.py`
- `src/marl_topology/training/production_mappo_adapter.py`

## Changes

`src/marl_topology/training/stage32_production_training.py` now declares:

- `STAGE32_CUSTOM_LOOP_ACTIVE_PRODUCTION_PATH = False`
- `STAGE32_CUSTOM_LOOP_STATUS = "archived_inactive_reproducibility_only_stage33"`

`scripts/train/stage32_production_training_run.py` now requires `--allow-inactive-stage32-legacy-loop` and tells users to use the Stage33 script for current production training.

The Stage33 adapter reports:

- official rollout collector: `_collect_repaired_rollout_batch`
- official advantage function: `compute_gae_returns`
- official loss function: `clipped_policy_value_loss`
- official repaired critic loss adapter: `_loss_for_repaired_indices`
- Stage32 custom loop called: `False`

## Code-Doc Consistency Check

The reported Stage32a node-count range and the Stage31 generator code both used 4..8. Stage33 did not silently rewrite that history. Stage33 introduces a new graph-structure dataset with 6..10 node choices.

## Acceptance

Accepted for Stage33 integration: official-loop path is active, Stage32 custom loop is archived, and contract tests check that Stage33 does not import or call `train_production`.

## Residual Risk

Historical docs still mention Stage32 reproduce commands. They are preserved as historical reports, not active production guidance.
