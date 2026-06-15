# Stage 33 - GNN Stability Repair and MAPPO Loop Unification

## Controlled Object

The controlled object is the active production actor-training path: Stage24/25/28 MAPPO rollout, GAE, clipped policy/value loss, repaired graph critic adapter, active local GNN actor registry, Plackett-Luce physical-link sampler, selected physical-link assembler, Stage3/4 evaluator, frozen Stage5 reward surrogate, graph-structure Stage33 dataset, manifest-validated reports, and project state.

## Desired State

Production training should no longer route through the Stage32/32a custom loop. It should call the official MAPPO stack, train a GNN production actor under a fixed low-LR stability protocol, compare full GNN variants against an MLP diagnostic baseline on graph-structure scenarios, and select exactly one active production GNN only if the gate passes.

## Baseline

Stage32a was more realistic than Stage32 because it used a real clipped policy/value loss and `loss.backward()`, but it still kept a custom training loop outside the Stage24/25 MAPPO path. Stage32a reported larger graphs using node counts 4..8, and the code matched that range. Stage33 intentionally uses 6..10 node choices through the new graph-structure dataset.

The active Stage24/25/28 MAPPO path already provided official rollout, GAE, clipped loss, and repaired graph-critic value handling. Stage33 added a production adapter instead of continuing Stage32 custom training.

## Implemented Change

- Added `src/marl_topology/training/production_mappo_adapter.py`.
- Added `src/marl_topology/data/stage33_graph_structure_dataset.py`.
- Added GNN v3 full-component variants in `src/marl_topology/models/local_gnn_edge_scorer.py`.
- Updated the model registry so exactly one Stage33 GNN is active for production: `local_message_passing_gnn_edge_scorer_v3_residual_norm`.
- Kept MLP as diagnostic-only.
- Marked Stage32 custom training inactive and guarded the Stage32 legacy script.
- Added Stage33 training and replay scripts.
- Registered `harness/tasks/stage33_gnn_stability_and_mappo_loop_unification.yaml`.

## Sensors

- `python -m pytest -q tests\unit\test_stage33_production_mappo_adapter.py tests\unit\test_stage33_gnn_stability_config.py tests\unit\test_stage33_gnn_architecture_variants.py tests\unit\test_stage33_graph_structure_dataset.py tests\contract\test_stage33_no_stage32_custom_loop_active.py tests\contract\test_stage33_no_reward_tuning_no_lstm_coma_transformer.py tests\contract\test_stage33_gnn_production_gate.py`
- `python scripts\train\stage33_gnn_stability_repair_training.py`
- `python scripts\replay\stage33_gnn_failure_attribution_report.py`
- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`

## Stage33 Run Result

The bounded Stage33 protocol completed and wrote:

- `result_save/stage33_gnn_stability_repair/stage33_fixed_protocol/manifest.json`
- `result_save/stage33_gnn_stability_repair/stage33_fixed_protocol/training_report.json`
- `result_save/stage33_gnn_stability_repair/stage33_fixed_protocol/failure_attribution_report.json`

Verdict: `stage33_gnn_repair_blocked_awaiting_owner_decision`.

Gate issues:

- `selected_gnn_not_active_stage33_production_actor`
- `selected_gnn_collapse_rate_above_20_percent`

The selected metric winner was `local_message_passing_gnn_edge_scorer_v2` with `low_lr_with_warmup`, but v2 is archived/diagnostic after Stage33 and is not the active production actor. Its collapse rate was 0.8. The active v3 residual-norm GNN collapsed in all five seeds for all three fixed configs.

## Completion Gate

Completion Gate passed? No.

Reason: Stage33 implemented the adapter, dataset, tests, reports, and official-loop routing, but no active production GNN passed the stability and selection gate.

## Next-Stage Readiness Gate

Next-stage readiness gate passed? No.

Blocker: the active v3 GNN did not survive the fixed repair protocol, and the only selected GNN was the archived v2 diagnostic variant.

What could make the next stage invalid: treating the Stage33 report as a production artifact, promoting MLP, promoting archived v2, or continuing with more configs without owner approval.

Evidence that next work is safe only as an owner-approved repair task: the official adapter and graph-structure dataset are now test-covered, Stage32 is retired from active path, and the failure report isolates optimization plus architecture-selection blockers.

## Regression Protected

Stage33 protects the non-target behavior that reward weights, tau=0.9, Stage3 reliability, Stage4 PBFT reliability, active sampler, action semantics, and actor input boundary remain unchanged.

## Residual Risk

The bounded run is intentionally small. It is enough to reject production promotion, not enough to rank future architecture repairs. A next task needs owner approval before trying additional optimization or architecture changes.
