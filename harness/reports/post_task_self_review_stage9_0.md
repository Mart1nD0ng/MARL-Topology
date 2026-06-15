# Post-Task Self-Review - Stage 9.0 Local MLP Edge Scorer Baseline

## Completed Task

Stage 9.0 - Local MLP Edge Scorer Baseline Scaffold, no training execution.

## Intended Desired State

Implement the first minimal actor model scaffold after unifying the active
actor output schema:

- active actor output is edge-score schema;
- `activate` belongs only to legacy local decisions or assembler-selected
  topology;
- `LocalMLPEdgeScorer` consumes only actor-safe local observations;
- no training, checkpoint, artifact, PPO/MAPPO/COMA, GNN/GRU/LSTM/Transformer,
  reward calibration, final tau selection, or v5 migration.

## Actual Achieved State

Stage 9.0 now has:

- active `ACTOR_POLICY_OUTPUT_SCHEMA_ID =
  actor_policy_local_edge_score_output_v1`;
- legacy `actor_policy_local_edge_decision_v1` retained only as legacy
  edge-decision schema;
- `LocalMLPEdgeScorerConfig`, `LocalMLPEdgeScorer`,
  `encode_actor_local_edges`, and `build_stage9_0_local_mlp_report`;
- PyTorch dependency declared in `pyproject.toml`;
- local feature schema using actor kind, local position, and incident-neighbor
  link estimates only;
- frozen parameters by default for Stage 9.0;
- no-gradient scoring to produce `EdgeScoreBatch`;
- Stage 9.0 docs, harness task, tests, and PROJECT_STATE update.

## Evidence

- Actor/critic interfaces remain separated.
- Actor model output is edge scores, not selected topology and not `activate`.
- Environment-side assembler still owns hard topology activation.
- PyTorch imports are limited to `src/marl_topology/policies/local_mlp_edge_scorer.py`.
- Source scan found no training loop, optimizer step, backward execution,
  checkpoint save/load, PPO/MAPPO/COMA classes, GNN/GRU/LSTM/Transformer
  classes, or v5 path in `src`.

## Tests

- `python -m pytest tests\unit\test_stage9_0_local_mlp_edge_scorer.py tests\contract\test_stage9_0_local_mlp_edge_scorer_contract.py tests\unit\test_actor_policy_interface_contract_stage8_0.py -q`
  - passed, 20 tests.
- `python -m pytest -q`
  - passed, 639 tests.
- `python harness\scripts\validate_tasks.py`
  - passed, 70 tasks.
- `rg "optimizer\.step|backward\(|train_loop|torch\.save|torch\.load|class PPO|class MAPPO|class COMA|class GNN|class GRU|class LSTM|class Transformer|D:\\PhD_works\\v5" src -n`
  - no matches.
- `rg "import torch|from torch" src -n`
  - matches only `src/marl_topology/policies/local_mlp_edge_scorer.py`.

## Gates Passed

- active edge-score schema gate;
- legacy activate separation gate;
- Dec-POMDP actor-safe local input gate;
- local MLP scaffold gate;
- no training execution gate;
- no checkpoint/artifact gate;
- no PPO/MAPPO/COMA gate;
- no GNN/GRU/LSTM/Transformer gate;
- v5 no-migration gate;
- harness validation gate.

## Gates Deferred

- supervised edge-scoring warm-start design;
- loss/label/split protocol;
- run-manifest and artifact policy for future training;
- checkpoint format and serialization checks;
- optimizer and learner implementation;
- training execution;
- GNN/GRU/LSTM/Transformer architecture stages;
- COMA optional ablation.

## New Risks

- The untrained MLP produces structurally valid scores but no predictive
  evidence.
- Raw local feature scales are not normalized yet.
- Future training must define labels, losses, splits, seeds, manifests, and
  projected-action semantics before execution.

## Regressions

Protected non-target behavior:

- Stage 6.1 actor-safe input schema remains unchanged.
- Stage 8 assembler still owns hard topology activation.
- Critic outputs and learning targets remain excluded from actor input.
- Full graph remains a baseline, not oracle.
- v5 remains read-only and no v5 code was migrated.

## Candidate Next Tasks

- `stage_9_1_supervised_edge_scoring_warm_start_plan_without_training_execution`;
- `stage_9_1_local_mlp_training_preflight_without_execution`.

## Recommended Next Task

`stage_9_1_supervised_edge_scoring_warm_start_plan_without_training_execution`.

Reason: Stage 9.0 created only the untrained MLP scoring surface. The next
bounded step is to define warm-start labels, losses, splits, manifest needs,
and evaluation sensors before any training execution exists.

## Owner Decision Required

Yes. Stage 9.1 may begin only with explicit owner approval.
