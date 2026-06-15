# Post-Task Self-Review - Stage 20

## Completed Task

Stage 20 supervised actor policy evaluation with the environment-side topology
assembler.

## Intended Desired State

Evaluate Stage 19 supervised actor scores through `ConflictAwareGreedyAssembler`
and the registered topology evaluator. Keep actor outputs as edge scores only,
keep the assembler as hard topology owner, and do not run PPO/MAPPO, COMA,
Transformer, critic rerun, scale-up training, checkpoint writing, training
artifact writing, reward tuning, final tau selection, or `v5` modification.

## Actual Achieved State

Stage 20 completed. MLP and GNN were evaluated on all `59` Stage 18 rows. GRU
and LSTM were evaluated on the real multi-step subset of `18` rows. GNN was the
best full-row actor after projection, but it did not meet policy-gradient
readiness because its tau-feasible rate was below the greedy reliability
baseline.

## Evidence

- `python scripts\train\stage20_supervised_actor_policy_evaluation.py`
- `docs/STAGE20_SUPERVISED_ACTOR_POLICY_EVALUATION.md`
- `harness/tasks/stage20_supervised_actor_policy_evaluation.yaml`
- `tests/unit/test_stage20_supervised_actor_policy_evaluation.py`
- `tests/contract/test_stage20_supervised_actor_policy_evaluation_contract.py`

## Tests And Gates

Stage 20 added tests for:

- Stage 20 report generation through the environment-side assembler;
- MLP/GNN full-row evaluation;
- GRU/LSTM real multi-step subset evaluation;
- assembler rejection diagnostics;
- full graph baseline not being oracle;
- oracle diagnostic not entering actor input;
- policy-gradient readiness remaining blocked when assembler metrics lag the
  greedy baseline;
- no checkpoint or artifact write from the Stage 20 script.

## Gates Passed

- Completion Gate: passed.
- Environment-side assembler gate: passed.
- Actor edge-score boundary gate: passed.
- Baseline/oracle separation gate: passed.
- No policy-gradient/checkpoint/artifact gate: passed.
- Harness registration gate: passed.
- PROJECT_STATE sync gate: passed.

## Gates Deferred

- Policy-gradient readiness: failed and deferred.
- PPO/MAPPO remains blocked.
- COMA remains blocked.
- Transformer remains blocked.
- Scale-up training remains blocked.
- Reward-weight tuning and final tau selection remain blocked.
- Checkpoint creation remains blocked.

## New Risks

- Supervised actors still propose many edges that the assembler rejects.
- GNN improves over MLP after projection but remains below the greedy
  reliability baseline.
- GRU/LSTM results are promising only on the real multi-step subset and should
  not be generalized to the full policy yet.

## Regression Check

The actor input boundary remains local-only. The deployment assembler did not
consume objective, oracle, reward, consensus, latency, or energy metrics. Full
graph remains a baseline, and oracle output remains diagnostic-only.

## Candidate Next Tasks

- `stage_21_assembler_aware_supervised_target_refinement`
- owner decision on whether temporal evidence should be expanded before
  choosing a recurrent policy candidate

## Recommended Next Task

`stage_21_assembler_aware_supervised_target_refinement`

## Owner Decision Required

`true`

Codex must not self-authorize Stage 21.
