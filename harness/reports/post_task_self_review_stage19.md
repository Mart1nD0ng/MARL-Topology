# Post-Task Self-Review - Stage 19

## Completed Task

Stage 19 supervised actor stack rerun on Stage 18 disambiguated evidence.

## Intended Desired State

Run the approved actor-only supervised stack in order:
`MLP -> GNN -> GRU real multi-step subset -> LSTM after GRU sanity`.
Keep actor inputs actor-safe, use Stage 18 soft utility/ranking targets, avoid
critic training and policy-gradient updates, and write no checkpoints or
training artifacts.

## Actual Achieved State

Stage 19 completed. The rerun consumed `850` actor samples, `31` actor-safe
features, and `412` ranking pairs from
`stage18_disambiguated_actor_evidence_v1`.

- MLP: validation loss `0.6004618406`, pairwise accuracy `0.9358974359`.
- GNN: validation loss `0.5954169631`, pairwise accuracy `0.9358974359`.
- GRU: real multi-step final loss `0.5235753655`, sanity passed.
- LSTM: ran after GRU sanity and reached final loss `0.5229729414`.

GNN is the recommended starting actor for the next supervised policy
evaluation gate because it had the lowest validation loss among MLP/GNN.

## Evidence

- `python scripts\train\stage19_supervised_actor_stack.py`
- `docs/STAGE19_SUPERVISED_ACTOR_STACK_RERUN.md`
- `harness/tasks/stage19_supervised_actor_stack_rerun.yaml`
- `tests/unit/test_stage19_supervised_actor_stack.py`
- `tests/contract/test_stage19_supervised_actor_stack_contract.py`

## Tests And Gates

Stage 19 added tests for:

- Stage 18 actor-safe feature tensorization into the Stage 19 schema;
- ranking-pair generation;
- real multi-step temporal subset construction;
- MLP/GNN/GRU/LSTM execution order;
- no checkpoint or artifact write from the Stage 19 script;
- PROJECT_STATE and harness registry synchronization.

Full validation commands are:

- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`

## Gates Passed

- Completion Gate: passed.
- Actor-safe input gate: passed.
- Stage 18 soft/ranking target gate: passed.
- GRU-before-LSTM gate: passed.
- No policy-gradient/checkpoint/artifact gate: passed.
- Harness registration gate: passed.
- PROJECT_STATE sync gate: passed.

## Gates Deferred

- Environment-side assembler policy evaluation remains deferred to Stage 20.
- PPO/MAPPO remains blocked.
- COMA remains blocked.
- Transformer remains blocked.
- Scale-up training remains blocked.
- Reward-weight tuning and final tau selection remain blocked.
- Checkpoint creation remains blocked.

## New Risks

- GNN improves validation loss only slightly over MLP, so architecture choice is
  not final.
- GRU/LSTM evidence is real multi-step but still small.
- Stage 19 does not prove deployment feasibility because the trained scores
  have not been evaluated through the topology assembler.

## Regression Check

The actor input boundary remains Dec-POMDP local-only. Global objective,
oracle, consensus, latency, energy, reward, future-outcome, and critic-only
targets are not actor inputs. The environment-side assembler remains
responsible for hard topology activation.

## Candidate Next Tasks

- `stage_20_supervised_actor_policy_evaluation_with_environment_assembler`
- owner decision on whether the recurrent actors need more real multi-step
  evidence before any temporal deployment candidate is considered

## Recommended Next Task

`stage_20_supervised_actor_policy_evaluation_with_environment_assembler`

## Owner Decision Required

`true`

Codex must not self-authorize Stage 20.
