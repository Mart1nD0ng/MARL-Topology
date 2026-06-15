# Post-Task Self-Review: Stage 4.2 PBFT Three-Phase Reliability Record

## Completed Task

Stage 4.2 - PBFT three-phase reliability record.

## Intended Desired State

The project should have a tested PBFT reliability record that consumes declared
pre-prepare, prepare, and commit message-delivery matrices, uses the Stage 4.1
heterogeneous quorum-tail utility, validates PBFT quorum assumptions, keeps
mean-field and fault-filter boundaries explicit, and avoids Stage 3 adapters,
reward, timeout-gated aliases, training, actor/critic code, sampling, subset
enumeration, and v5 migration.

## Actual Achieved State

Stage 4.2 now provides `src/marl_topology/protocol/pbft_reliability.py` and
exports it from the protocol package. The evaluator computes
`pre_prepare_readiness`, `prepared_probability`, `committed_probability`, and
round-level `consensus_success_probability` from declared matrices only. It
validates `n >= 3f + 1`, uses `2f + 1` total quorum and `2f` external quorum,
and supports no-filter or conservative largest-probability fault filtering.

## Evidence

- `src/marl_topology/protocol/pbft_reliability.py`
- `tests/unit/test_pbft_reliability_stage4_2.py`
- `tests/contract/test_stage4_2_pbft_reliability_contract.py`
- `docs/STAGE4_2_PBFT_THREE_PHASE_RELIABILITY.md`
- `docs/PROTOCOL_CONTRACT.md`
- `docs/METRIC_CONTRACT.md`
- `docs/STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md`
- `harness/tasks/stage4_pbft_three_phase_reliability_record.yaml`
- `docs/PROJECT_STATE.md`

## Tests

- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`
- `python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json`

## Gates Passed

- `stage4_pbft_three_phase_reliability_record_gate`
- `stage4_heterogeneous_quorum_tail_utility_gate`
- `stage4_pbft_application_consensus_plan_gate`
- `metric_governance_gate`
- `consensus_protocol_naming_gate`
- `training_precondition_gate`
- `phase_script_entropy_gate`
- `post_task_self_review`

## Gates Deferred

- Stage 4.3 Stage 3 message-delivery matrix adapter.
- Stage 4.2a additional PBFT boundary fixtures.
- Stage 4.1a log-space quorum-tail review for larger committees.
- Reward contract update.
- MARL training, actor/critic, COMA, GNN, and LSTM work.

## New Risks

- The reliability calculation is a mean-field approximation and does not fully
  model shared-message correlations.
- Probability-space arithmetic may need log-space support for larger committees
  or extreme probabilities.
- Conservative largest-probability filtering is a v0 unknown-fault rule, not a
  full adversarial timing or equivocation model.
- Stage 4.3 must not directly rename Stage 3 network delivery as consensus
  reliability.

## Regressions

No Stage 3 communication behavior was changed. No Stage 3 adapter, deadline
adapter, reward, training, actor/critic, COMA, or v5 code migration was added.

## Candidate Next Tasks

- Stage 4.3 - Stage 3 message-matrix adapter.
- Stage 4.2a - PBFT reliability boundary fixture expansion.
- Stage 4.1a - log-space quorum-tail review.

## Recommended Next Task

Stage 4.3 - Stage 3 message-matrix adapter.

Reason: Stage 4.2 can now evaluate PBFT reliability from declared matrices.
The next safe step is a narrow adapter that derives those matrices from Stage 3
communication records while preserving latency, energy, timeout, and metric
boundaries.

## Owner Decision Required

Yes. Codex must not self-authorize Stage 4.3 or any follow-up task.
