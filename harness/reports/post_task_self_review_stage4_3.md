# Post-Task Self-Review: Stage 4.3 Stage 3 Message-Matrix Adapter

## Completed Task

Stage 4.3 - Stage 3 message-matrix adapter.

## Intended Desired State

The project should have a narrow, tested adapter that converts Stage 3 network
communication records into PBFT phase message-delivery matrices for the Stage
4.2 reliability evaluator. The adapter must preserve the boundary between
communication-layer delivery records and application-layer consensus metrics,
apply explicit per-phase deadline budgets, keep missing or late messages from
being silently treated as successful, and avoid reward, training, actor/critic,
model, or v5 migration paths.

## Actual Achieved State

Stage 4.3 now provides
`src/marl_topology/protocol/message_matrix_adapter.py` and exports it from the
protocol package. The adapter builds one matrix each for `pre_prepare`,
`prepare`, and `commit` from Stage 3 `NetworkCommunicationRecord` instances.
For each `(source, target)` pair it uses the best observed
`network_delivery_probability` only when `network_latency_s` is within the
declared phase budget; late records are zeroed and missing pairs remain absent.
It emits adapter diagnostics but does not export a new metric or rename Stage 3
delivery probability as consensus success.

## Evidence

- `src/marl_topology/protocol/message_matrix_adapter.py`
- `src/marl_topology/protocol/__init__.py`
- `tests/unit/test_message_matrix_adapter_stage4_3.py`
- `tests/contract/test_stage4_3_message_matrix_adapter_contract.py`
- `docs/STAGE4_3_STAGE3_MESSAGE_MATRIX_ADAPTER.md`
- `docs/STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md`
- `docs/PROTOCOL_CONTRACT.md`
- `docs/METRIC_CONTRACT.md`
- `docs/PROJECT_STATE.md`
- `harness/tasks/stage4_stage3_message_matrix_adapter.yaml`

## Tests

- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`
- `python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json`
- Source boundary scan for v5 imports, reward, P_eff, training loops, models,
  actor/critic, COMA/MAPPO, optimizers, and checkpoint routes.
- Scaffold hygiene scan for `.pytest_cache`, `__pycache__`, and `*.pyc`.
- `result_save` scan for generated artifacts.
- v5 status check or no-git fallback.

## Gates Passed

- `stage4_stage3_message_matrix_adapter_gate`
- `stage4_pbft_three_phase_reliability_record_gate`
- `stage4_heterogeneous_quorum_tail_utility_gate`
- `metric_governance_gate`
- `consensus_protocol_naming_gate`
- `protocol_timeout_gate`
- `stage3_network_layer_gate`
- `training_precondition_gate`
- `phase_script_entropy_gate`
- `post_task_self_review`

## Gates Deferred

- Stage 4.4 baseline and oracle review over PBFT reliability.
- Stage 4.3a additional adapter boundary fixtures if larger topology cases
  reveal ambiguity.
- Stage 4.2a PBFT reliability boundary fixture expansion.
- Stage 4.1a log-space quorum-tail review for larger committees or extreme
  probabilities.
- Reward implementation, training, actor/critic, COMA, GNN, and LSTM work.

## New Risks

- The adapter currently collapses multiple Stage 3 records for the same
  `(source, target)` pair by max probability. This is explicit and tested, but
  future resource scheduling may need a richer selection policy.
- Deadline filtering is per phase and per record; it does not yet model a full
  PBFT round clock with cross-phase accumulated latency.
- The new module introduces a protocol-to-network record dependency. This is
  acceptable for a narrow adapter but should not grow into protocol logic
  depending on Stage 3 internals.
- Stage 4.3 still inherits the Stage 4.2 mean-field independence assumption.

## Regressions

No Stage 3 communication behavior was changed. No reward, training, actor,
critic, COMA, MAPPO, GNN, LSTM, checkpoint handling, result artifact, or v5 code
migration was added. Full graph remains a baseline, not an oracle.

## Candidate Next Tasks

- Stage 4.4 - baseline and oracle review using Stage 4 PBFT reliability.
- Stage 4.3a - add adapter boundary fixtures for larger or asymmetric topology
  records.
- Stage 4.2a - add PBFT reliability boundary fixtures.
- Stage 4.1a - review log-space quorum-tail evaluation for larger committees.

## Recommended Next Task

Stage 4.4 - baseline and oracle review.

Reason: Stage 4.3 now connects Stage 3 communication records to the Stage 4.2
PBFT message matrices. The next safe step is to compare empty, full, sparse,
and oracle-candidate topologies while preserving the rule that full graph is
only a baseline and cannot prove feasibility or optimality.

## Owner Decision Required

Yes. Codex may recommend Stage 4.4, but user approval is required before
executing the next task.
