# Post-Task Self-Review: Stage 4.1 Heterogeneous Quorum-Tail Utility

## Completed Task

Stage 4.1 - heterogeneous quorum-tail utility.

## Intended Desired State

The project should have a standalone, tested utility for evaluating the
probability that at least `k` heterogeneous Bernoulli inputs succeed. It must
avoid success-subset enumeration, random sampling, Monte Carlo simulation,
learned prediction, v5 imports, reward logic, training logic, and PBFT
three-phase implementation.

## Actual Achieved State

Stage 4.1 now provides `src/marl_topology/protocol/quorum_tail.py` with a
generating-polynomial coefficient evaluator, diagnostic result record, and
conservative largest-probability filter. It is exported from the protocol
package and documented as a standalone utility. It is not connected to PBFT
three-phase reliability, Stage 3 message matrices, reward, or training.

## Evidence

- `src/marl_topology/protocol/quorum_tail.py`
- `tests/unit/test_quorum_tail_stage4_1.py`
- `tests/contract/test_stage4_1_quorum_tail_contract.py`
- `docs/STAGE4_1_QUORUM_TAIL_UTILITY.md`
- `docs/PROTOCOL_CONTRACT.md`
- `docs/STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md`
- `harness/tasks/stage4_heterogeneous_quorum_tail_utility.yaml`
- `docs/PROJECT_STATE.md`

## Tests

- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`
- `python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json`

## Gates Passed

- `stage4_heterogeneous_quorum_tail_utility_gate`
- `stage4_pbft_application_consensus_plan_gate`
- `metric_governance_gate`
- `consensus_protocol_naming_gate`
- `phase_script_entropy_gate`
- `training_precondition_gate`
- `post_task_self_review`

## Gates Deferred

- Stage 4.2 PBFT three-phase reliability record.
- Stage 4.3 Stage 3 message-delivery matrix adapter.
- Stage 4.1a log-space review for larger committees or extreme probabilities.
- Reward contract update.
- MARL training, actor/critic, COMA, GNN, and LSTM work.

## New Risks

- Probability-space arithmetic is sufficient for current tests but may need a
  log-space variant for larger committees or very small probabilities.
- The conservative largest-probability filter is a clean v0 policy choice, not
  a full adversarial scheduling model.
- Future PBFT integration must document the mean-field independence boundary.

## Regressions

No Stage 3 communication behavior was changed. No PBFT cascade, reward,
training, actor/critic, COMA, or v5 migration was added.

## Candidate Next Tasks

- Stage 4.2 - PBFT three-phase reliability record using declared matrices.
- Stage 4.1a - log-space quorum-tail review before larger committee support.
- Stage 4.0a - PBFT plan refinement if owner wants semantic changes.

## Recommended Next Task

Stage 4.2 - PBFT three-phase reliability record.

Reason: Stage 4.1 now provides the standalone quorum-tail building block. The
next safe actuator is a PBFT three-phase record that consumes declared
pre-prepare, prepare, and commit matrices only, still without Stage 3 adapters,
reward, or training.

## Owner Decision Required

Yes. Codex must not self-authorize Stage 4.2 or any follow-up task.
