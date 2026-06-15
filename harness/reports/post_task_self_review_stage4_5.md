# Post-Task Self-Review: Stage 4.5 Baseline And Oracle-Candidate Review

## Completed Task

Stage 4.5 - baseline and oracle-candidate review using Stage 3 communication
records and Stage 4 expected-initiator PBFT reliability.

## Intended Desired State

The project should have a deterministic review sensor that compares empty,
sparse, and full topology baselines under the Stage 4.4 PBFT reliability model,
keeps full graph as a baseline rather than an oracle, distinguishes baseline
failure from infeasibility, reports only registered metric concepts, and avoids
reward, training, actor/critic, model, or v5 migration paths.

## Actual Achieved State

Stage 4.5 now provides
`src/marl_topology/evaluation/stage4_baseline_oracle_review.py`. It builds a
four-node reference scenario, evaluates `empty`, `sparse_star`, `sparse_chain`,
and `full` baselines, adapts Stage 3 directed communication records into PBFT
phase matrices, and evaluates `pbft_expected_initiator_mean_field_v1`.

It also performs a bounded small-graph oracle-candidate feasibility search. The
oracle-candidate is marked as non-full and not a deployment actor input. Full
graph is still reported only as a baseline.

## Evidence

- `src/marl_topology/evaluation/stage4_baseline_oracle_review.py`
- `src/marl_topology/evaluation/__init__.py`
- `scripts/replay/stage4_5_baseline_oracle_review.py`
- `tests/unit/test_stage4_5_baseline_oracle_review.py`
- `tests/contract/test_stage4_5_baseline_oracle_review_contract.py`
- `docs/STAGE4_5_BASELINE_ORACLE_REVIEW.md`
- `docs/STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md`
- `docs/PROTOCOL_CONTRACT.md`
- `docs/METRIC_CONTRACT.md`
- `docs/PROJECT_STATE.md`
- `harness/tasks/stage4_5_baseline_oracle_review.yaml`

## Tests

- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`
- `python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json`
- `python scripts\replay\stage4_5_baseline_oracle_review.py`

## Gates Passed

- `stage4_5_baseline_oracle_review_gate`
- `stage4_4_expected_initiator_pbft_gate`
- `stage4_stage3_message_matrix_adapter_gate`
- `stage3_6_urlcc_finite_blocklength_gate`
- `metric_governance_gate`
- `consensus_protocol_naming_gate`
- `full_mask_not_oracle_gate`
- `oracle_before_infeasible_gate`
- `training_precondition_gate`
- `phase_script_entropy_gate`
- `post_task_self_review`

## Gates Deferred

- Stage 4.6 protocol latency and energy accounting review.
- Stage 4.5a broader baseline/oracle-candidate boundary fixtures.
- Stage 4.4a expected-initiator PBFT boundary fixtures.
- Stage 3.6a finite-blocklength boundary fixtures.
- Reward implementation, training, actor/critic, COMA, GNN, and LSTM work.

## New Risks

- The oracle-candidate search optimizes sparse feasibility first. It does not
  prove latency/energy optimality.
- The reference scenario is intentionally small and deterministic.
- Stage 4.5 latency and energy are summed over all directed pair records across
  three phases. This is a review aggregation, not yet a protocol accounting
  contract.
- Stage 4.4 reliability still uses the mean-field approximation.

## Regressions

No reward, training, actor, critic, COMA, GNN, LSTM, checkpoint handling, or v5
code migration was added. The review does not export new metric names and does
not label full graph as oracle.

## Candidate Next Tasks

- Stage 4.6 - protocol latency and energy accounting review.
- Stage 4.5a - more baseline/oracle-candidate boundary fixtures.
- Stage 4.4a - expected-initiator PBFT asymmetric boundary fixtures.
- Stage 3.6a - finite-blocklength boundary fixtures.

## Recommended Next Task

Stage 4.6 - protocol latency and energy accounting review.

Reason: Stage 4.5 can now compare topology reliability and baseline behavior.
Before reward or training, protocol-level latency and energy accounting must be
made explicit so objective comparisons do not rely on review-only aggregation.

## Owner Decision Required

Yes. Codex may recommend Stage 4.6, but user approval is required before
executing the next task.
