# Post-Task Self-Review - Stage 7.0 Learning Evidence Dataset

## Completed Task

Stage 7.0 - Learning Evidence / Dataset Generation Contract and Minimal Evidence Export.

## Intended Desired State

Create a Stage 7 learning evidence boundary that turns current simulator and
topology evaluation records into reproducible in-memory evidence rows while
keeping deployment actor inputs separate from centralized critic context,
learning targets, diagnostics, oracle labels, objective metrics, and future
outcomes.

## Actual Achieved State

- Added `docs/STAGE7_LEARNING_EVIDENCE_DATASET_CONTRACT.md`.
- Added `docs/STAGE7_0_LEARNING_EVIDENCE_DATASET.md`.
- Added `src/marl_topology/data/learning_evidence.py`.
- Added `scripts/replay/stage7_0_learning_evidence_report.py`.
- Added `harness/tasks/stage7_0_learning_evidence_dataset_generation.yaml`.
- Added Stage 7 unit and contract tests.
- Updated `docs/PROJECT_STATE.md`, `docs/TRAINING_CONTRACT.md`,
  `docs/METRIC_CONTRACT.md`, and `docs/DEC_POMDP_CONTRACT.md`.

## Evidence

The Stage 7.0 builder creates:

- `actor_safe_view` rows from Stage 6.1 actor-safe observations;
- `critic_centralized_view` with centralized training-only context;
- `learning_target_view` with edge-delta targets;
- `diagnostics_view` with audit-only metadata.

The writer is guarded by owner approval, Stage 5.10 manifest validation, and
`artifact_scope = evidence_dataset_only`. The replay script reports evidence
without writing project artifacts.

## Tests

Local targeted check:

```powershell
python -m pytest tests\unit\test_learning_evidence_stage7_0.py tests\contract\test_stage7_0_learning_evidence_dataset.py -q
```

Result:

```text
13 passed
```

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

Result:

```text
571 passed
Task validation passed: 65 tasks
```

## Gates Passed

- Actor-safe view leakage gate.
- Learning target separation gate.
- Artifact writer approval and manifest guard gate.
- Metric governance gate.
- Stage 7 project-state correction gate.
- No default project artifact write gate.

## Gates Deferred

- Stage 7 learning evidence data quality gate.
- Stage 8 actor policy interface gate.
- Model implementation gate.
- Training execution gate.
- Checkpoint creation gate.

## New Risks

- The minimal demo evidence rows prove shape and leakage separation, not
  statistical sufficiency for learning.
- Edge-delta targets are one-step topology counterfactuals; they are not yet a
  calibrated credit-assignment target.
- The evidence writer is implemented but default project export is intentionally
  not performed in Stage 7.0.

## Regressions Checked

- `result_save` remains scaffold-only during default report execution.
- v5 remains read-only and no v5 code is migrated.
- No training, model implementation, checkpoint creation, final tau selection,
  or reward weight calibration is introduced.

## Required Answers

- 是否仍未训练？是。Stage 7.0 没有执行训练。
- 是否仍未实现模型？是。Stage 7.0 没有实现 actor、critic、GNN、LSTM 或其他模型。
- 是否生成了学习证据？是。已生成 in-memory learning evidence rows 和 edge-delta learning targets；默认脚本不写项目 artifact。
- 是否 actor-safe view 与 learning target 分离？是。`actor_safe_view` 与 `learning_target_view` 分离，并有负向泄漏测试。
- Stage 8 是否仍应等待数据质量报告？是。Stage 8 actor policy interface 应等待 Stage 7.1 data quality report。

## Candidate Next Tasks

- `stage_7_1_learning_evidence_data_quality_report_with_owner_approval`
- `stage_7_2_evidence_artifact_export_run_with_owner_manifest_approval`
- `stage_8_0_actor_policy_interface_contract_after_data_quality_report`

## Recommended Next Task

`stage_7_1_learning_evidence_data_quality_report_with_owner_approval`

Reason: Stage 7.0 has created the evidence boundary and minimal rows. The next
sensor should evaluate data coverage, leakage safety, target usefulness, and
fixture diversity before Stage 8 actor policy interface work.

## Owner Decision Required

Yes. Codex must not self-authorize Stage 7.1 or Stage 8.
