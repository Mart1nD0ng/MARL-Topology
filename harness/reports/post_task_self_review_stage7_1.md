# Post-Task Self-Review - Stage 7.1 Learning Evidence Data Quality Report

## Completed Task

Stage 7.1 - Learning Evidence Data Quality Report with owner approval.

## Intended Desired State

Audit Stage 7.0 learning evidence rows for view coverage, actor leakage, target
coverage, objective metric sanity, topology family coverage, and readiness for
Stage 8 interface-contract design without training, model implementation,
checkpoint creation, artifact export runs, final tau selection, or v5
migration.

## Actual Achieved State

- Added `src/marl_topology/data/learning_evidence_quality.py`.
- Added `scripts/replay/stage7_1_learning_evidence_quality_report.py`.
- Added `docs/STAGE7_1_LEARNING_EVIDENCE_DATA_QUALITY_REPORT.md`.
- Added `harness/tasks/stage7_1_learning_evidence_data_quality_report.yaml`.
- Added Stage 7.1 unit and contract tests.
- Updated `docs/PROJECT_STATE.md`, `docs/TRAINING_CONTRACT.md`,
  `docs/METRIC_CONTRACT.md`, `docs/DEC_POMDP_CONTRACT.md`, and
  `docs/STAGE7_LEARNING_EVIDENCE_DATASET_CONTRACT.md`.

## Evidence

The Stage 7.1 report over the minimal demo evidence shows:

- rows: 4;
- edge-delta targets: 48;
- actor-safe rows checked: 16;
- actor leakage issues: 0;
- required views present;
- weak/sparse/dense topology families present;
- add/remove/keep edge-delta actions present;
- nonzero edge-delta targets: 24;
- blocking issues: 0;
- warnings: no feasible rows at tau 0.9 and low feasibility class diversity.

## Tests

Targeted check:

```powershell
python -m pytest tests\unit\test_learning_evidence_quality_stage7_1.py tests\contract\test_stage7_1_learning_evidence_quality_report.py -q
```

Result:

```text
11 passed
```

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

Result:

```text
582 passed
Task validation passed: 66 tasks
```

## Gates Passed

- View coverage gate.
- Actor leakage gate.
- Metric range gate.
- Topology family coverage gate.
- Edge-delta action coverage gate.
- Edge-delta informativeness gate.
- Stage 7 closure discipline gate.

## Gates Deferred

- Stage 8 actor policy interface owner approval.
- Model implementation gate.
- Training execution gate.
- Checkpoint creation gate.
- Large-scale dataset quality gate.

## New Risks

- The minimal evidence is not training sufficient because it has no feasible
  rows at tau 0.9 and lacks feasibility class diversity.
- Stage 8 should define interfaces only; it should not implement models or run
  training from this evidence alone.

## Regressions Checked

- Default report script writes no project artifacts.
- `result_save` remains scaffold-only.
- v5 remains read-only and no v5 code is migrated.
- No model, checkpoint, training execution, reward weight calibration, or final
  tau selection is introduced.

## Candidate Next Tasks

- `stage_8_0_actor_policy_interface_contract_with_owner_approval`
- `stage_7_defect_fix_only_if_quality_gate_regresses`
- `future_large_scale_evidence_generation_after_stage8_interface_contract`

## Recommended Next Task

`stage_8_0_actor_policy_interface_contract_with_owner_approval`

Reason: Stage 7 is closed for the current control objective. Evidence quality
is sufficient to define the actor policy interface contract, but not sufficient
for training execution.

## Owner Decision Required

Yes. Codex must not self-authorize Stage 8.
