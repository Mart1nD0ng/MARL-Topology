# Post-Task Self-Review - Stage 7 Completion Learning Evidence Exit Gate

## Completed Task

Stage 7 Completion - Learning Evidence Dataset Build + Quality Exit Gate.

## Intended Desired State

Close Stage 7 only after it answers the pre-model data-layer questions:

- what topologies are good;
- which edges are useful to add, remove, keep, or swap;
- whether actor-local observations cover candidate-edge information;
- what centralized critic context is available;
- whether the dataset contains learnable signal before actor/critic design.

## Actual Achieved State

- Added `src/marl_topology/data/learning_evidence_completion.py`.
- Extended edge-delta targets with `swap_edge` and training-only surrogate
  diagnostic deltas.
- Added `scripts/replay/stage7_completion_evidence_dataset_report.py`.
- Added `docs/STAGE7_COMPLETION_LEARNING_EVIDENCE_EXIT_GATE.md`.
- Added `harness/tasks/stage7_completion_learning_evidence_exit_gate.yaml`.
- Added unit and contract tests for the Stage 7 completion exit criteria.
- Updated `docs/PROJECT_STATE.md` and earlier result-save guard tests to allow
  the new evidence-only artifact scope.
- Wrote an evidence-only artifact under
  `result_save/evidence_dataset_only/stage7_completion_learning_evidence_dataset_v1`.

## Evidence

The Stage 7 completion report shows:

- 5 scenarios;
- 29 scenario/topology evidence rows;
- 367 edge-delta learning targets;
- 8 feasible and 21 infeasible rows under tau 0.9;
- topology variants: empty, random, sparse heuristic, full graph, greedy, and
  oracle candidate where feasible;
- add/remove/keep/swap target coverage;
- 234 nonzero edge-delta targets;
- sparse-vs-full trade-off cases;
- actor-observable edge coverage: 1.0;
- actor leakage issues: 0;
- all 10 Stage 7 exit criteria pass.

Artifact files:

- `manifest.json`;
- `learning_evidence.json`;
- `quality_report.json`.

## Tests

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

Result:

```text
594 passed
Task validation passed: 67 tasks
```

## Gates Passed

- Scenario/topology evidence rows gate.
- Oracle/heuristic label gate.
- Add/remove/keep/swap edge-delta target gate.
- Actor-safe / critic / target / diagnostics separation gate.
- Learnable signal gate.
- Oracle label actor-leakage gate.
- Surrogate diagnostic actor-leakage gate.
- Manifest-validated evidence artifact gate.
- No-training gate.
- No-model-implementation gate.

## Gates Deferred

- Stage 8 owner approval.
- Actor policy interface contract.
- Actor/critic/GNN/LSTM implementation.
- Training execution.
- Checkpoint creation.
- Final tau selection.

## New Risks

- The Stage 7 completion dataset is still small and deterministic. It is enough
  for interface design, not enough for training-scale claims.
- Actor-observable predictability is only a structural risk estimate. It is not
  a trained predictability model.
- Stage 8 must remain contract-first and must not jump directly to model code.

## Regressions Checked

- v5 was not modified or migrated.
- No training was run.
- No actor, critic, COMA, GNN, LSTM, checkpoint, or model implementation was
  added.
- Full graph remains a baseline, not an oracle.
- Oracle labels and surrogate diagnostics are absent from actor-safe rows.

## Candidate Next Tasks

- `stage_8_0_actor_policy_interface_contract_with_owner_approval`
- `stage_8_1_actor_policy_interface_skeleton_after_contract`
- `future_large_scale_evidence_generation_after_interface_contract`

## Recommended Next Task

`stage_8_0_actor_policy_interface_contract_with_owner_approval`

Reason: Stage 7 now closes on the intended data-layer evidence rather than on
smoke-only rows. The next safe step is an actor policy interface contract, not
model implementation or training.

## Owner Decision Required

Yes. Codex must not self-authorize Stage 8.
