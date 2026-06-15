# Post-Task Self-Review - Stage 8.0 Actor Policy Interface Contract

## Completed Task

Stage 8.0 - actor policy interface contract with owner approval.

## Intended Desired State

Freeze a Dec-POMDP-safe actor policy interface that consumes Stage 6.1
actor-safe rows and respects Stage 7 evidence separation, without implementing
actor/critic models, training, checkpoints, reward-weight calibration, final
tau selection, artifact export, or v5 migration.

## Actual Achieved State

Stage 8.0 now has:

- a structured contract module at `src/marl_topology/policies/interface_contract.py`;
- a contract document at `docs/STAGE8_0_ACTOR_POLICY_INTERFACE_CONTRACT.md`;
- updates to `POLICY_ARCHITECTURE_CONTRACT.md`, `DEC_POMDP_CONTRACT.md`,
  `TRAINING_CONTRACT.md`, `METRIC_CONTRACT.md`, and `PROJECT_STATE.md`;
- a no-write replay script at
  `scripts/replay/stage8_0_actor_policy_interface_contract.py`;
- a harness task at `harness/tasks/stage8_0_actor_policy_interface_contract.yaml`;
- unit and contract tests for input/output schema boundaries and source
  negative checks.

## Evidence

- Actor input schema equals the Stage 6.1 actor-safe batch fields.
- Actor input validation rejects objective metrics, oracle labels, learning
  targets, surrogate diagnostics, centralized views, global topology, and
  future outcomes.
- Actor output validation accepts local edge-decision fields and rejects global
  topology, selected edge sets, objective metrics, and learning targets.
- Critic centralized views and Stage 7 learning targets remain reference-only
  and training-only.

## Tests

- `python -m pytest tests\unit\test_actor_policy_interface_contract_stage8_0.py -q`
  - `8 passed`
- `python -m pytest tests\contract\test_stage8_0_actor_policy_interface_contract.py -q`
  - `5 passed`
- `python -m pytest -q`
  - `607 passed`
- `python harness\scripts\validate_tasks.py`
  - `Task validation passed: 68 tasks`

## Gates Passed

- `actor_policy_input_schema_gate`
- `actor_policy_output_schema_gate`
- `dec_pomdp_leakage_gate`
- `stage7_target_separation_gate`
- `critic_training_view_separation_gate`
- `model_implementation_block_gate`
- `training_execution_block_gate`
- `metric_governance_gate`
- `post_task_self_review`

## Gates Deferred

- Model-family selection.
- Credit-method selection.
- Actor implementation.
- Critic implementation.
- Training execution.
- Checkpoint creation.
- Reward-weight calibration.
- Final tau selection.

## New Risks

- The interface is still abstract: it does not prove that future model inputs
  are tensorized consistently.
- The output schema permits optional local scores and probabilities, but no
  calibration rule exists yet.
- Stage 8.1 must avoid turning the interface skeleton into a hidden model
  implementation.

## Regressions Checked

- Stage 7 historical project-state text remains present for older contract
  tests.
- Source scan did not introduce model/training/checkpoint/v5/legacy metric
  terms.
- No artifact export was added.
- No result files were written.

## Candidate Next Tasks

- `stage_8_1_actor_policy_interface_skeleton_without_model_or_training`
- `stage_8_1_policy_interface_tensorization_contract_without_model`
- `stage_8_1_actor_policy_adapter_smoke_test_without_training`

## Recommended Next Task

`stage_8_1_actor_policy_interface_skeleton_without_model_or_training`.

Reason: Stage 8.0 has frozen the schema. The next useful step is a minimal
non-learning adapter around the schema so future policy families can share one
validated interface, still without model code or training.

## Owner Decision Required

Yes. Codex must not proceed to Stage 8.1 without owner approval.
