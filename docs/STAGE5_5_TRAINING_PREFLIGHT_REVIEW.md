# Stage 5.5 Training Preflight Review Without Training

## Controlled Object

The controlled object is the transition from Stage 5 surrogate diagnostics to
future learning-system design.

This task reviews whether the project is ready for training execution. It does
not execute training, add actor or critic code, calibrate reward weights, select
final `tau_consensus`, or migrate v5 code.

## Desired State

The project can state, with evidence, which gates are ready, blocked, or
deferred before any learning run is designed or executed.

Desired review conclusion:

```text
verdict = not_ready_for_training_execution
training_execution_allowed = false
training_design_contract_allowed = true
recommended_next_task = stage_5_6_training_design_contract_without_execution
owner_decision_required = true
```

## Baseline

Stage 5.4 provides a component-only diagnostic report:

```text
component_policy = component_only_no_scalar_reward_v1
```

It reports reliability violation, reliability penalty, normalized latency,
normalized energy, latency penalty, and energy penalty as training-only
diagnostics. It does not report a scalar surrogate.

Stage 5.4 does not provide:

- reward weight policy;
- return, advantage, or value-target contract;
- actor or critic architecture contract;
- multi-seed learning protocol;
- training artifact policy;
- owner approval for training execution.

## Sensors

- `src/marl_topology/evaluation/training_preflight.py`
- `scripts/replay/stage5_5_training_preflight_review.py`
- unit tests for preflight report structure and verdict
- contract tests for documentation, project state, harness task, and negative
  source checks
- existing Stage 5.4 component-only report
- replay dataset column contract
- Dec-POMDP contract

## Actuators

- Add a Stage 5.5 report builder.
- Add a replay script that prints the review report.
- Add a harness task for the review gate.
- Update reward, surrogate, metric, Dec-POMDP, replay, and project-state
  documents.
- Add tests that fail if the review becomes a training entry point.

## Gate Results

Passed gates:

- metric governance is ready for design;
- objective and training-side surrogate boundary is ready;
- normalization references are available;
- Dec-POMDP boundary is ready for architecture design;
- baseline and oracle evidence is available;
- component-only diagnostic report is available.

Blocked gates:

- surrogate scalarization and weight policy is missing;
- return, advantage, and value-target contract is missing;
- actor and critic architecture contract is missing;
- training artifact and run-manifest policy is missing;
- multi-seed stochastic evidence protocol is missing.

Deferred gates:

- deployment scenario distribution calibration;
- multi-agent credit-assignment calibration;
- final `tau_consensus` selection.

## Required Before Training Execution

Training execution remains blocked until a future owner-approved task provides:

- `stage_5_6_training_design_contract_without_execution`;
- surrogate scalarization and weight policy;
- learning-target and replay-column contract;
- actor and critic architecture contract;
- artifact, seed, and run-manifest policy;
- multi-seed evaluation protocol;
- explicit owner approval for training execution.

## Negative Checks

Stage 5.5 must not:

- run training;
- create checkpoints;
- add actor, critic, COMA, GNN, LSTM, or optimizer code;
- calibrate reward weights;
- select final `tau_consensus`;
- make surrogate diagnostics deployment actor inputs;
- report surrogate components as evaluation metrics;
- migrate v5 code or copy v5 reward logic.

## Acceptance Criteria

- The preflight report exists and is runnable.
- The report verdict is `not_ready_for_training_execution`.
- Training execution is explicitly disallowed.
- A training design contract may be recommended as the next owner-approved
  task.
- All blocked gates are visible in the report.
- Tests and harness validation pass.

## Residual Risks

- The preflight review checks readiness at the contract level; it does not prove
  future learning stability.
- Future Stage 5.6 still needs to decide architecture options, stochastic
  evidence protocol, artifact policy, and scalarization policy.
- Scenario distribution remains alpha and deterministic, not deployment
  calibrated.
