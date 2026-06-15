# Stage 6.0 Minimal Training Stack With Manifest Guard

## Controlled Object

The controlled object is the first executable Stage 6 learning-stack boundary:

```text
manifest guard -> contract reference check -> dry-run stack readiness report
```

This stage implements only a guarded preparation surface. It does not run
training, write artifacts, write manifests, export datasets, create
checkpoints, implement actor/critic/model code, calibrate reward weights,
select final `tau_consensus`, or migrate v5 code.

## Desired State

The project can construct a minimal training-stack readiness report only after
the Stage 5.10 run manifest validator accepts the manifest.

Verdict:

```text
minimal_training_stack_guard_ready_execution_blocked
```

Required boundary flags:

```text
writes_performed = false
artifact_write_allowed = false
manifest_write_allowed = false
dataset_export_allowed = false
checkpoint_creation_allowed = false
training_execution_allowed = false
model_implementation_allowed = false
weight_calibration_allowed = false
final_tau_selection_allowed = false
```

## Implementation Boundary

Structured module:

- `src/marl_topology/training/minimal_stack.py`

Replay report:

- `scripts/replay/stage6_0_minimal_training_stack_guard.py`

The stack guard performs:

- Stage 5.10 manifest validation;
- owner approval marker check for Stage 6 preparation;
- required contract marker check;
- blocked-operation rejection;
- read-only `result_save` baseline inspection.

Allowed dry-run operations:

- `validate_manifest`
- `inspect_contract_refs`
- `report_boundaries`

Blocked operations:

- artifact writing;
- manifest writing;
- dataset export;
- checkpoint creation;
- training execution;
- model instantiation;
- reward-weight calibration;
- final tau selection.

## Dec-POMDP Boundary

Stage 6.0 does not instantiate a deployment actor, centralized critic, graph
encoder, recurrent encoder, checkpoint loader, or replay writer.

The next implementation surface must still preserve:

- deployment actor inputs are local-only;
- centralized information is training-only;
- actor-safe batch projection is required before any model work;
- no oracle labels, objective metrics, future outcomes, or centralized views may
  enter deployment actor inputs.

## Acceptance Criteria

- A valid owner-approved dry-run manifest produces `stack_ready = true`.
- Missing owner approval blocks readiness.
- Missing required contract markers block readiness.
- Missing or invalid manifest fields block readiness through the Stage 5.10
  validator.
- Requests for blocked operations are rejected.
- `result_save` remains `.gitkeep` only.
- No artifact, dataset, checkpoint, training, model, reward-weight, final tau,
  or v5 migration side effect occurs.

## Recommended Next Task

`stage_6_1_actor_safe_batch_builder_without_model_or_training`

Reason: after the manifest guard is in place, the next narrow implementation is
an actor-safe batch builder that projects existing local observations without
models, checkpointing, dataset export, or training execution.

## Residual Risks

- Stage 6.0 does not build batches or store replay.
- Stage 6.0 does not implement a learner or model.
- Stage 6.1 must prove actor-safe projection before any model-facing code.
