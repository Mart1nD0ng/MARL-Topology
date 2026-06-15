# Stage 5.10 Run Manifest Validator Exit Gate

## Controlled Object

The controlled object is the Stage 5 boundary between future training-run
manifests, artifact-root declarations, path containment, owner approval, and
execution permission.

Stage 5.10 implements only a dry-run validator. It does not write artifacts,
write manifests, export datasets, create checkpoints, run training, implement
actor/critic/model code, calibrate reward weights, select final `tau_consensus`,
or migrate v5 code.

## Desired State

Stage 5 has a concrete exit gate:

```text
verdict = stage5_closed_manifest_validator_exit_gate_passable
artifact_write_allowed = false
manifest_write_allowed = false
checkpoint_creation_allowed = false
dataset_export_allowed = false
training_execution_allowed = false
```

If the dry-run validator accepts the sample manifest, Stage 5 is closed. Stage
6 may begin only with owner approval.

## Validator Checks

The validator checks:

- required manifest fields present;
- `owner_approval_id` present;
- artifact root under result_save;
- no path escape;
- legacy reference path is rejected as artifact root;
- `seed` and `seed_group_id` recorded;
- `config_id` recorded;
- `code_version_marker` recorded;
- `contract_ids` recorded;
- `metric_registry_version` recorded;
- `physics_regime_id` recorded;
- `protocol_model_id` recorded;
- `objective_contract_id` recorded;
- `surrogate_config_id` recorded;
- `normalization_reference_id` recorded;
- `architecture_contract_id` recorded;
- `replay_schema_version` recorded;
- `artifact_policy_id` recorded.

Optional artifact paths may be validated if a manifest includes them, but the
validator still performs no writes.

## Stage Closure Discipline

Stage 5.10 is an exit gate, not another planning loop. Do not open additional
Stage 5.x planning tasks after this gate passes. Any remaining work should be
framed as a Stage 6 owner-approved task or as a clearly scoped defect in the
Stage 5 exit gate.

## Implementation

Structured validator:

- `src/marl_topology/training/run_manifest_validator.py`

Dry-run replay:

- `scripts/replay/stage5_10_run_manifest_validator_exit_gate.py`

The validator returns a result object with issues and normalized artifact root.
It does not create directories or files. `result_save` remains at the scaffold
baseline with `.gitkeep` only.

## Acceptance Criteria

- A valid dry-run manifest is accepted.
- Missing required fields are rejected.
- Artifact root path escape is rejected.
- Legacy reference artifact root is rejected.
- Optional artifact paths, when supplied, must stay under `result_save`.
- No artifact files are written under `result_save` except `.gitkeep`.
- No training code or model code is added.
- Stage 5 is closed.
- Stage 6 may begin only with owner approval.

## Recommended Stage 6 Task

`stage_6_0_minimal_training_stack_implementation_with_manifest_guard`

Reason: Stage 5 has frozen objective, surrogate, training design, architecture,
replay, manifest, artifact, and validator boundaries. The next useful step is
to begin Stage 6 with owner approval by implementing the smallest training-stack
surface that consumes the manifest guard and preserves Dec-POMDP and no-write
boundaries until execution is separately authorized.

## Residual Risks

- The validator is dry-run only and does not allocate real run ids.
- The workspace may not be a git repository, so a future execution task still
  needs a robust code-version marker source.
- Stage 6 must decide what minimal implementation is allowed before any
  optimization loop, checkpoint, dataset export, or training run exists.
