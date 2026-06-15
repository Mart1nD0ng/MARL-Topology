# Stage 5.9 Training Run Manifest And Artifact Contract Without Execution

## Controlled Object

The controlled object is the future boundary between approved run manifests,
artifact paths, retention policy, reproducibility evidence, and training
execution.

Stage 5.9 freezes the run-manifest and artifact contract. It does not write
artifacts, create checkpoints, export datasets, run training, add model code,
calibrate reward weights, select final `tau_consensus`, or migrate v5 code.

## Desired State

The project has explicit future artifact and manifest semantics before any
writer, checkpoint, dataset export, or run exists.

Design verdict:

```text
run_manifest_artifact_contract_frozen_execution_blocked
artifact_write_allowed = false
manifest_writer_allowed = false
checkpoint_creation_allowed = false
dataset_export_allowed = false
training_execution_allowed = false
```

## Artifact Root Contract

Future artifacts must live under:

```text
result_save
```

Stage 5.9 writes no files under `result_save`. The expected current baseline is
only:

```text
.gitkeep
```

Path escape is forbidden. The legacy reference path must never be used as an
artifact root.

## Required Future Run Manifest Fields

Future run manifests must include:

- `run_id`
- `stage_id`
- `created_at_utc`
- `owner_approval_id`
- `config_id`
- `scenario_set_id`
- `split_id`
- `seed`
- `seed_group_id`
- `code_version_marker`
- `contract_ids`
- `metric_registry_version`
- `physics_regime_id`
- `protocol_model_id`
- `objective_contract_id`
- `surrogate_config_id`
- `normalization_reference_id`
- `architecture_contract_id`
- `replay_schema_version`
- `artifact_policy_id`

The manifest schema is planned, not active. Stage 5.9 does not implement a
manifest writer or validator.

## Artifact Groups

Future artifact groups:

- `manifests`
- `configs`
- `metric_reports`
- `diagnostics`
- `replay_exports`
- `checkpoints`

All writes remain blocked. Replay exports and checkpoints are explicitly
blocked until future owner approval.

## Retention Policy

Planned retention policy:

- manifests: keep;
- registered metric reports: keep;
- diagnostics: future budget required;
- replay exports: future budget required;
- checkpoints: future budget required.

Delete policy changes require owner approval.

## Reproducibility Checks

Future runs must prove:

- run manifest contains all required fields;
- artifact paths stay under `result_save`;
- owner approval id is present;
- seed and seed group are recorded;
- config id is recorded;
- code version marker is recorded;
- contract ids are recorded;
- metric registry version is recorded;
- physics and protocol ids are recorded;
- actor boundary contract is recorded;
- replay schema version is recorded;
- legacy reference path is not used as artifact root.

## Acceptance Criteria

- Structured Stage 5.9 contract exists and is runnable.
- Artifact root is declared as `result_save`.
- Required run-manifest fields are explicit.
- Artifact groups and retention policy are explicit.
- Reproducibility checks are explicit.
- `result_save` remains at scaffold baseline with `.gitkeep` only.
- No artifact writer, manifest writer, checkpoint writer, dataset export,
  training execution, model code, reward-weight calibration, final tau
  selection, or v5 migration occurs.

## Residual Risks

- Stage 5.9 does not implement a manifest validator.
- Future artifact path validation still needs code before any write is allowed.
- Future execution still needs owner approval and a dry-run validator before
  writing run artifacts.
