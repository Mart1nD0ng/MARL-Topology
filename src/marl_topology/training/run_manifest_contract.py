"""Stage 5.9 design-only run manifest and artifact contract.

This module is a structured manifest. It does not write artifacts, create
checkpoints, export datasets, execute runs, or change result directories.
"""

from __future__ import annotations


STAGE5_9_RUN_MANIFEST_ARTIFACT_STAGE_ID = (
    "stage_5_9_training_run_manifest_artifact_contract_without_execution"
)
STAGE5_9_VERDICT = "run_manifest_artifact_contract_frozen_execution_blocked"
STAGE5_9_RECOMMENDED_NEXT_TASK = (
    "stage_5_10_run_manifest_validator_implementation_without_training"
)


REQUIRED_RUN_MANIFEST_FIELDS = (
    "run_id",
    "stage_id",
    "created_at_utc",
    "owner_approval_id",
    "config_id",
    "scenario_set_id",
    "split_id",
    "seed",
    "seed_group_id",
    "code_version_marker",
    "contract_ids",
    "metric_registry_version",
    "physics_regime_id",
    "protocol_model_id",
    "objective_contract_id",
    "surrogate_config_id",
    "normalization_reference_id",
    "architecture_contract_id",
    "replay_schema_version",
    "artifact_policy_id",
)

PLANNED_ARTIFACT_GROUPS = (
    "manifests",
    "configs",
    "metric_reports",
    "diagnostics",
    "replay_exports",
    "checkpoints",
)


def build_stage5_9_run_manifest_artifact_contract() -> dict[str, object]:
    """Return the Stage 5.9 run manifest and artifact contract."""

    return {
        "stage": STAGE5_9_RUN_MANIFEST_ARTIFACT_STAGE_ID,
        "verdict": STAGE5_9_VERDICT,
        "artifact_write_allowed": False,
        "manifest_writer_allowed": False,
        "checkpoint_creation_allowed": False,
        "dataset_export_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "owner_decision_required": True,
        "recommended_next_task": STAGE5_9_RECOMMENDED_NEXT_TASK,
        "controlled_object": "future run manifest and artifact boundary",
        "artifact_root_contract": _artifact_root_contract(),
        "run_manifest_schema": _run_manifest_schema(),
        "artifact_group_policy": _artifact_group_policy(),
        "retention_policy": _retention_policy(),
        "reproducibility_checks": _reproducibility_checks(),
        "blocked_until_future_approval": _blocked_until_future_approval(),
        "checks": _checks(),
    }


def _artifact_root_contract() -> dict[str, object]:
    return {
        "artifact_root": "result_save",
        "current_contract_writes_files": False,
        "expected_baseline_entries": [".gitkeep"],
        "future_write_requires_owner_approval": True,
        "path_escape_allowed": False,
        "legacy_reference_write_allowed": False,
    }


def _run_manifest_schema() -> dict[str, object]:
    return {
        "status": "planned_not_active",
        "required_fields": list(REQUIRED_RUN_MANIFEST_FIELDS),
        "manifest_format": "future_json_or_yaml_with_schema_validation",
        "schema_version_required": True,
        "contract_ids_required": True,
        "metric_registry_version_required": True,
        "owner_approval_id_required": True,
    }


def _artifact_group_policy() -> tuple[dict[str, object], ...]:
    return (
        _artifact_group("manifests", "future_run_metadata", "required_before_execution"),
        _artifact_group("configs", "future_config_snapshots", "required_before_execution"),
        _artifact_group("metric_reports", "registered_evaluation_reports", "required_after_run"),
        _artifact_group("diagnostics", "training_only_diagnostics", "allowed_future"),
        _artifact_group("replay_exports", "future_dataset_exports", "blocked"),
        _artifact_group("checkpoints", "future_model_state", "blocked"),
    )


def _artifact_group(
    name: str,
    purpose: str,
    status: str,
) -> dict[str, object]:
    return {
        "name": name,
        "purpose": purpose,
        "status": status,
        "writes_allowed_now": False,
        "requires_manifest_reference": True,
    }


def _retention_policy() -> dict[str, object]:
    return {
        "status": "planned_not_active",
        "manifest_retention": "keep",
        "registered_metric_report_retention": "keep",
        "diagnostic_retention": "future_budget_required",
        "replay_export_retention": "future_budget_required",
        "checkpoint_retention": "future_budget_required",
        "delete_policy_requires_owner_approval": True,
    }


def _reproducibility_checks() -> tuple[str, ...]:
    return (
        "run_manifest_contains_all_required_fields",
        "artifact_paths_stay_under_result_save",
        "owner_approval_id_present",
        "seed_and_seed_group_recorded",
        "config_id_recorded",
        "code_version_marker_recorded",
        "contract_ids_recorded",
        "metric_registry_version_recorded",
        "physics_and_protocol_ids_recorded",
        "actor_boundary_contract_recorded",
        "replay_schema_version_recorded",
        "legacy_reference_path_not_used_as_artifact_root",
    )


def _blocked_until_future_approval() -> tuple[str, ...]:
    return (
        "artifact_writer_implementation",
        "manifest_writer_implementation",
        "dataset_export_implementation",
        "checkpoint_creation",
        "training_execution",
        "model_implementation",
        "weight_calibration",
        "final_tau_selection",
        "legacy_code_migration",
    )


def _checks() -> dict[str, object]:
    return {
        "artifact_write_allowed": False,
        "manifest_writer_allowed": False,
        "checkpoint_creation_allowed": False,
        "dataset_export_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "artifact_root_declared": True,
        "required_manifest_fields_declared": True,
        "retention_policy_declared": True,
        "reproducibility_checks_declared": True,
        "path_escape_allowed": False,
    }
