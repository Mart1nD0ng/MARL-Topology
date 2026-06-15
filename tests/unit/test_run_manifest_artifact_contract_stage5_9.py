from marl_topology.training import (
    PLANNED_ARTIFACT_GROUPS,
    REQUIRED_RUN_MANIFEST_FIELDS,
    STAGE5_9_RECOMMENDED_NEXT_TASK,
    STAGE5_9_RUN_MANIFEST_ARTIFACT_STAGE_ID,
    STAGE5_9_VERDICT,
    build_stage5_9_run_manifest_artifact_contract,
)


def test_stage5_9_contract_blocks_artifact_writes_and_training() -> None:
    contract = build_stage5_9_run_manifest_artifact_contract()
    checks = contract["checks"]

    assert contract["stage"] == STAGE5_9_RUN_MANIFEST_ARTIFACT_STAGE_ID
    assert contract["verdict"] == STAGE5_9_VERDICT
    assert contract["artifact_write_allowed"] is False
    assert contract["manifest_writer_allowed"] is False
    assert contract["checkpoint_creation_allowed"] is False
    assert contract["dataset_export_allowed"] is False
    assert contract["training_execution_allowed"] is False
    assert contract["model_code_added"] is False
    assert contract["weight_calibration_performed"] is False
    assert contract["final_tau_selected"] is False
    assert contract["v5_code_migrated"] is False
    assert checks["artifact_write_allowed"] is False
    assert checks["path_escape_allowed"] is False


def test_stage5_9_required_manifest_fields_are_explicit() -> None:
    contract = build_stage5_9_run_manifest_artifact_contract()
    schema = contract["run_manifest_schema"]

    assert schema["status"] == "planned_not_active"
    assert tuple(schema["required_fields"]) == REQUIRED_RUN_MANIFEST_FIELDS
    for field in (
        "run_id",
        "owner_approval_id",
        "seed",
        "contract_ids",
        "metric_registry_version",
        "replay_schema_version",
        "artifact_policy_id",
    ):
        assert field in schema["required_fields"]
    assert schema["owner_approval_id_required"] is True


def test_stage5_9_artifact_root_groups_and_retention_are_design_only() -> None:
    contract = build_stage5_9_run_manifest_artifact_contract()
    root = contract["artifact_root_contract"]
    groups = {group["name"]: group for group in contract["artifact_group_policy"]}
    retention = contract["retention_policy"]

    assert root["artifact_root"] == "result_save"
    assert root["expected_baseline_entries"] == [".gitkeep"]
    assert root["path_escape_allowed"] is False
    assert root["legacy_reference_write_allowed"] is False
    assert set(groups) == set(PLANNED_ARTIFACT_GROUPS)
    assert groups["checkpoints"]["status"] == "blocked"
    assert groups["replay_exports"]["status"] == "blocked"
    assert all(group["writes_allowed_now"] is False for group in groups.values())
    assert retention["manifest_retention"] == "keep"
    assert retention["delete_policy_requires_owner_approval"] is True


def test_stage5_9_reproducibility_checks_and_next_task_are_explicit() -> None:
    contract = build_stage5_9_run_manifest_artifact_contract()
    checks = set(contract["reproducibility_checks"])

    assert "run_manifest_contains_all_required_fields" in checks
    assert "artifact_paths_stay_under_result_save" in checks
    assert "legacy_reference_path_not_used_as_artifact_root" in checks
    assert contract["recommended_next_task"] == STAGE5_9_RECOMMENDED_NEXT_TASK
    assert STAGE5_9_RECOMMENDED_NEXT_TASK == (
        "stage_5_10_run_manifest_validator_implementation_without_training"
    )
    assert "artifact_writer_implementation" in contract["blocked_until_future_approval"]
