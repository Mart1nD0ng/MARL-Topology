from pathlib import Path

from marl_topology.training import (
    REQUIRED_CONTRACT_MARKERS,
    STAGE6_0_MINIMAL_STACK_STAGE_ID,
    STAGE6_0_RECOMMENDED_NEXT_TASK,
    STAGE6_0_VERDICT,
    build_stage6_0_minimal_stack_report,
    build_valid_stage5_10_dry_run_manifest,
    prepare_minimal_training_stack,
)


ROOT = Path(__file__).resolve().parents[2]


def _stage6_manifest(**overrides):
    base = build_valid_stage5_10_dry_run_manifest(
        overrides={
            "stage_id": STAGE6_0_MINIMAL_STACK_STAGE_ID,
            "contract_ids": list(REQUIRED_CONTRACT_MARKERS),
        }
    )
    base.update(overrides)
    return base


def test_stage6_0_accepts_valid_owner_approved_manifest() -> None:
    report = prepare_minimal_training_stack(
        _stage6_manifest(),
        project_root=ROOT,
        owner_approved_stage6=True,
    )

    assert report.stack_ready is True
    assert report.verdict == STAGE6_0_VERDICT
    assert report.manifest_valid is True
    assert report.owner_approval_checked is True
    assert report.issues == ()
    assert report.manifest_issue_codes == ()
    assert report.writes_performed is False
    assert report.training_execution_allowed is False
    assert report.model_implementation_allowed is False
    assert set(report.result_save_entries) <= {
        ".gitkeep",
        "evidence_dataset_only",
        "stage25_small_scale_mappo_training_pilot",
        "stage26_full_system_health_diagnostic",
        "stage27_critic_baseline_repair",
        "stage28_repaired_critic_mappo_rerun",
        "stage32_production_training",
        "stage33_gnn_stability_repair",
        "stage34_gnn_ablation_diagnostics",
    }


def test_stage6_0_missing_owner_approval_blocks_readiness() -> None:
    report = prepare_minimal_training_stack(
        _stage6_manifest(),
        project_root=ROOT,
        owner_approved_stage6=False,
    )

    assert report.stack_ready is False
    assert _issue_codes(report) == {"stage6_owner_approval_missing"}


def test_stage6_0_missing_contract_marker_blocks_readiness() -> None:
    manifest = _stage6_manifest(contract_ids=["metric_governance"])

    report = prepare_minimal_training_stack(
        manifest,
        project_root=ROOT,
        owner_approved_stage6=True,
    )

    assert report.stack_ready is False
    assert "missing_contract_marker" in _issue_codes(report)


def test_stage6_0_invalid_manifest_is_blocked_by_stage5_10_validator() -> None:
    manifest = _stage6_manifest(owner_approval_id="", artifact_root="..\\outside")

    report = prepare_minimal_training_stack(
        manifest,
        project_root=ROOT,
        owner_approved_stage6=True,
    )

    assert report.stack_ready is False
    assert "missing_required_field" in report.manifest_issue_codes
    assert "artifact_path_escape" in report.manifest_issue_codes


def test_stage6_0_rejects_blocked_operation_requests() -> None:
    report = prepare_minimal_training_stack(
        _stage6_manifest(),
        project_root=ROOT,
        owner_approved_stage6=True,
        requested_operations=(
            "validate_manifest",
            "write_artifact",
            "export_dataset",
            "create_checkpoint",
            "execute_training",
            "instantiate_model",
            "calibrate_weights",
            "select_final_tau",
        ),
    )

    assert report.stack_ready is False
    assert set(report.blocked_operations) == {
        "write_artifact",
        "export_dataset",
        "create_checkpoint",
        "execute_training",
        "instantiate_model",
        "calibrate_weights",
        "select_final_tau",
    }
    assert "artifact_write_blocked" in _issue_codes(report)
    assert "training_execution_blocked" in _issue_codes(report)
    assert "model_implementation_blocked" in _issue_codes(report)


def test_stage6_0_default_report_is_ready_but_execution_blocked() -> None:
    report = build_stage6_0_minimal_stack_report(project_root=ROOT)

    assert report["stage"] == STAGE6_0_MINIMAL_STACK_STAGE_ID
    assert report["verdict"] == STAGE6_0_VERDICT
    assert report["stack_ready"] is True
    assert report["recommended_next_task"] == STAGE6_0_RECOMMENDED_NEXT_TASK
    assert report["writes_performed"] is False
    assert report["artifact_write_allowed"] is False
    assert report["dataset_export_allowed"] is False
    assert report["checkpoint_creation_allowed"] is False
    assert report["training_execution_allowed"] is False
    assert report["model_implementation_allowed"] is False
    assert report["weight_calibration_allowed"] is False
    assert report["final_tau_selection_allowed"] is False
    assert report["v5_code_migrated"] is False


def _issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}
