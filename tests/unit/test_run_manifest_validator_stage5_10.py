from pathlib import Path

from marl_topology.training import (
    STAGE5_10_RECOMMENDED_STAGE6_TASK,
    STAGE5_10_RUN_MANIFEST_VALIDATOR_STAGE_ID,
    build_stage5_10_exit_gate_report,
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage5_10_validator_accepts_valid_dry_run_manifest() -> None:
    manifest = build_valid_stage5_10_dry_run_manifest()

    result = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert result.is_valid is True
    assert result.issues == ()
    assert result.writes_performed is False
    assert result.training_execution_allowed is False
    assert result.normalized_artifact_root is not None
    assert Path(result.normalized_artifact_root) == (ROOT / "result_save").resolve()


def test_stage5_10_validator_rejects_missing_required_fields() -> None:
    manifest = build_valid_stage5_10_dry_run_manifest()
    del manifest["owner_approval_id"]
    manifest["seed_group_id"] = ""
    manifest["contract_ids"] = []

    result = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert result.is_valid is False
    assert ("missing_required_field", "owner_approval_id") in _issue_pairs(result)
    assert ("missing_required_field", "seed_group_id") in _issue_pairs(result)
    assert ("missing_required_field", "contract_ids") in _issue_pairs(result)


def test_stage5_10_validator_rejects_artifact_root_escape() -> None:
    manifest = build_valid_stage5_10_dry_run_manifest(
        artifact_root="result_save\\..\\outside"
    )

    result = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert result.is_valid is False
    assert "artifact_path_escape" in result.error_codes()


def test_stage5_10_validator_rejects_legacy_reference_as_artifact_root() -> None:
    legacy_root = str(ROOT.parent / "v5")
    manifest = build_valid_stage5_10_dry_run_manifest(artifact_root=legacy_root)

    result = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert result.is_valid is False
    assert "legacy_reference_artifact_root" in result.error_codes()
    assert "artifact_path_escape" in result.error_codes()


def test_stage5_10_validator_checks_optional_artifact_paths_without_writing() -> None:
    manifest = build_valid_stage5_10_dry_run_manifest(
        overrides={
            "artifact_paths": {
                "manifest": "result_save\\manifests\\dry_run.json",
                "bad": "..\\outside\\bad.json",
            }
        }
    )

    result = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert result.is_valid is False
    assert "artifact_paths" in result.checked_fields
    assert "artifact_path_escape" in result.error_codes()
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    _unexpected = {c for c in _committed if c != ".gitkeep" and not c.endswith(".md")}
    assert not _unexpected, f"result_save COMMITTED non-report run artifacts: {sorted(_unexpected)}"


def test_stage5_10_exit_gate_report_closes_stage5_without_execution() -> None:
    report = build_stage5_10_exit_gate_report(project_root=ROOT)

    assert report["stage"] == STAGE5_10_RUN_MANIFEST_VALIDATOR_STAGE_ID
    assert report["stage5_closed"] is True
    assert report["stage6_allowed_with_owner_approval"] is True
    assert report["recommended_stage6_task"] == STAGE5_10_RECOMMENDED_STAGE6_TASK
    assert report["artifact_write_allowed"] is False
    assert report["manifest_write_allowed"] is False
    assert report["checkpoint_creation_allowed"] is False
    assert report["dataset_export_allowed"] is False
    assert report["training_execution_allowed"] is False
    assert report["model_code_added"] is False
    assert report["weight_calibration_performed"] is False
    assert report["final_tau_selected"] is False
    assert report["v5_code_migrated"] is False
    assert report["validator_result"]["writes_performed"] is False


def _issue_pairs(result) -> set[tuple[str, str]]:
    return {(issue.code, issue.field) for issue in result.issues}
