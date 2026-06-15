"""Minimal Stage 6 training-stack guard.

This module prepares a guarded learning-stack surface only. It validates a
run manifest and blocks execution-side operations. It does not write artifacts,
export datasets, create checkpoints, instantiate models, or execute any run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .run_manifest_validator import (
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)


STAGE6_0_MINIMAL_STACK_STAGE_ID = (
    "stage_6_0_minimal_training_stack_implementation_with_manifest_guard"
)
STAGE6_0_VERDICT = "minimal_training_stack_guard_ready_execution_blocked"
STAGE6_0_RECOMMENDED_NEXT_TASK = (
    "stage_6_1_actor_safe_batch_builder_without_model_or_training"
)

ALLOWED_DRY_RUN_OPERATIONS = (
    "validate_manifest",
    "inspect_contract_refs",
    "report_boundaries",
)
BLOCKED_OPERATION_CODES = {
    "write_artifact": "artifact_write_blocked",
    "write_manifest": "manifest_write_blocked",
    "export_dataset": "dataset_export_blocked",
    "create_checkpoint": "checkpoint_creation_blocked",
    "execute_training": "training_execution_blocked",
    "instantiate_model": "model_implementation_blocked",
    "calibrate_weights": "weight_calibration_blocked",
    "select_final_tau": "final_tau_selection_blocked",
}
REQUIRED_CONTRACT_MARKERS = (
    "metric_governance",
    "objective_contract",
    "training_contract",
    "run_manifest_artifact_contract",
    "run_manifest_validator",
)


@dataclass(frozen=True)
class MinimalTrainingStackIssue:
    """One Stage 6.0 stack-guard issue."""

    code: str
    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }


@dataclass(frozen=True)
class MinimalTrainingStackReport:
    """Stage 6.0 dry-run training-stack report."""

    stage: str
    verdict: str
    stack_ready: bool
    manifest_valid: bool
    owner_approval_checked: bool
    requested_operations: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    blocked_operations: tuple[str, ...]
    issues: tuple[MinimalTrainingStackIssue, ...]
    manifest_issue_codes: tuple[str, ...]
    result_save_entries: tuple[str, ...]
    writes_performed: bool = False
    artifact_write_allowed: bool = False
    manifest_write_allowed: bool = False
    dataset_export_allowed: bool = False
    checkpoint_creation_allowed: bool = False
    training_execution_allowed: bool = False
    model_implementation_allowed: bool = False
    weight_calibration_allowed: bool = False
    final_tau_selection_allowed: bool = False
    v5_code_migrated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "verdict": self.verdict,
            "stack_ready": self.stack_ready,
            "manifest_valid": self.manifest_valid,
            "owner_approval_checked": self.owner_approval_checked,
            "requested_operations": list(self.requested_operations),
            "allowed_operations": list(self.allowed_operations),
            "blocked_operations": list(self.blocked_operations),
            "issues": [issue.to_dict() for issue in self.issues],
            "manifest_issue_codes": list(self.manifest_issue_codes),
            "result_save_entries": list(self.result_save_entries),
            "writes_performed": self.writes_performed,
            "artifact_write_allowed": self.artifact_write_allowed,
            "manifest_write_allowed": self.manifest_write_allowed,
            "dataset_export_allowed": self.dataset_export_allowed,
            "checkpoint_creation_allowed": self.checkpoint_creation_allowed,
            "training_execution_allowed": self.training_execution_allowed,
            "model_implementation_allowed": self.model_implementation_allowed,
            "weight_calibration_allowed": self.weight_calibration_allowed,
            "final_tau_selection_allowed": self.final_tau_selection_allowed,
            "v5_code_migrated": self.v5_code_migrated,
            "recommended_next_task": STAGE6_0_RECOMMENDED_NEXT_TASK,
        }


def prepare_minimal_training_stack(
    manifest: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
    owner_approved_stage6: bool = False,
    requested_operations: Sequence[str] = ALLOWED_DRY_RUN_OPERATIONS,
) -> MinimalTrainingStackReport:
    """Validate the manifest and return a guarded Stage 6 stack report."""

    root = _project_root(project_root)
    manifest_result = validate_run_manifest_dry_run(manifest, project_root=root)
    issues: list[MinimalTrainingStackIssue] = []

    if not owner_approved_stage6:
        issues.append(
            MinimalTrainingStackIssue(
                code="stage6_owner_approval_missing",
                field="owner_approved_stage6",
                message="Stage 6 stack preparation requires owner approval",
            )
        )

    _check_required_contract_markers(manifest, issues)
    blocked_operations = _find_blocked_operations(requested_operations, issues)

    stack_ready = manifest_result.is_valid and not issues
    verdict = STAGE6_0_VERDICT if stack_ready else "minimal_training_stack_guard_blocked"
    return MinimalTrainingStackReport(
        stage=STAGE6_0_MINIMAL_STACK_STAGE_ID,
        verdict=verdict,
        stack_ready=stack_ready,
        manifest_valid=manifest_result.is_valid,
        owner_approval_checked=owner_approved_stage6,
        requested_operations=tuple(requested_operations),
        allowed_operations=tuple(
            operation
            for operation in requested_operations
            if operation in ALLOWED_DRY_RUN_OPERATIONS
        ),
        blocked_operations=blocked_operations,
        issues=tuple(issues),
        manifest_issue_codes=manifest_result.error_codes(),
        result_save_entries=_result_save_entries(root),
    )


def build_stage6_0_minimal_stack_report(
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Build the default Stage 6.0 dry-run report."""

    manifest = build_valid_stage5_10_dry_run_manifest(
        overrides={
            "stage_id": STAGE6_0_MINIMAL_STACK_STAGE_ID,
            "contract_ids": list(REQUIRED_CONTRACT_MARKERS),
        }
    )
    report = prepare_minimal_training_stack(
        manifest,
        project_root=project_root,
        owner_approved_stage6=True,
    )
    return report.to_dict()


def _check_required_contract_markers(
    manifest: Mapping[str, object],
    issues: list[MinimalTrainingStackIssue],
) -> None:
    contract_ids = manifest.get("contract_ids")
    if not isinstance(contract_ids, Sequence) or isinstance(contract_ids, str):
        issues.append(
            MinimalTrainingStackIssue(
                code="contract_ids_not_sequence",
                field="contract_ids",
                message="contract_ids must be a sequence of contract markers",
            )
        )
        return
    present = {str(value) for value in contract_ids}
    for marker in REQUIRED_CONTRACT_MARKERS:
        if marker not in present:
            issues.append(
                MinimalTrainingStackIssue(
                    code="missing_contract_marker",
                    field="contract_ids",
                    message=f"missing required contract marker: {marker}",
                )
            )


def _find_blocked_operations(
    requested_operations: Sequence[str],
    issues: list[MinimalTrainingStackIssue],
) -> tuple[str, ...]:
    blocked: list[str] = []
    for operation in requested_operations:
        if operation in BLOCKED_OPERATION_CODES:
            code = BLOCKED_OPERATION_CODES[operation]
            blocked.append(operation)
            issues.append(
                MinimalTrainingStackIssue(
                    code=code,
                    field="requested_operations",
                    message=f"operation is blocked in Stage 6.0: {operation}",
                )
            )
    return tuple(blocked)


def _result_save_entries(project_root: Path) -> tuple[str, ...]:
    result_root = project_root / "result_save"
    if not result_root.exists():
        return ()
    return tuple(sorted(path.name for path in result_root.iterdir()))


def _project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve(strict=False)
    return Path(__file__).resolve(strict=False).parents[3]
