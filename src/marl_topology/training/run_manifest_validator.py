"""Dry-run run manifest validator for the Stage 5 exit gate.

The validator checks manifest completeness and artifact-root containment. It
does not create directories, write files, export datasets, create checkpoints,
or execute any run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .run_manifest_contract import REQUIRED_RUN_MANIFEST_FIELDS


STAGE5_10_RUN_MANIFEST_VALIDATOR_STAGE_ID = (
    "stage_5_10_run_manifest_validator_exit_gate_without_execution"
)
STAGE5_10_VERDICT = "stage5_closed_manifest_validator_exit_gate_passable"
STAGE5_10_RECOMMENDED_STAGE6_TASK = (
    "stage_6_0_minimal_training_stack_implementation_with_manifest_guard"
)
ARTIFACT_ROOT_FIELD = "artifact_root"
OPTIONAL_ARTIFACT_PATHS_FIELD = "artifact_paths"
DEFAULT_ARTIFACT_ROOT = "result_save"


@dataclass(frozen=True)
class RunManifestValidationIssue:
    """One validation issue for a dry-run run manifest."""

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
class RunManifestValidationResult:
    """Dry-run validation result with no side effects."""

    is_valid: bool
    issues: tuple[RunManifestValidationIssue, ...]
    checked_fields: tuple[str, ...]
    normalized_artifact_root: str | None
    writes_performed: bool = False
    training_execution_allowed: bool = False

    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "checked_fields": list(self.checked_fields),
            "normalized_artifact_root": self.normalized_artifact_root,
            "writes_performed": self.writes_performed,
            "training_execution_allowed": self.training_execution_allowed,
        }


def validate_run_manifest_dry_run(
    manifest: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
) -> RunManifestValidationResult:
    """Validate a future run manifest without writing any artifact."""

    root = _project_root(project_root)
    issues: list[RunManifestValidationIssue] = []
    checked_fields = list(REQUIRED_RUN_MANIFEST_FIELDS) + [ARTIFACT_ROOT_FIELD]

    for field in REQUIRED_RUN_MANIFEST_FIELDS:
        if _is_blank(manifest.get(field)):
            issues.append(
                RunManifestValidationIssue(
                    code="missing_required_field",
                    field=field,
                    message=f"required manifest field is missing or blank: {field}",
                )
            )

    normalized_artifact_root = _validate_artifact_root(
        manifest=manifest,
        project_root=root,
        issues=issues,
    )
    _validate_optional_artifact_paths(
        manifest=manifest,
        project_root=root,
        issues=issues,
        checked_fields=checked_fields,
    )

    return RunManifestValidationResult(
        is_valid=not issues,
        issues=tuple(issues),
        checked_fields=tuple(checked_fields),
        normalized_artifact_root=normalized_artifact_root,
    )


def build_valid_stage5_10_dry_run_manifest(
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build an in-memory manifest used by tests and dry-run reports."""

    manifest: dict[str, object] = {
        "run_id": "dry_run_stage5_10_manifest_validator",
        "stage_id": STAGE5_10_RUN_MANIFEST_VALIDATOR_STAGE_ID,
        "created_at_utc": "2026-05-25T00:00:00Z",
        "owner_approval_id": "owner_approved_stage5_10_exit_gate",
        "config_id": "dry_run_config_stage5_10",
        "scenario_set_id": "dry_run_scenario_set",
        "split_id": "dry_run_split",
        "seed": 0,
        "seed_group_id": "dry_run_seed_group",
        "code_version_marker": "workspace_marker_required_before_execution",
        "contract_ids": [
            "metric_governance",
            "objective_contract",
            "training_contract",
            "run_manifest_artifact_contract",
        ],
        "metric_registry_version": "metric_governance_stage5",
        "physics_regime_id": "urlcc_finite_blocklength_v1",
        "protocol_model_id": "pbft_expected_initiator_mean_field_v1",
        "objective_contract_id": "objective_contract_stage5_0",
        "surrogate_config_id": "surrogate_interface_stage5_2",
        "normalization_reference_id": "normalization_reference_stage5_3",
        "architecture_contract_id": "policy_architecture_contract_stage5_7",
        "replay_schema_version": "learning_target_replay_contract_stage5_8",
        "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
        ARTIFACT_ROOT_FIELD: artifact_root,
    }
    if overrides:
        manifest.update(overrides)
    return manifest


def build_stage5_10_exit_gate_report(
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Return a dry-run exit-gate report for Stage 5."""

    manifest = build_valid_stage5_10_dry_run_manifest()
    result = validate_run_manifest_dry_run(manifest, project_root=project_root)
    return {
        "stage": STAGE5_10_RUN_MANIFEST_VALIDATOR_STAGE_ID,
        "verdict": STAGE5_10_VERDICT if result.is_valid else "stage5_exit_gate_failed",
        "stage5_closed": result.is_valid,
        "stage6_allowed_with_owner_approval": result.is_valid,
        "recommended_stage6_task": STAGE5_10_RECOMMENDED_STAGE6_TASK,
        "validator_result": result.to_dict(),
        "artifact_write_allowed": False,
        "manifest_write_allowed": False,
        "checkpoint_creation_allowed": False,
        "dataset_export_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
    }


def _validate_artifact_root(
    *,
    manifest: Mapping[str, object],
    project_root: Path,
    issues: list[RunManifestValidationIssue],
) -> str | None:
    raw_root = manifest.get(ARTIFACT_ROOT_FIELD)
    if _is_blank(raw_root):
        issues.append(
            RunManifestValidationIssue(
                code="missing_artifact_root",
                field=ARTIFACT_ROOT_FIELD,
                message="artifact root must be declared for dry-run validation",
            )
        )
        return None

    candidate = _normalize_path(raw_root, project_root)
    allowed_root = _normalize_path(DEFAULT_ARTIFACT_ROOT, project_root)
    if _looks_like_legacy_reference(candidate):
        issues.append(
            RunManifestValidationIssue(
                code="legacy_reference_artifact_root",
                field=ARTIFACT_ROOT_FIELD,
                message="legacy reference path cannot be used as artifact root",
            )
        )
    if not _is_relative_to(candidate, allowed_root):
        issues.append(
            RunManifestValidationIssue(
                code="artifact_path_escape",
                field=ARTIFACT_ROOT_FIELD,
                message="artifact root must stay under result_save",
            )
        )
    return str(candidate)


def _validate_optional_artifact_paths(
    *,
    manifest: Mapping[str, object],
    project_root: Path,
    issues: list[RunManifestValidationIssue],
    checked_fields: list[str],
) -> None:
    if OPTIONAL_ARTIFACT_PATHS_FIELD not in manifest:
        return
    checked_fields.append(OPTIONAL_ARTIFACT_PATHS_FIELD)
    paths = _coerce_artifact_paths(manifest[OPTIONAL_ARTIFACT_PATHS_FIELD])
    allowed_root = _normalize_path(DEFAULT_ARTIFACT_ROOT, project_root)
    for index, path_value in enumerate(paths):
        candidate = _normalize_path(path_value, project_root)
        field = f"{OPTIONAL_ARTIFACT_PATHS_FIELD}[{index}]"
        if _looks_like_legacy_reference(candidate):
            issues.append(
                RunManifestValidationIssue(
                    code="legacy_reference_artifact_path",
                    field=field,
                    message="legacy reference path cannot be used as artifact output",
                )
            )
        if not _is_relative_to(candidate, allowed_root):
            issues.append(
                RunManifestValidationIssue(
                    code="artifact_path_escape",
                    field=field,
                    message="artifact path must stay under result_save",
                )
            )


def _coerce_artifact_paths(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (Sequence, Mapping)):
        return len(value) == 0
    return False


def _project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve(strict=False)
    return Path(__file__).resolve(strict=False).parents[3]


def _normalize_path(value: object, project_root: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _is_relative_to(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _looks_like_legacy_reference(path: Path) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    return "phd_works" in parts and "v5" in parts
