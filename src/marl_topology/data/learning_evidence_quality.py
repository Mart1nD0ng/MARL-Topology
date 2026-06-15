"""Stage 7.1 learning evidence data quality sensors.

This module audits Stage 7 evidence datasets without training, model creation,
checkpoint writing, or dataset export.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from .learning_evidence import (
    ACTOR_VIEW_FORBIDDEN_FIELDS,
    REQUIRED_LEARNING_EVIDENCE_VIEWS,
    LearningEvidenceDataset,
    validate_actor_safe_view,
)


STAGE7_1_LEARNING_EVIDENCE_QUALITY_STAGE_ID = (
    "stage_7_1_learning_evidence_data_quality_report_with_owner_approval"
)
STAGE7_1_VERDICT = "stage7_closed_evidence_quality_passed_training_blocked"
STAGE7_1_RECOMMENDED_NEXT_TASK = (
    "stage_8_0_actor_policy_interface_contract_with_owner_approval"
)
REQUIRED_STAGE7_1_TOPOLOGY_FAMILIES = (
    "weak_or_disconnected_baseline",
    "sparse_candidate",
    "dense_full_graph_baseline",
)
REQUIRED_STAGE7_1_ACTION_TYPES = ("add_edge", "remove_edge", "keep_edge")


@dataclass(frozen=True, slots=True)
class LearningEvidenceQualityIssue:
    """One quality issue from a Stage 7 evidence audit."""

    severity: str
    code: str
    message: str
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.severity not in {"blocking", "warning", "info"}:
            raise ValueError(f"unknown issue severity: {self.severity}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LearningEvidenceQualityReport:
    """Structured Stage 7.1 quality report."""

    dataset_id: str
    row_count: int
    target_count: int
    view_coverage: Mapping[str, object]
    topology_family_coverage: Mapping[str, object]
    actor_leakage_summary: Mapping[str, object]
    metric_range_summary: Mapping[str, object]
    target_quality_summary: Mapping[str, object]
    feasibility_summary: Mapping[str, object]
    issues: tuple[LearningEvidenceQualityIssue, ...]
    gates_passed: tuple[str, ...]
    gates_deferred: tuple[str, ...]
    stage7_exit_ready: bool
    stage8_policy_interface_ready: bool
    training_execution_ready: bool = False
    checkpoint_creation_allowed: bool = False
    model_implementation_allowed: bool = False
    artifact_written: bool = False
    recommended_next_task: str = STAGE7_1_RECOMMENDED_NEXT_TASK

    @property
    def blocking_issue_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "blocking")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": STAGE7_1_LEARNING_EVIDENCE_QUALITY_STAGE_ID,
            "verdict": STAGE7_1_VERDICT if self.stage7_exit_ready else "evidence_quality_blocked",
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "target_count": self.target_count,
            "view_coverage": dict(self.view_coverage),
            "topology_family_coverage": dict(self.topology_family_coverage),
            "actor_leakage_summary": dict(self.actor_leakage_summary),
            "metric_range_summary": dict(self.metric_range_summary),
            "target_quality_summary": dict(self.target_quality_summary),
            "feasibility_summary": dict(self.feasibility_summary),
            "issues": [issue.to_dict() for issue in self.issues],
            "blocking_issue_count": self.blocking_issue_count,
            "warning_count": self.warning_count,
            "gates_passed": list(self.gates_passed),
            "gates_deferred": list(self.gates_deferred),
            "stage7_exit_ready": self.stage7_exit_ready,
            "stage8_policy_interface_ready": self.stage8_policy_interface_ready,
            "training_execution_ready": self.training_execution_ready,
            "checkpoint_creation_allowed": self.checkpoint_creation_allowed,
            "model_implementation_allowed": self.model_implementation_allowed,
            "artifact_written": self.artifact_written,
            "recommended_next_task": self.recommended_next_task,
        }


def detect_actor_view_leakage(
    actor_rows: Iterable[Mapping[str, object]],
) -> tuple[LearningEvidenceQualityIssue, ...]:
    """Return actor-view leakage issues without mutating rows."""

    issues: list[LearningEvidenceQualityIssue] = []
    for index, row in enumerate(actor_rows):
        forbidden = sorted(set(row) & ACTOR_VIEW_FORBIDDEN_FIELDS)
        if forbidden:
            issues.append(
                LearningEvidenceQualityIssue(
                    severity="blocking",
                    code="actor_safe_view_leakage",
                    message="actor_safe_view contains forbidden fields",
                    evidence={"row_index": index, "forbidden_fields": forbidden},
                )
            )
    return tuple(issues)


def evaluate_learning_evidence_quality(
    dataset: LearningEvidenceDataset,
) -> LearningEvidenceQualityReport:
    """Audit Stage 7 evidence quality and readiness for Stage 8 interface work."""

    issues: list[LearningEvidenceQualityIssue] = []
    view_coverage = _view_coverage(dataset)
    if view_coverage["missing_views"]:
        issues.append(
            LearningEvidenceQualityIssue(
                severity="blocking",
                code="missing_required_views",
                message="learning evidence dataset is missing required views",
                evidence=view_coverage,
            )
        )

    if dataset.row_count == 0:
        issues.append(
            LearningEvidenceQualityIssue(
                severity="blocking",
                code="no_evidence_rows",
                message="learning evidence dataset has no rows",
                evidence={"dataset_id": dataset.dataset_id},
            )
        )
    if dataset.target_count == 0:
        issues.append(
            LearningEvidenceQualityIssue(
                severity="blocking",
                code="no_learning_targets",
                message="learning evidence dataset has no edge-delta targets",
                evidence={"dataset_id": dataset.dataset_id},
            )
        )

    actor_issues = _actor_leakage_issues(dataset)
    issues.extend(actor_issues)

    family_coverage = _topology_family_coverage(dataset)
    if family_coverage["missing_required_families"]:
        issues.append(
            LearningEvidenceQualityIssue(
                severity="blocking",
                code="missing_topology_family_coverage",
                message="learning evidence lacks required weak sparse dense topology families",
                evidence=family_coverage,
            )
        )

    metric_summary, metric_issues = _metric_range_summary(dataset)
    issues.extend(metric_issues)

    target_summary = _target_quality_summary(dataset)
    if target_summary["missing_action_types"]:
        issues.append(
            LearningEvidenceQualityIssue(
                severity="blocking",
                code="missing_edge_delta_action_types",
                message="edge-delta targets do not cover add remove and keep actions",
                evidence=target_summary,
            )
        )
    if target_summary["nonzero_delta_target_count"] == 0:
        issues.append(
            LearningEvidenceQualityIssue(
                severity="blocking",
                code="no_nonzero_edge_delta_targets",
                message="edge-delta targets contain no informative objective deltas",
                evidence=target_summary,
            )
        )

    feasibility_summary = _feasibility_summary(dataset)
    if feasibility_summary["feasible_row_count"] == 0:
        issues.append(
            LearningEvidenceQualityIssue(
                severity="warning",
                code="no_feasible_rows_at_current_tau",
                message=(
                    "dataset is structurally useful but not sufficient as training "
                    "coverage because all rows are infeasible at the declared tau"
                ),
                evidence=feasibility_summary,
            )
        )
    if feasibility_summary["feasibility_class_count"] < 2:
        issues.append(
            LearningEvidenceQualityIssue(
                severity="warning",
                code="low_feasibility_class_diversity",
                message="dataset lacks both feasible and infeasible classes",
                evidence=feasibility_summary,
            )
        )

    gates_passed = _gates_passed(issues)
    gates_deferred = (
        "stage8_actor_policy_interface_owner_approval",
        "training_execution_gate",
        "model_implementation_gate",
        "checkpoint_creation_gate",
        "large_scale_dataset_quality_gate",
    )
    blocking_issue_count = sum(1 for issue in issues if issue.severity == "blocking")
    stage7_exit_ready = blocking_issue_count == 0
    return LearningEvidenceQualityReport(
        dataset_id=dataset.dataset_id,
        row_count=dataset.row_count,
        target_count=dataset.target_count,
        view_coverage=view_coverage,
        topology_family_coverage=family_coverage,
        actor_leakage_summary={
            "leakage_issue_count": len(actor_issues),
            "forbidden_field_registry_size": len(ACTOR_VIEW_FORBIDDEN_FIELDS),
            "actor_safe_rows_checked": sum(len(row.actor_safe_rows) for row in dataset.rows),
        },
        metric_range_summary=metric_summary,
        target_quality_summary=target_summary,
        feasibility_summary=feasibility_summary,
        issues=tuple(issues),
        gates_passed=gates_passed,
        gates_deferred=gates_deferred,
        stage7_exit_ready=stage7_exit_ready,
        stage8_policy_interface_ready=stage7_exit_ready,
    )


def _view_coverage(dataset: LearningEvidenceDataset) -> dict[str, object]:
    present = set(dataset.views)
    required = set(REQUIRED_LEARNING_EVIDENCE_VIEWS)
    return {
        "required_views": list(REQUIRED_LEARNING_EVIDENCE_VIEWS),
        "present_views": list(dataset.views),
        "missing_views": sorted(required - present),
        "extra_views": sorted(present - required),
    }


def _actor_leakage_issues(
    dataset: LearningEvidenceDataset,
) -> tuple[LearningEvidenceQualityIssue, ...]:
    issues: list[LearningEvidenceQualityIssue] = []
    for row_index, row in enumerate(dataset.rows):
        try:
            validate_actor_safe_view(row.actor_safe_rows)
        except Exception as exc:  # pragma: no cover - defensive for malformed external rows
            issues.append(
                LearningEvidenceQualityIssue(
                    severity="blocking",
                    code="actor_safe_view_validation_failed",
                    message=str(exc),
                    evidence={"row_index": row_index, "topology_id": row.topology_id},
                )
            )
        for issue in detect_actor_view_leakage(row.actor_safe_rows):
            issues.append(
                LearningEvidenceQualityIssue(
                    severity=issue.severity,
                    code=issue.code,
                    message=issue.message,
                    evidence={**dict(issue.evidence), "topology_id": row.topology_id},
                )
            )
    return tuple(issues)


def _topology_family_coverage(dataset: LearningEvidenceDataset) -> dict[str, object]:
    families = Counter(
        str(row.diagnostics.get("topology_family", "unknown")) for row in dataset.rows
    )
    present = set(families)
    required = set(REQUIRED_STAGE7_1_TOPOLOGY_FAMILIES)
    return {
        "required_families": list(REQUIRED_STAGE7_1_TOPOLOGY_FAMILIES),
        "family_counts": dict(sorted(families.items())),
        "missing_required_families": sorted(required - present),
        "oracle_candidate_rows": sum(
            1 for row in dataset.rows if bool(row.diagnostics.get("is_oracle_candidate"))
        ),
        "full_graph_baseline_rows": sum(
            1 for row in dataset.rows if bool(row.diagnostics.get("is_full_graph_baseline"))
        ),
        "oracle_candidate_actor_input_rows": sum(
            1
            for row in dataset.rows
            if bool(row.diagnostics.get("is_oracle_candidate"))
            and bool(row.diagnostics.get("is_deployment_actor_input"))
        ),
    }


def _metric_range_summary(
    dataset: LearningEvidenceDataset,
) -> tuple[dict[str, object], tuple[LearningEvidenceQualityIssue, ...]]:
    issues: list[LearningEvidenceQualityIssue] = []
    probabilities = [row.consensus_success_probability for row in dataset.rows]
    latencies = [row.latency for row in dataset.rows]
    energies = [row.energy for row in dataset.rows]

    for row in dataset.rows:
        if not 0.0 <= row.consensus_success_probability <= 1.0:
            issues.append(
                LearningEvidenceQualityIssue(
                    severity="blocking",
                    code="probability_out_of_range",
                    message="consensus success probability is outside [0, 1]",
                    evidence={"topology_id": row.topology_id, "value": row.consensus_success_probability},
                )
            )
        if row.latency < 0.0:
            issues.append(
                LearningEvidenceQualityIssue(
                    severity="blocking",
                    code="negative_latency",
                    message="latency must be nonnegative",
                    evidence={"topology_id": row.topology_id, "value": row.latency},
                )
            )
        if row.energy < 0.0:
            issues.append(
                LearningEvidenceQualityIssue(
                    severity="blocking",
                    code="negative_energy",
                    message="energy must be nonnegative",
                    evidence={"topology_id": row.topology_id, "value": row.energy},
                )
            )

    return (
        {
            "probability_min": min(probabilities) if probabilities else None,
            "probability_max": max(probabilities) if probabilities else None,
            "latency_min": min(latencies) if latencies else None,
            "latency_max": max(latencies) if latencies else None,
            "energy_min": min(energies) if energies else None,
            "energy_max": max(energies) if energies else None,
            "metric_rows_checked": dataset.row_count,
        },
        tuple(issues),
    )


def _target_quality_summary(dataset: LearningEvidenceDataset) -> dict[str, object]:
    action_counts = Counter(target.action_type for target in dataset.edge_delta_targets)
    present = set(action_counts)
    required = set(REQUIRED_STAGE7_1_ACTION_TYPES)
    nonzero_delta_count = sum(
        1
        for target in dataset.edge_delta_targets
        if target.delta_consensus_success_probability != 0.0
        or target.delta_latency != 0.0
        or target.delta_energy != 0.0
        or target.delta_feasibility != 0
    )
    return {
        "required_action_types": list(REQUIRED_STAGE7_1_ACTION_TYPES),
        "action_type_counts": dict(sorted(action_counts.items())),
        "missing_action_types": sorted(required - present),
        "nonzero_delta_target_count": nonzero_delta_count,
        "target_count": dataset.target_count,
        "all_targets_learning_only": all(
            target.target_role == "learning_target_only" for target in dataset.edge_delta_targets
        ),
    }


def _feasibility_summary(dataset: LearningEvidenceDataset) -> dict[str, object]:
    feasible_count = sum(1 for row in dataset.rows if row.feasible_under_tau_requirement)
    infeasible_count = dataset.row_count - feasible_count
    tau_values = sorted(
        {
            row.diagnostics.get("tau_requirement_min")
            for row in dataset.rows
            if row.diagnostics.get("tau_requirement_min") is not None
        }
    )
    return {
        "feasible_row_count": feasible_count,
        "infeasible_row_count": infeasible_count,
        "feasibility_class_count": int(feasible_count > 0) + int(infeasible_count > 0),
        "tau_requirement_values": tau_values,
    }


def _gates_passed(
    issues: Iterable[LearningEvidenceQualityIssue],
) -> tuple[str, ...]:
    blocking_codes = {issue.code for issue in issues if issue.severity == "blocking"}
    passed: list[str] = []
    gate_by_code = {
        "missing_required_views": "view_coverage_gate",
        "no_evidence_rows": "evidence_rows_present_gate",
        "no_learning_targets": "learning_targets_present_gate",
        "actor_safe_view_leakage": "actor_leakage_gate",
        "actor_safe_view_validation_failed": "actor_leakage_gate",
        "missing_topology_family_coverage": "topology_family_coverage_gate",
        "probability_out_of_range": "metric_range_gate",
        "negative_latency": "metric_range_gate",
        "negative_energy": "metric_range_gate",
        "missing_edge_delta_action_types": "edge_delta_action_coverage_gate",
        "no_nonzero_edge_delta_targets": "edge_delta_informativeness_gate",
    }
    all_gates = tuple(sorted(set(gate_by_code.values())))
    failed_gates = {gate_by_code[code] for code in blocking_codes if code in gate_by_code}
    for gate in all_gates:
        if gate not in failed_gates:
            passed.append(gate)
    return tuple(passed)
