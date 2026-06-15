"""Stage 5.4 surrogate diagnostic report integration without training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from marl_topology.objectives import (
    NormalizationReferenceConfig,
    SurrogateSignalConfig,
    SurrogateSignalInput,
    evaluate_reward_surrogate,
    select_normalization_references,
)

from .feasibility_envelope_sweep_range_review import (
    STAGE5_0L_RANGE_REVIEW_STAGE_ID,
    build_stage5_0l_stage3_backed_sweep_range_review,
)
from .normalization_reference_selection import (
    STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID,
    build_stage5_3_normalization_reference_selection,
)
from .requirement_feasibility_diagnosis import TAU_REQUIREMENT_MIN


STAGE5_4_REWARD_REPORT_STAGE_ID = (
    "stage_5_4_reward_report_integration_without_training"
)
STAGE5_4_RECOMMENDED_NEXT_TASK = "stage_5_5_training_preflight_review_without_training"
STAGE5_4_COMPONENT_POLICY = "component_only_no_scalar_reward_v1"
STAGE5_4_COMPONENT_CONFIG_ID = "stage5_4_component_only_zero_weight_config"


@dataclass(frozen=True, slots=True)
class SurrogateDiagnosticRow:
    """Training-only diagnostic row with no scalar training signal reported."""

    source_stage: str
    source_row_id: str
    scenario_family: str
    topology_name: str
    consensus_success_probability: float
    latency: float
    energy: float
    reliability_violation: float
    reliability_penalty: float
    normalized_latency: float
    normalized_energy: float
    latency_penalty: float
    energy_penalty: float
    constraint_satisfied: bool
    tau: float
    latency_reference_s: float
    energy_reference_j: float
    normalization_policy: str
    component_policy: str
    training_only: bool
    is_deployment_actor_input: bool
    surrogate_outputs_are_metrics: bool
    scalar_surrogate_reported: bool
    config_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_stage",
            "source_row_id",
            "scenario_family",
            "topology_name",
            "normalization_policy",
            "component_policy",
            "config_id",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "consensus_success_probability",
            "latency",
            "energy",
            "reliability_violation",
            "reliability_penalty",
            "normalized_latency",
            "normalized_energy",
            "latency_penalty",
            "energy_penalty",
            "tau",
            "latency_reference_s",
            "energy_reference_j",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if self.latency < 0.0 or self.energy < 0.0:
            raise ValueError("latency and energy must be nonnegative")
        if self.latency_reference_s <= 0.0 or self.energy_reference_j <= 0.0:
            raise ValueError("normalization references must be positive")
        if not self.training_only:
            raise ValueError("Stage 5.4 diagnostics must be training-only")
        if self.is_deployment_actor_input:
            raise ValueError("Stage 5.4 diagnostics must not be actor inputs")
        if self.surrogate_outputs_are_metrics:
            raise ValueError("Stage 5.4 diagnostics must not be metrics")
        if self.scalar_surrogate_reported:
            raise ValueError("Stage 5.4 must not report scalar surrogate values")

    def to_payload(self) -> dict[str, object]:
        return {
            "source_stage": self.source_stage,
            "source_row_id": self.source_row_id,
            "scenario_family": self.scenario_family,
            "topology_name": self.topology_name,
            "consensus_success_probability": self.consensus_success_probability,
            "latency": self.latency,
            "energy": self.energy,
            "reliability_violation": self.reliability_violation,
            "reliability_penalty": self.reliability_penalty,
            "normalized_latency": self.normalized_latency,
            "normalized_energy": self.normalized_energy,
            "latency_penalty": self.latency_penalty,
            "energy_penalty": self.energy_penalty,
            "constraint_satisfied": self.constraint_satisfied,
            "tau": self.tau,
            "latency_reference_s": self.latency_reference_s,
            "energy_reference_j": self.energy_reference_j,
            "normalization_policy": self.normalization_policy,
            "component_policy": self.component_policy,
            "training_only": self.training_only,
            "is_deployment_actor_input": self.is_deployment_actor_input,
            "surrogate_outputs_are_metrics": self.surrogate_outputs_are_metrics,
            "scalar_surrogate_reported": self.scalar_surrogate_reported,
            "config_id": self.config_id,
        }


def build_stage5_4_reward_report_integration() -> dict[str, object]:
    """Build Stage 5.4 training-only surrogate diagnostic report."""

    source_report = build_stage5_0l_stage3_backed_sweep_range_review()
    stage5_3_report = build_stage5_3_normalization_reference_selection()
    references = select_normalization_references(
        source_report["sweep_rows"],
        NormalizationReferenceConfig(
            tau_requirement_min=TAU_REQUIREMENT_MIN,
            source_stage_id=STAGE5_0L_RANGE_REVIEW_STAGE_ID,
        ),
    )
    component_config = SurrogateSignalConfig(
        tau=TAU_REQUIREMENT_MIN,
        reliability_weight=0.0,
        latency_weight=0.0,
        energy_weight=0.0,
        latency_reference_s=references.latency_reference_s,
        energy_reference_j=references.energy_reference_j,
        config_id=STAGE5_4_COMPONENT_CONFIG_ID,
    )
    rows = tuple(
        _diagnostic_row_from_source(
            source_row,
            component_config,
            references.to_payload(),
        )
        for source_row in source_report["sweep_rows"]
    )
    payload_rows = [row.to_payload() for row in rows]
    return {
        "stage": STAGE5_4_REWARD_REPORT_STAGE_ID,
        "source_stage": STAGE5_0L_RANGE_REVIEW_STAGE_ID,
        "normalization_source_stage": STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID,
        "normalization_reference": stage5_3_report["normalization_reference"],
        "component_policy": STAGE5_4_COMPONENT_POLICY,
        "surrogate_diagnostic_rows": payload_rows,
        "diagnostic_summary": _diagnostic_summary(rows),
        "metric_governance": {
            "new_metric_names_introduced": [],
            "source_metric_inputs": [
                "consensus_success_probability",
                "latency",
                "energy",
                "topology_diagnostics",
            ],
            "surrogate_outputs_are_metrics": False,
            "wide_metric_export": False,
        },
        "checks": _checks(source_report, stage5_3_report, rows),
    }


def _diagnostic_row_from_source(
    source_row: Mapping[str, object],
    config: SurrogateSignalConfig,
    reference_payload: Mapping[str, object],
) -> SurrogateDiagnosticRow:
    record = evaluate_reward_surrogate(
        SurrogateSignalInput(
            consensus_success_probability=float(
                source_row["consensus_success_probability"]
            ),
            latency=float(source_row["latency"]),
            energy=float(source_row["energy"]),
            topology_diagnostics={
                "source_stage": STAGE5_0L_RANGE_REVIEW_STAGE_ID,
                "diagnostic_flags": tuple(source_row.get("diagnostic_flags", ())),
            },
        ),
        config,
    )
    return SurrogateDiagnosticRow(
        source_stage=STAGE5_0L_RANGE_REVIEW_STAGE_ID,
        source_row_id=str(source_row["control_value_label"]),
        scenario_family=str(source_row["scenario_family"]),
        topology_name=str(source_row["topology_name"]),
        consensus_success_probability=float(source_row["consensus_success_probability"]),
        latency=float(source_row["latency"]),
        energy=float(source_row["energy"]),
        reliability_violation=record.reliability_violation,
        reliability_penalty=record.reliability_penalty,
        normalized_latency=record.normalized_latency,
        normalized_energy=record.normalized_energy,
        latency_penalty=record.latency_penalty,
        energy_penalty=record.energy_penalty,
        constraint_satisfied=record.constraint_satisfied,
        tau=record.tau,
        latency_reference_s=float(reference_payload["latency_reference_s"]),
        energy_reference_j=float(reference_payload["energy_reference_j"]),
        normalization_policy=str(reference_payload["selection_policy"]),
        component_policy=STAGE5_4_COMPONENT_POLICY,
        training_only=True,
        is_deployment_actor_input=False,
        surrogate_outputs_are_metrics=False,
        scalar_surrogate_reported=False,
        config_id=config.config_id,
    )


def _diagnostic_summary(
    rows: tuple[SurrogateDiagnosticRow, ...],
) -> dict[str, object]:
    feasible_rows = [row for row in rows if row.constraint_satisfied]
    infeasible_rows = [row for row in rows if not row.constraint_satisfied]
    return {
        "row_count": len(rows),
        "constraint_satisfied_count": len(feasible_rows),
        "constraint_violation_count": len(infeasible_rows),
        "max_reliability_violation": max(row.reliability_violation for row in rows),
        "max_normalized_latency": max(row.normalized_latency for row in rows),
        "max_normalized_energy": max(row.normalized_energy for row in rows),
        "scalar_surrogate_reported": False,
        "training_only": True,
    }


def _checks(
    source_report: Mapping[str, object],
    stage5_3_report: Mapping[str, object],
    rows: tuple[SurrogateDiagnosticRow, ...],
) -> dict[str, object]:
    return {
        "source_stage3_backed": source_report["checks"]["all_rows_stage3_backed"],
        "normalization_reference_stage5_3": (
            stage5_3_report["stage"] == STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID
        ),
        "row_count_matches_source": len(rows) == len(source_report["sweep_rows"]),
        "all_rows_training_only": all(row.training_only for row in rows),
        "no_rows_are_actor_inputs": all(not row.is_deployment_actor_input for row in rows),
        "surrogate_outputs_not_metrics": all(
            not row.surrogate_outputs_are_metrics for row in rows
        ),
        "scalar_surrogate_not_reported": all(
            not row.scalar_surrogate_reported for row in rows
        ),
        "constraint_violation_visible": any(
            row.reliability_violation > 0.0 for row in rows
        ),
        "constraint_plateau_visible": any(
            row.reliability_penalty == 0.0 and row.constraint_satisfied for row in rows
        ),
        "normalization_reference_applied": all(
            row.latency_reference_s > 0.0 and row.energy_reference_j > 0.0
            for row in rows
        ),
        "reward_weight_calibration_performed": False,
        "training_run": False,
        "model_code_added": False,
        "v5_code_migrated": False,
        "recommended_next_task": STAGE5_4_RECOMMENDED_NEXT_TASK,
    }
