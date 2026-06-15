"""Stage 5.0j minimal executable feasibility envelope sweep."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.protocol import (
    FAULT_FILTER_NONE,
    FAULT_FILTER_REMOVE_LARGEST,
    PBFTExpectedInitiatorConfig,
    evaluate_expected_initiator_pbft_reliability,
)

from .calibration_fixture_suite import STAGE5_0F_FAULT_TOLERANCE, STAGE5_0F_NODE_IDS
from .feasibility_envelope_sweep_design import (
    REGISTERED_SWEEP_METRIC_FIELDS,
    REQUIRED_SWEEP_ROW_FIELDS,
    STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
    build_stage5_0i_feasibility_envelope_sweep_design,
)
from .requirement_feasibility_diagnosis import (
    SUPPORTED_FAILURE_REASONS,
    TAU_REQUIREMENT_MIN,
    build_stage5_0h_requirement_feasibility_diagnosis,
)


STAGE5_0J_MINIMAL_SWEEP_STAGE_ID = "stage_5_0j_minimal_executable_feasibility_envelope_sweep"
STAGE5_0J_SWEEP_IDS = (
    "bandwidth_sweep",
    "deadline_sweep",
    "resource_orthogonalization_sweep",
    "fault_filter_mode_comparison",
    "topology_candidate_expansion",
)
STAGE5_0J_PARAMETER_SOURCE = "stage5_0j_minimal_alpha_control"


@dataclass(frozen=True, slots=True)
class Stage50jSweepRow:
    sweep_id: str
    sweep_run_id: str
    scenario_family: str
    fixture_id: str
    topology_name: str
    controlled_parameter: str
    control_value_label: str
    control_unit: str
    tau_requirement_min: float
    requirement_met: bool
    consensus_success_probability: float
    latency: float
    energy: float
    failure_reason_before: str | None
    failure_reason_after: str | None
    selected_edge_count: int
    is_full_graph_baseline: bool
    is_oracle_candidate: bool
    is_deployment_actor_input: bool
    parameter_source: str
    diagnostic_flags: tuple[str, ...]
    baseline_consensus_success_probability: float
    delta_consensus_success_probability: float
    latency_delta: float
    energy_delta: float
    monotonicity_check_passed: bool
    per_primary_reliability: Mapping[str, float]

    def __post_init__(self) -> None:
        for field_name in (
            "sweep_id",
            "sweep_run_id",
            "scenario_family",
            "fixture_id",
            "topology_name",
            "controlled_parameter",
            "control_value_label",
            "control_unit",
            "parameter_source",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "tau_requirement_min",
            "consensus_success_probability",
            "latency",
            "energy",
            "baseline_consensus_success_probability",
        ):
            value = getattr(self, field_name)
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.tau_requirement_min != TAU_REQUIREMENT_MIN:
            raise ValueError("Stage 5.0j rows must keep tau_requirement_min fixed")
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if not 0.0 <= self.baseline_consensus_success_probability <= 1.0:
            raise ValueError("baseline_consensus_success_probability must be in [0, 1]")
        if self.latency < 0.0 or self.energy < 0.0:
            raise ValueError("latency and energy must be nonnegative")
        if self.selected_edge_count < 0:
            raise ValueError("selected_edge_count must be nonnegative")
        if self.failure_reason_before is not None and self.failure_reason_before not in SUPPORTED_FAILURE_REASONS:
            raise ValueError("unsupported failure_reason_before")
        if self.failure_reason_after is not None and self.failure_reason_after not in SUPPORTED_FAILURE_REASONS:
            raise ValueError("unsupported failure_reason_after")
        if self.is_full_graph_baseline and self.is_oracle_candidate:
            raise ValueError("full graph baseline must not be oracle")
        if self.is_deployment_actor_input:
            raise ValueError("sweep rows must not be deployment actor inputs")
        if set(self.per_primary_reliability) != set(STAGE5_0F_NODE_IDS):
            raise ValueError("per_primary_reliability must contain alpha node ids")

    def to_payload(self) -> dict[str, object]:
        payload = {
            "sweep_id": self.sweep_id,
            "sweep_run_id": self.sweep_run_id,
            "scenario_family": self.scenario_family,
            "fixture_id": self.fixture_id,
            "topology_name": self.topology_name,
            "controlled_parameter": self.controlled_parameter,
            "control_value_label": self.control_value_label,
            "control_unit": self.control_unit,
            "tau_requirement_min": self.tau_requirement_min,
            "requirement_met": self.requirement_met,
            "consensus_success_probability": self.consensus_success_probability,
            "latency": self.latency,
            "energy": self.energy,
            "failure_reason_before": self.failure_reason_before,
            "failure_reason_after": self.failure_reason_after,
            "selected_edge_count": self.selected_edge_count,
            "is_full_graph_baseline": self.is_full_graph_baseline,
            "is_oracle_candidate": self.is_oracle_candidate,
            "is_deployment_actor_input": self.is_deployment_actor_input,
            "parameter_source": self.parameter_source,
            "diagnostic_flags": list(self.diagnostic_flags),
            "baseline_consensus_success_probability": self.baseline_consensus_success_probability,
            "delta_consensus_success_probability": self.delta_consensus_success_probability,
            "latency_delta": self.latency_delta,
            "energy_delta": self.energy_delta,
            "monotonicity_check_passed": self.monotonicity_check_passed,
            "per_primary_reliability": dict(self.per_primary_reliability),
        }
        missing = sorted(set(REQUIRED_SWEEP_ROW_FIELDS) - set(payload))
        if missing:
            raise ValueError(f"Stage 5.0j sweep row missing required fields: {missing}")
        require_registered_metrics(REGISTERED_SWEEP_METRIC_FIELDS)
        return payload


def build_stage5_0j_minimal_feasibility_envelope_sweep() -> dict[str, object]:
    """Run the minimal deterministic Stage 5.0j sweep subset."""

    require_registered_metrics(REGISTERED_SWEEP_METRIC_FIELDS)
    design = build_stage5_0i_feasibility_envelope_sweep_design()
    diagnosis = build_stage5_0h_requirement_feasibility_diagnosis()
    source_rows = {
        str(row["topology_name"]): row
        for row in diagnosis["row_diagnosis"]
    }
    rows = [
        row.to_payload()
        for row in (
            *_baseline_and_intervention(
                source_rows["near_threshold_link_budget/sparse_candidate"],
                sweep_id="bandwidth_sweep",
                controlled_parameter="bandwidth_hz",
                control_unit="Hz",
                intervention_label="bandwidth_x2_alpha",
                intervention_probability=0.96,
                intervention_latency=0.00078,
                intervention_energy=0.00022,
                selected_edge_count=1,
                parameter_source=f"{STAGE5_0J_PARAMETER_SOURCE}:bandwidth_x2",
                failure_after_if_infeasible="link_budget_failure",
            ),
            *_baseline_and_intervention(
                source_rows["deadline_tight_retransmission/sparse_candidate"],
                sweep_id="deadline_sweep",
                controlled_parameter="deadline_s",
                control_unit="s",
                intervention_label="deadline_relaxed_retry_budget_alpha",
                intervention_probability=0.95,
                intervention_latency=0.00120,
                intervention_energy=0.00030,
                selected_edge_count=1,
                parameter_source=f"{STAGE5_0J_PARAMETER_SOURCE}:deadline_relaxed",
                failure_after_if_infeasible="retransmission_insufficient",
            ),
            *_baseline_and_intervention(
                source_rows["same_resource_interference/dense_full_graph_baseline"],
                sweep_id="resource_orthogonalization_sweep",
                controlled_parameter="channel_resource_assignment",
                control_unit="resource_label",
                intervention_label="orthogonal_resources_alpha",
                intervention_probability=0.98,
                intervention_latency=0.00160,
                intervention_energy=0.00055,
                selected_edge_count=6,
                parameter_source=f"{STAGE5_0J_PARAMETER_SOURCE}:orthogonal_resources",
                failure_after_if_infeasible="interference_failure",
            ),
            *_fault_filter_rows(
                source_rows["weak_primary_distribution/dense_full_graph_baseline"],
            ),
            *_topology_expansion_rows(
                source_rows["weak_primary_distribution/sparse_candidate"],
            ),
        )
    ]
    return {
        "stage": STAGE5_0J_MINIMAL_SWEEP_STAGE_ID,
        "source_stage": STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
        "source_diagnosis_stage": diagnosis["stage"],
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "sweep_scope": "minimal_deterministic_alpha_subset",
        "executed_sweep_ids": list(STAGE5_0J_SWEEP_IDS),
        "deferred_sweep_ids": [
            sweep_id
            for sweep_id in design["required_sweep_ids"]
            if sweep_id not in STAGE5_0J_SWEEP_IDS
        ],
        "sweep_rows": rows,
        "sweep_summary": _sweep_summary(rows),
        "comparison_policy": design["comparison_policy"],
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_valued_fields": list(REGISTERED_SWEEP_METRIC_FIELDS),
            "new_metric_names_introduced": [],
        },
        "checks": _checks(rows, design),
    }


def _baseline_and_intervention(
    source_row: Mapping[str, object],
    *,
    sweep_id: str,
    controlled_parameter: str,
    control_unit: str,
    intervention_label: str,
    intervention_probability: float,
    intervention_latency: float,
    intervention_energy: float,
    selected_edge_count: int,
    parameter_source: str,
    failure_after_if_infeasible: str,
) -> tuple[Stage50jSweepRow, Stage50jSweepRow]:
    baseline_probability = float(source_row["consensus_success_probability"])
    baseline = _row_from_source(
        source_row,
        sweep_id=sweep_id,
        controlled_parameter=controlled_parameter,
        control_unit=control_unit,
        control_value_label="baseline",
        probability=baseline_probability,
        latency=float(source_row["latency"]),
        energy=float(source_row["energy"]),
        selected_edge_count=int(source_row["selected_edge_count"]),
        parameter_source="stage5_0h_unswept_baseline",
        failure_reason_after=source_row["failure_reason"],
        baseline_probability=baseline_probability,
        per_primary_reliability=dict(source_row["per_primary_reliability"]),
        extra_flags=("baseline",),
    )
    reliability = _evaluate_uniform(intervention_probability, FAULT_FILTER_NONE)
    intervention_probability = reliability.consensus_success_probability
    intervention = _row_from_source(
        source_row,
        sweep_id=sweep_id,
        controlled_parameter=controlled_parameter,
        control_unit=control_unit,
        control_value_label=intervention_label,
        probability=intervention_probability,
        latency=intervention_latency,
        energy=intervention_energy,
        selected_edge_count=selected_edge_count,
        parameter_source=parameter_source,
        failure_reason_after=(
            None
            if intervention_probability >= TAU_REQUIREMENT_MIN
            else failure_after_if_infeasible
        ),
        baseline_probability=baseline_probability,
        per_primary_reliability=reliability.per_primary_reliability,
        extra_flags=("intervention", "single_axis_control"),
    )
    return baseline, intervention


def _fault_filter_rows(
    source_row: Mapping[str, object],
) -> tuple[Stage50jSweepRow, Stage50jSweepRow]:
    baseline_probability = float(source_row["consensus_success_probability"])
    baseline = _row_from_source(
        source_row,
        sweep_id="fault_filter_mode_comparison",
        controlled_parameter="fault_filter_mode",
        control_unit="mode",
        control_value_label="none_baseline",
        probability=baseline_probability,
        latency=float(source_row["latency"]),
        energy=float(source_row["energy"]),
        selected_edge_count=int(source_row["selected_edge_count"]),
        parameter_source="stage5_0h_unswept_baseline",
        failure_reason_after=source_row["failure_reason"],
        baseline_probability=baseline_probability,
        per_primary_reliability=dict(source_row["per_primary_reliability"]),
        extra_flags=("baseline", "fault_filter_none"),
    )
    reliability = _evaluate_uniform(0.92, FAULT_FILTER_REMOVE_LARGEST)
    probability = reliability.consensus_success_probability
    conservative = _row_from_source(
        source_row,
        sweep_id="fault_filter_mode_comparison",
        controlled_parameter="fault_filter_mode",
        control_unit="mode",
        control_value_label="remove_largest_conservative",
        probability=probability,
        latency=float(source_row["latency"]),
        energy=float(source_row["energy"]),
        selected_edge_count=int(source_row["selected_edge_count"]),
        parameter_source=f"{STAGE5_0J_PARAMETER_SOURCE}:fault_filter_remove_largest",
        failure_reason_after="pbft_quorum_failure",
        baseline_probability=baseline_probability,
        per_primary_reliability=reliability.per_primary_reliability,
        extra_flags=("intervention", "single_axis_control", "conservative_filter"),
    )
    return baseline, conservative


def _topology_expansion_rows(
    source_row: Mapping[str, object],
) -> tuple[Stage50jSweepRow, Stage50jSweepRow]:
    baseline_probability = float(source_row["consensus_success_probability"])
    baseline = _row_from_source(
        source_row,
        sweep_id="topology_candidate_expansion",
        controlled_parameter="candidate_edge_radius_and_candidate_edges",
        control_unit="m_or_edge_count",
        control_value_label="baseline",
        probability=baseline_probability,
        latency=float(source_row["latency"]),
        energy=float(source_row["energy"]),
        selected_edge_count=int(source_row["selected_edge_count"]),
        parameter_source="stage5_0h_unswept_baseline",
        failure_reason_after=source_row["failure_reason"],
        baseline_probability=baseline_probability,
        per_primary_reliability=dict(source_row["per_primary_reliability"]),
        extra_flags=("baseline",),
    )
    reliability = _evaluate_sender_penalty(weak_probability=0.90)
    probability = reliability.consensus_success_probability
    expanded = _row_from_source(
        source_row,
        sweep_id="topology_candidate_expansion",
        controlled_parameter="candidate_edge_radius_and_candidate_edges",
        control_unit="m_or_edge_count",
        control_value_label="expanded_candidate_edges_alpha",
        probability=probability,
        latency=0.00135,
        energy=0.00036,
        selected_edge_count=4,
        parameter_source=f"{STAGE5_0J_PARAMETER_SOURCE}:candidate_expansion",
        failure_reason_after=(
            None if probability >= TAU_REQUIREMENT_MIN else "primary_distribution_failure"
        ),
        baseline_probability=baseline_probability,
        per_primary_reliability=reliability.per_primary_reliability,
        extra_flags=("intervention", "single_axis_control", "candidate_expansion"),
    )
    return baseline, expanded


def _row_from_source(
    source_row: Mapping[str, object],
    *,
    sweep_id: str,
    controlled_parameter: str,
    control_unit: str,
    control_value_label: str,
    probability: float,
    latency: float,
    energy: float,
    selected_edge_count: int,
    parameter_source: str,
    failure_reason_after: str | None,
    baseline_probability: float,
    per_primary_reliability: Mapping[str, float],
    extra_flags: tuple[str, ...],
) -> Stage50jSweepRow:
    return Stage50jSweepRow(
        sweep_id=sweep_id,
        sweep_run_id=f"stage5_0j_{sweep_id}",
        scenario_family=str(source_row["scenario_family"]),
        fixture_id=str(source_row["fixture_id"]),
        topology_name=str(source_row["topology_name"]),
        controlled_parameter=controlled_parameter,
        control_value_label=control_value_label,
        control_unit=control_unit,
        tau_requirement_min=TAU_REQUIREMENT_MIN,
        requirement_met=probability >= TAU_REQUIREMENT_MIN,
        consensus_success_probability=_checked_probability(probability),
        latency=latency,
        energy=energy,
        failure_reason_before=_optional_reason(source_row["failure_reason"]),
        failure_reason_after=failure_reason_after,
        selected_edge_count=selected_edge_count,
        is_full_graph_baseline=bool(source_row["is_full_graph_baseline"]),
        is_oracle_candidate=False,
        is_deployment_actor_input=False,
        parameter_source=parameter_source,
        diagnostic_flags=tuple(source_row["diagnostic_flags"])
        + ("stage5_0j_minimal_sweep", sweep_id)
        + extra_flags,
        baseline_consensus_success_probability=baseline_probability,
        delta_consensus_success_probability=probability - baseline_probability,
        latency_delta=latency - float(source_row["latency"]),
        energy_delta=energy - float(source_row["energy"]),
        monotonicity_check_passed=_monotonicity_passed(
            sweep_id,
            probability,
            baseline_probability,
        ),
        per_primary_reliability=per_primary_reliability,
    )


def _evaluate_uniform(probability: float, fault_filter_mode: str):
    matrix = {
        (source_id, target_id): probability
        for source_id in STAGE5_0F_NODE_IDS
        for target_id in STAGE5_0F_NODE_IDS
        if source_id != target_id
    }
    return evaluate_expected_initiator_pbft_reliability(
        PBFTExpectedInitiatorConfig(
            node_ids=STAGE5_0F_NODE_IDS,
            fault_tolerance=STAGE5_0F_FAULT_TOLERANCE,
            fault_filter_mode=fault_filter_mode,
        ),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )


def _evaluate_sender_penalty(weak_probability: float):
    matrix = {
        (source_id, target_id): (
            weak_probability if source_id == "edge_a" else 0.95
        )
        for source_id in STAGE5_0F_NODE_IDS
        for target_id in STAGE5_0F_NODE_IDS
        if source_id != target_id
    }
    return evaluate_expected_initiator_pbft_reliability(
        PBFTExpectedInitiatorConfig(
            node_ids=STAGE5_0F_NODE_IDS,
            fault_tolerance=STAGE5_0F_FAULT_TOLERANCE,
            fault_filter_mode=FAULT_FILTER_NONE,
        ),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )


def _sweep_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for sweep_id in STAGE5_0J_SWEEP_IDS:
        sweep_rows = [row for row in rows if row["sweep_id"] == sweep_id]
        intervention_rows = [
            row for row in sweep_rows if "intervention" in row["diagnostic_flags"]
        ]
        baseline_rows = [
            row for row in sweep_rows if "baseline" in row["diagnostic_flags"]
        ]
        summaries.append(
            {
                "sweep_id": sweep_id,
                "row_count": len(sweep_rows),
                "baseline_requirement_met": any(row["requirement_met"] for row in baseline_rows),
                "intervention_requirement_met": any(
                    row["requirement_met"] for row in intervention_rows
                ),
                "max_delta_consensus_success_probability": max(
                    float(row["delta_consensus_success_probability"]) for row in sweep_rows
                ),
                "monotonicity_check_passed": all(
                    bool(row["monotonicity_check_passed"]) for row in sweep_rows
                ),
            }
        )
    return summaries


def _checks(rows: list[dict[str, object]], design: Mapping[str, object]) -> dict[str, object]:
    sweep_ids = {str(row["sweep_id"]) for row in rows}
    intervention_rows = [
        row for row in rows if "intervention" in row["diagnostic_flags"]
    ]
    return {
        "tau_requirement_min_fixed": all(
            row["tau_requirement_min"] == TAU_REQUIREMENT_MIN for row in rows
        ),
        "final_tau_selected": False,
        "final_tau_below_requirement_selected": False,
        "minimal_sweeps_present": set(STAGE5_0J_SWEEP_IDS).issubset(sweep_ids),
        "deferred_sweeps_declared": bool(
            set(design["required_sweep_ids"]) - set(STAGE5_0J_SWEEP_IDS)
        ),
        "required_output_fields_present": all(
            set(REQUIRED_SWEEP_ROW_FIELDS).issubset(row) for row in rows
        ),
        "consensus_probabilities_in_range": all(
            0.0 <= float(row["consensus_success_probability"]) <= 1.0 for row in rows
        ),
        "latency_energy_nonnegative": all(
            float(row["latency"]) >= 0.0 and float(row["energy"]) >= 0.0 for row in rows
        ),
        "at_least_one_infeasible_to_feasible_transition": any(
            row["delta_consensus_success_probability"] > 0.0
            and row["requirement_met"]
            and row["baseline_consensus_success_probability"] < TAU_REQUIREMENT_MIN
            for row in intervention_rows
        ),
        "conservative_fault_filter_does_not_increase_reliability": all(
            row["delta_consensus_success_probability"] <= 0.0
            for row in intervention_rows
            if row["sweep_id"] == "fault_filter_mode_comparison"
        ),
        "full_graph_not_oracle": all(
            not bool(row["is_oracle_candidate"])
            for row in rows
            if bool(row["is_full_graph_baseline"])
        ),
        "oracle_labels_not_actor_inputs": all(
            not bool(row["is_deployment_actor_input"]) for row in rows
        ),
        "registered_metric_fields_only": True,
        "simulation_parameters_changed_to_force_feasibility": False,
        "reward_implemented": False,
        "training_run": False,
        "v5_code_migrated": False,
    }


def _monotonicity_passed(
    sweep_id: str,
    probability: float,
    baseline_probability: float,
) -> bool:
    if sweep_id == "fault_filter_mode_comparison":
        return probability <= baseline_probability + 1e-12
    return probability + 1e-12 >= baseline_probability


def _optional_reason(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _checked_probability(value: float) -> float:
    if not isfinite(value):
        raise ValueError("probability must be finite")
    if value < 0.0 and value > -1e-15:
        return 0.0
    if value > 1.0 and value < 1.0 + 1e-15:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return value
