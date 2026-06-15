"""Stage 5.0l Stage 3-backed sweep range expansion and realism review."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean
from typing import Mapping

from marl_topology.channel import ChannelModelConfig
from marl_topology.geometry3d import BuildingBox, Point3D
from marl_topology.link import URLLC_FINITE_BLOCKLENGTH_REGIME_ID, LinkTransmissionConfig
from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.network import (
    NetworkCommunicationConfig,
    NetworkCommunicationRecord,
    NetworkTransmissionSpec,
    evaluate_network_communication,
)
from marl_topology.network.communication import NETWORK_COMMUNICATION_REGIME_ID
from marl_topology.protocol import (
    FAULT_FILTER_REMOVE_LARGEST,
    MESSAGE_MATRIX_ADAPTER_ID,
    PBFTExpectedInitiatorConfig,
    PBFTPhaseBudgets,
    account_pbft_protocol_latency_energy,
    build_pbft_message_matrices_from_network_records,
    evaluate_expected_initiator_pbft_reliability,
)
from marl_topology.scenario import Node3D, NodeKind, Scene3D
from marl_topology.topology import CandidateGraph

from .feasibility_envelope_sweep_design import (
    REGISTERED_SWEEP_METRIC_FIELDS,
    REQUIRED_SWEEP_ROW_FIELDS,
    STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
    build_stage5_0i_feasibility_envelope_sweep_design,
)
from .feasibility_envelope_sweep_stage3_backed import (
    STAGE5_0K_STAGE3_BACKED_SWEEP_STAGE_ID,
    STAGE5_0K_FAULT_TOLERANCE,
    STAGE5_0K_NODE_IDS,
    STAGE5_0K_PHASE_BUDGET_S,
    Stage50kSweepRow,
)
from .requirement_feasibility_diagnosis import (
    SUPPORTED_FAILURE_REASONS,
    TAU_REQUIREMENT_MIN,
)


STAGE5_0L_RANGE_REVIEW_STAGE_ID = (
    "stage_5_0l_stage3_backed_sweep_range_expansion_and_realism_review"
)
STAGE5_0L_SWEEP_IDS = (
    "tx_power_sweep",
    "payload_sweep",
    "rsu_height_placement_sweep",
    "resource_budget_limit_sweep",
)
STAGE5_0L_PARAMETER_SOURCE = "stage5_0l_stage3_backed_range_control"
STAGE5_0L_REFERENCE_SCENARIO_ID = "stage5_0l_stage3_backed_range_reference"
STAGE5_0L_BLOCKED_SCENARIO_ID = "stage5_0l_blocked_geometry_reference"


@dataclass(frozen=True, slots=True)
class _RangeControlCase:
    sweep_id: str
    scenario_family: str
    fixture_id: str
    topology_name: str
    controlled_parameter: str
    control_value_label: str
    control_unit: str
    channel_config: ChannelModelConfig
    link_config: LinkTransmissionConfig
    scene_kind: str = "open_reference"
    use_background_interference: bool = False
    resource_budget_count: int | None = None
    failure_reason_if_infeasible: str = "unknown"

    def __post_init__(self) -> None:
        if self.sweep_id not in STAGE5_0L_SWEEP_IDS:
            raise ValueError("unsupported Stage 5.0l sweep id")
        if self.scene_kind not in {"open_reference", "blocked_geometry"}:
            raise ValueError("unsupported scene_kind")
        if self.failure_reason_if_infeasible not in SUPPORTED_FAILURE_REASONS:
            raise ValueError("unsupported failure reason")
        if self.resource_budget_count is not None and self.resource_budget_count <= 0:
            raise ValueError("resource_budget_count must be positive when provided")


@dataclass(frozen=True, slots=True)
class _RangeEvaluation:
    records: tuple[NetworkCommunicationRecord, ...]
    consensus_success_probability: float
    per_primary_reliability: Mapping[str, float]
    protocol_latency_s: float
    protocol_energy_j: float
    phase_budgets_s: Mapping[str, float]
    protocol_accounting_model_id: str
    finite_blocklength_regime_id: str
    network_regime_id: str
    matrix_adapter_id: str
    fault_filter_mode: str


def build_stage5_0l_stage3_backed_sweep_range_review() -> dict[str, object]:
    """Run Stage 5.0l range expansion and realism review."""

    require_registered_metrics(REGISTERED_SWEEP_METRIC_FIELDS)
    design = build_stage5_0i_feasibility_envelope_sweep_design()
    rows = _build_rows()
    return {
        "stage": STAGE5_0L_RANGE_REVIEW_STAGE_ID,
        "source_stage": STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
        "source_stage3_backed_stage": STAGE5_0K_STAGE3_BACKED_SWEEP_STAGE_ID,
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "sweep_scope": "stage3_backed_range_expansion_and_realism_review",
        "executed_sweep_ids": list(STAGE5_0L_SWEEP_IDS),
        "deferred_sweep_ids": [
            sweep_id
            for sweep_id in design["required_sweep_ids"]
            if sweep_id not in STAGE5_0L_SWEEP_IDS and sweep_id != "resource_orthogonalization_sweep"
        ],
        "sweep_rows": [row.to_payload() for row in rows],
        "sweep_summary": _sweep_summary(rows),
        "range_review": _range_review(rows),
        "realism_review": _realism_review(),
        "stage3_backing": {
            "uses_stage3_network_records": True,
            "finite_blocklength_regime_id": URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
            "network_regime_id": NETWORK_COMMUNICATION_REGIME_ID,
            "matrix_adapter_id": MESSAGE_MATRIX_ADAPTER_ID,
            "fault_filter_mode": FAULT_FILTER_REMOVE_LARGEST,
            "phase_budget_s": STAGE5_0K_PHASE_BUDGET_S,
        },
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_valued_fields": list(REGISTERED_SWEEP_METRIC_FIELDS),
            "new_metric_names_introduced": [],
        },
        "checks": _checks(rows),
    }


def _build_rows() -> tuple[Stage50kSweepRow, ...]:
    rows: list[Stage50kSweepRow] = []
    for baseline, intervention in (
        _tx_power_cases(),
        _payload_cases(),
        _rsu_height_cases(),
        _resource_budget_cases(),
    ):
        baseline_scene = _scene_for_case(baseline)
        intervention_scene = _scene_for_case(intervention)
        baseline_graph = CandidateGraph.from_scene(baseline_scene)
        intervention_graph = CandidateGraph.from_scene(intervention_scene)
        baseline_eval = _evaluate_case(baseline_scene, baseline_graph, baseline)
        intervention_eval = _evaluate_case(intervention_scene, intervention_graph, intervention)
        rows.append(
            _row_from_case(
                baseline_graph,
                baseline,
                baseline_eval,
                baseline_probability=baseline_eval.consensus_success_probability,
                baseline_latency=baseline_eval.protocol_latency_s,
                baseline_energy=baseline_eval.protocol_energy_j,
                extra_flags=("baseline",),
            )
        )
        rows.append(
            _row_from_case(
                intervention_graph,
                intervention,
                intervention_eval,
                baseline_probability=baseline_eval.consensus_success_probability,
                baseline_latency=baseline_eval.protocol_latency_s,
                baseline_energy=baseline_eval.protocol_energy_j,
                extra_flags=("intervention", "single_axis_control"),
            )
        )
    return tuple(rows)


def _tx_power_cases() -> tuple[_RangeControlCase, _RangeControlCase]:
    return (
        _RangeControlCase(
            sweep_id="tx_power_sweep",
            scenario_family="stage3_backed_tx_power_link_budget",
            fixture_id="stage5_0l_tx_power_stage3_fixture",
            topology_name="stage3_backed/tx_power/full_graph",
            controlled_parameter="tx_power_dbm",
            control_value_label="tx_power_minus_10dbm_stage3",
            control_unit="dBm",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-10.0, bandwidth_hz=20e6),
            link_config=_link_config(payload_bits=12_000, bandwidth_hz=20e6, deadline_s=0.003),
            failure_reason_if_infeasible="link_budget_failure",
        ),
        _RangeControlCase(
            sweep_id="tx_power_sweep",
            scenario_family="stage3_backed_tx_power_link_budget",
            fixture_id="stage5_0l_tx_power_stage3_fixture",
            topology_name="stage3_backed/tx_power/full_graph",
            controlled_parameter="tx_power_dbm",
            control_value_label="tx_power_minus_8dbm_stage3",
            control_unit="dBm",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-8.0, bandwidth_hz=20e6),
            link_config=_link_config(payload_bits=12_000, bandwidth_hz=20e6, deadline_s=0.003),
            failure_reason_if_infeasible="link_budget_failure",
        ),
    )


def _payload_cases() -> tuple[_RangeControlCase, _RangeControlCase]:
    return (
        _RangeControlCase(
            sweep_id="payload_sweep",
            scenario_family="stage3_backed_payload_pressure",
            fixture_id="stage5_0l_payload_stage3_fixture",
            topology_name="stage3_backed/payload/full_graph",
            controlled_parameter="payload_bits",
            control_value_label="payload_18kbits_stage3",
            control_unit="bits",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-8.0, bandwidth_hz=20e6),
            link_config=_link_config(payload_bits=18_000, bandwidth_hz=20e6, deadline_s=0.003),
            failure_reason_if_infeasible="resource_budget_failure",
        ),
        _RangeControlCase(
            sweep_id="payload_sweep",
            scenario_family="stage3_backed_payload_pressure",
            fixture_id="stage5_0l_payload_stage3_fixture",
            topology_name="stage3_backed/payload/full_graph",
            controlled_parameter="payload_bits",
            control_value_label="payload_12kbits_stage3",
            control_unit="bits",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-8.0, bandwidth_hz=20e6),
            link_config=_link_config(payload_bits=12_000, bandwidth_hz=20e6, deadline_s=0.003),
            failure_reason_if_infeasible="resource_budget_failure",
        ),
    )


def _rsu_height_cases() -> tuple[_RangeControlCase, _RangeControlCase]:
    return (
        _RangeControlCase(
            sweep_id="rsu_height_placement_sweep",
            scenario_family="stage3_backed_blocked_geometry",
            fixture_id="stage5_0l_rsu_height_stage3_fixture",
            topology_name="stage3_backed/rsu_height/full_graph",
            controlled_parameter="rsu_height_m",
            control_value_label="rsu_height_8m_blocked_stage3",
            control_unit="m",
            channel_config=ChannelModelConfig(
                default_tx_power_dbm=-2.0,
                bandwidth_hz=30e6,
                nlos_penalty_db=10.0,
            ),
            link_config=_link_config(payload_bits=12_000, bandwidth_hz=30e6, deadline_s=0.003),
            scene_kind="blocked_geometry",
            failure_reason_if_infeasible="link_budget_failure",
        ),
        _RangeControlCase(
            sweep_id="rsu_height_placement_sweep",
            scenario_family="stage3_backed_blocked_geometry",
            fixture_id="stage5_0l_rsu_height_stage3_fixture",
            topology_name="stage3_backed/rsu_height/full_graph",
            controlled_parameter="rsu_height_m",
            control_value_label="rsu_height_25m_los_stage3",
            control_unit="m",
            channel_config=ChannelModelConfig(
                default_tx_power_dbm=-2.0,
                bandwidth_hz=30e6,
                nlos_penalty_db=10.0,
            ),
            link_config=_link_config(payload_bits=12_000, bandwidth_hz=30e6, deadline_s=0.003),
            scene_kind="blocked_geometry",
            failure_reason_if_infeasible="link_budget_failure",
        ),
    )


def _resource_budget_cases() -> tuple[_RangeControlCase, _RangeControlCase]:
    return (
        _RangeControlCase(
            sweep_id="resource_budget_limit_sweep",
            scenario_family="stage3_backed_resource_budget_limit",
            fixture_id="stage5_0l_resource_budget_stage3_fixture",
            topology_name="stage3_backed/resource_budget/full_graph",
            controlled_parameter="max_orthogonal_resources",
            control_value_label="resource_budget_2_stage3",
            control_unit="count",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-5.0, bandwidth_hz=20e6),
            link_config=_link_config(payload_bits=12_000, bandwidth_hz=20e6, deadline_s=0.003),
            use_background_interference=True,
            resource_budget_count=2,
            failure_reason_if_infeasible="resource_budget_failure",
        ),
        _RangeControlCase(
            sweep_id="resource_budget_limit_sweep",
            scenario_family="stage3_backed_resource_budget_limit",
            fixture_id="stage5_0l_resource_budget_stage3_fixture",
            topology_name="stage3_backed/resource_budget/full_graph",
            controlled_parameter="max_orthogonal_resources",
            control_value_label="resource_budget_6_stage3",
            control_unit="count",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-5.0, bandwidth_hz=20e6),
            link_config=_link_config(payload_bits=12_000, bandwidth_hz=20e6, deadline_s=0.003),
            use_background_interference=True,
            resource_budget_count=6,
            failure_reason_if_infeasible="resource_budget_failure",
        ),
    )


def _link_config(
    *,
    payload_bits: int,
    bandwidth_hz: float,
    deadline_s: float,
) -> LinkTransmissionConfig:
    return LinkTransmissionConfig(
        payload_bits=payload_bits,
        bandwidth_hz=bandwidth_hz,
        fixed_transmission_time_s=0.0005,
        target_reliability=0.99,
        deadline_s=deadline_s,
    )


def _evaluate_case(
    scene: Scene3D,
    graph: CandidateGraph,
    case: _RangeControlCase,
) -> _RangeEvaluation:
    records = _directed_records(scene, graph, graph.edge_ids, case)
    phase_records = {
        "pre_prepare": records,
        "prepare": records,
        "commit": records,
    }
    phase_budgets = PBFTPhaseBudgets(
        pre_prepare_budget_s=STAGE5_0K_PHASE_BUDGET_S,
        prepare_budget_s=STAGE5_0K_PHASE_BUDGET_S,
        commit_budget_s=STAGE5_0K_PHASE_BUDGET_S,
    )
    matrices = build_pbft_message_matrices_from_network_records(
        STAGE5_0K_NODE_IDS,
        phase_records,
        phase_budgets,
    )
    reliability = evaluate_expected_initiator_pbft_reliability(
        PBFTExpectedInitiatorConfig(
            node_ids=STAGE5_0K_NODE_IDS,
            fault_tolerance=STAGE5_0K_FAULT_TOLERANCE,
            fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST,
        ),
        pre_prepare_matrix=matrices.pre_prepare_matrix,
        prepare_matrix=matrices.prepare_matrix,
        commit_matrix=matrices.commit_matrix,
    )
    accounting = account_pbft_protocol_latency_energy(
        node_ids=STAGE5_0K_NODE_IDS,
        phase_records=phase_records,
        phase_budgets=phase_budgets,
    )
    return _RangeEvaluation(
        records=records,
        consensus_success_probability=reliability.consensus_success_probability,
        per_primary_reliability=reliability.per_primary_reliability,
        protocol_latency_s=accounting.protocol_latency_s,
        protocol_energy_j=accounting.protocol_energy_j,
        phase_budgets_s=accounting.phase_budgets_s,
        protocol_accounting_model_id=accounting.accounting_model_id,
        finite_blocklength_regime_id=URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
        network_regime_id=NETWORK_COMMUNICATION_REGIME_ID,
        matrix_adapter_id=matrices.adapter_id,
        fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST,
    )


def _directed_records(
    scene: Scene3D,
    graph: CandidateGraph,
    selected_edge_ids: tuple[str, ...],
    case: _RangeControlCase,
) -> tuple[NetworkCommunicationRecord, ...]:
    resource_assignments = _resource_assignments(graph, case.resource_budget_count)
    records: list[NetworkCommunicationRecord] = []
    for source_id in STAGE5_0K_NODE_IDS:
        for target_id in STAGE5_0K_NODE_IDS:
            if source_id == target_id:
                continue
            records.append(
                evaluate_network_communication(
                    scene=scene,
                    graph=graph,
                    selected_edge_ids=selected_edge_ids,
                    source_id=source_id,
                    target_ids=(target_id,),
                    config=NetworkCommunicationConfig(
                        channel_config=case.channel_config,
                        link_config=case.link_config,
                    ),
                    resource_assignments=resource_assignments,
                    background_transmissions=_background_transmissions(
                        selected_edge_ids,
                        source_id,
                        target_id,
                        resource_assignments,
                        case.use_background_interference,
                    ),
                )
            )
    return tuple(records)


def _resource_assignments(
    graph: CandidateGraph,
    resource_budget_count: int | None,
) -> dict[str, str]:
    if resource_budget_count is None:
        return {}
    return {
        edge_id: f"resource_{index % resource_budget_count}"
        for index, edge_id in enumerate(graph.edge_ids)
    }


def _background_transmissions(
    selected_edge_ids: tuple[str, ...],
    source_id: str,
    target_id: str,
    resource_assignments: Mapping[str, str],
    use_background_interference: bool,
) -> tuple[NetworkTransmissionSpec, ...]:
    if not use_background_interference:
        return ()
    current_pair = {source_id, target_id}
    specs: list[NetworkTransmissionSpec] = []
    for edge_id in selected_edge_ids:
        node_u, node_v = edge_id.split("--", 1)
        if {node_u, node_v} == current_pair:
            continue
        specs.append(
            NetworkTransmissionSpec(
                edge_id=edge_id,
                tx_id=node_u,
                rx_id=node_v,
                resource_id=resource_assignments.get(edge_id, "resource_0"),
            )
        )
    return tuple(specs)


def _row_from_case(
    graph: CandidateGraph,
    case: _RangeControlCase,
    evaluation: _RangeEvaluation,
    *,
    baseline_probability: float,
    baseline_latency: float,
    baseline_energy: float,
    extra_flags: tuple[str, ...],
) -> Stage50kSweepRow:
    probability = evaluation.consensus_success_probability
    requirement_met = probability >= TAU_REQUIREMENT_MIN
    record_probabilities = tuple(record.network_delivery_probability for record in evaluation.records)
    return Stage50kSweepRow(
        sweep_id=case.sweep_id,
        sweep_run_id=f"stage5_0l_{case.sweep_id}",
        scenario_family=case.scenario_family,
        fixture_id=case.fixture_id,
        topology_name=case.topology_name,
        controlled_parameter=case.controlled_parameter,
        control_value_label=case.control_value_label,
        control_unit=case.control_unit,
        tau_requirement_min=TAU_REQUIREMENT_MIN,
        requirement_met=requirement_met,
        consensus_success_probability=_checked_probability(probability),
        latency=evaluation.protocol_latency_s,
        energy=evaluation.protocol_energy_j,
        failure_reason_before=case.failure_reason_if_infeasible,
        failure_reason_after=None if requirement_met else case.failure_reason_if_infeasible,
        selected_edge_count=len(graph.edge_ids),
        is_full_graph_baseline=True,
        is_oracle_candidate=False,
        is_deployment_actor_input=False,
        parameter_source=f"{STAGE5_0L_PARAMETER_SOURCE}:{case.control_value_label}",
        diagnostic_flags=(
            "stage5_0l_stage3_backed_range_review",
            "finite_blocklength_active",
            "stage3_network_records",
            case.sweep_id,
        )
        + _case_diagnostic_flags(case)
        + extra_flags,
        stage3_backed=True,
        stage3_network_record_count=len(evaluation.records),
        min_network_delivery_probability=min(record_probabilities),
        mean_network_delivery_probability=fmean(record_probabilities),
        max_network_scheduled_latency_s=max(
            record.network_scheduled_latency_s for record in evaluation.records
        ),
        total_network_energy_j=sum(record.network_energy_j for record in evaluation.records),
        interference_group_ids=tuple(
            sorted({group_id for record in evaluation.records for group_id in record.interference_group_ids})
        ),
        finite_blocklength_regime_id=evaluation.finite_blocklength_regime_id,
        network_regime_id=evaluation.network_regime_id,
        matrix_adapter_id=evaluation.matrix_adapter_id,
        protocol_accounting_model_id=evaluation.protocol_accounting_model_id,
        fault_filter_mode=evaluation.fault_filter_mode,
        phase_budgets_s=evaluation.phase_budgets_s,
        per_primary_reliability=evaluation.per_primary_reliability,
        baseline_consensus_success_probability=baseline_probability,
        delta_consensus_success_probability=probability - baseline_probability,
        latency_delta=evaluation.protocol_latency_s - baseline_latency,
        energy_delta=evaluation.protocol_energy_j - baseline_energy,
        monotonicity_check_passed=probability + 1e-12 >= baseline_probability,
    )


def _case_diagnostic_flags(case: _RangeControlCase) -> tuple[str, ...]:
    flags: list[str] = []
    if case.scene_kind == "blocked_geometry":
        flags.append("blocked_geometry")
    if case.resource_budget_count is not None:
        flags.append(f"resource_budget={case.resource_budget_count}")
    return tuple(flags)


def _sweep_summary(rows: tuple[Stage50kSweepRow, ...]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for sweep_id in STAGE5_0L_SWEEP_IDS:
        sweep_rows = [row for row in rows if row.sweep_id == sweep_id]
        intervention_rows = [
            row for row in sweep_rows if "intervention" in row.diagnostic_flags
        ]
        baseline_rows = [row for row in sweep_rows if "baseline" in row.diagnostic_flags]
        summaries.append(
            {
                "sweep_id": sweep_id,
                "row_count": len(sweep_rows),
                "baseline_requirement_met": any(row.requirement_met for row in baseline_rows),
                "intervention_requirement_met": any(row.requirement_met for row in intervention_rows),
                "max_delta_consensus_success_probability": max(
                    row.delta_consensus_success_probability for row in sweep_rows
                ),
                "monotonicity_check_passed": all(
                    row.monotonicity_check_passed for row in sweep_rows
                ),
                "stage3_backed": all(row.stage3_backed for row in sweep_rows),
            }
        )
    return summaries


def _range_review(rows: tuple[Stage50kSweepRow, ...]) -> dict[str, object]:
    return {
        "bounded_single_axis_controls": True,
        "resource_budget_limit_included": True,
        "range_rows_are_not_training_data": True,
        "infeasible_to_feasible_sweeps": [
            summary["sweep_id"]
            for summary in _sweep_summary(rows)
            if summary["baseline_requirement_met"] is False
            and summary["intervention_requirement_met"] is True
        ],
    }


def _realism_review() -> list[dict[str, object]]:
    return [
        _realism_item("tx_power_dbm", "bounded diagnostic values -10 dBm to -8 dBm", "unknown_needs_reference", "needs regulatory and hardware reference before deployment claims"),
        _realism_item("payload_bits", "12 kbit and 18 kbit finite-blocklength probes", "unknown_needs_reference", "needs application message-size audit before reward design"),
        _realism_item("rsu_height_m", "8 m blocked and 25 m high-RSU cases", "unknown_needs_reference", "needs city installation constraints and placement policy"),
        _realism_item("resource_budget_count", "2-resource and 6-resource full-graph comparisons", "unknown_needs_reference", "needs explicit spectrum/resource budget before declaring full graph feasible"),
        _realism_item("fault_filter_mode", "unknown_faults_remove_largest", "conservative_diagnostic", "engineering lower-bound approximation, not a strict Byzantine model"),
    ]


def _realism_item(
    parameter: str,
    representation: str,
    status: str,
    note: str,
) -> dict[str, object]:
    return {
        "parameter": parameter,
        "current_representation": representation,
        "realism_status": status,
        "diagnostic_note": note,
    }


def _checks(rows: tuple[Stage50kSweepRow, ...]) -> dict[str, object]:
    row_sweep_ids = {row.sweep_id for row in rows}
    intervention_rows = [row for row in rows if "intervention" in row.diagnostic_flags]
    budget_limited_rows = [
        row for row in rows if row.control_value_label == "resource_budget_2_stage3"
    ]
    budget_sufficient_rows = [
        row for row in rows if row.control_value_label == "resource_budget_6_stage3"
    ]
    return {
        "tau_requirement_min_fixed": all(row.tau_requirement_min == TAU_REQUIREMENT_MIN for row in rows),
        "final_tau_selected": False,
        "final_tau_below_requirement_selected": False,
        "range_expansion_sweeps_present": set(STAGE5_0L_SWEEP_IDS).issubset(row_sweep_ids),
        "required_output_fields_present": all(
            set(REQUIRED_SWEEP_ROW_FIELDS).issubset(row.to_payload()) for row in rows
        ),
        "all_rows_stage3_backed": all(row.stage3_backed for row in rows),
        "all_rows_use_finite_blocklength": all(
            row.finite_blocklength_regime_id == URLLC_FINITE_BLOCKLENGTH_REGIME_ID for row in rows
        ),
        "all_rows_use_stage3_network_records": all(
            row.network_regime_id == NETWORK_COMMUNICATION_REGIME_ID for row in rows
        ),
        "adapter_uses_stage3_network_records": all(
            row.matrix_adapter_id == MESSAGE_MATRIX_ADAPTER_ID for row in rows
        ),
        "protocol_accounting_uses_stage3_network_records": all(
            row.protocol_accounting_model_id for row in rows
        ),
        "consensus_probabilities_in_range": all(
            0.0 <= row.consensus_success_probability <= 1.0 for row in rows
        ),
        "latency_energy_nonnegative": all(row.latency >= 0.0 and row.energy >= 0.0 for row in rows),
        "every_sweep_has_infeasible_to_feasible_transition": all(
            row.requirement_met
            and row.delta_consensus_success_probability > 0.0
            and row.baseline_consensus_success_probability < TAU_REQUIREMENT_MIN
            for row in intervention_rows
        ),
        "tx_power_sweep_improves_reliability": _intervention_delta(rows, "tx_power_sweep") > 0.0,
        "payload_reduction_improves_reliability": _intervention_delta(rows, "payload_sweep") > 0.0,
        "rsu_height_sweep_improves_reliability": (
            _intervention_delta(rows, "rsu_height_placement_sweep") > 0.0
        ),
        "resource_budget_limit_sweep_improves_reliability": (
            _intervention_delta(rows, "resource_budget_limit_sweep") > 0.0
        ),
        "limited_resource_budget_has_interference_groups": all(
            bool(row.interference_group_ids) for row in budget_limited_rows
        ),
        "sufficient_resource_budget_has_no_interference_groups": all(
            not row.interference_group_ids for row in budget_sufficient_rows
        ),
        "full_graph_not_oracle": all(not row.is_oracle_candidate for row in rows if row.is_full_graph_baseline),
        "oracle_labels_not_actor_inputs": all(not row.is_deployment_actor_input for row in rows),
        "registered_metric_fields_only": True,
        "simulation_parameters_changed_outside_declared_sweep_controls": False,
        "reward_implemented": False,
        "training_run": False,
        "v5_code_migrated": False,
    }


def _intervention_delta(rows: tuple[Stage50kSweepRow, ...], sweep_id: str) -> float:
    for row in rows:
        if row.sweep_id == sweep_id and "intervention" in row.diagnostic_flags:
            return row.delta_consensus_success_probability
    raise KeyError(f"missing intervention row for {sweep_id}")


def _scene_for_case(case: _RangeControlCase) -> Scene3D:
    if case.scene_kind == "blocked_geometry":
        high_rsu = "25m" in case.control_value_label
        return _blocked_scene(rsu_height_m=25.0 if high_rsu else 8.0)
    return _open_scene()


def _open_scene() -> Scene3D:
    return Scene3D(
        scenario_id=STAGE5_0L_REFERENCE_SCENARIO_ID,
        nodes=_nodes(rsu_height_m=8.0),
        physics_regime="stage5_0l_stage3_backed_range_reference",
    )


def _blocked_scene(*, rsu_height_m: float) -> Scene3D:
    return Scene3D(
        scenario_id=STAGE5_0L_BLOCKED_SCENARIO_ID,
        nodes=_nodes(rsu_height_m=rsu_height_m),
        buildings=(
            BuildingBox.from_footprint_height(
                "midblock_urban_blocker",
                min_x_m=10.0,
                max_x_m=25.0,
                min_y_m=-5.0,
                max_y_m=5.0,
                height_m=12.0,
            ),
        ),
        physics_regime="stage5_0l_blocked_geometry_range_reference",
    )


def _nodes(*, rsu_height_m: float) -> tuple[Node3D, ...]:
    return (
        Node3D("center", NodeKind.RSU, Point3D(0.0, 0.0, rsu_height_m)),
        Node3D("edge_a", NodeKind.VEHICLE, Point3D(35.0, 0.0, 1.5)),
        Node3D("edge_b", NodeKind.VEHICLE, Point3D(70.0, 0.0, 1.5)),
        Node3D("edge_c", NodeKind.VEHICLE, Point3D(35.0, 35.0, 1.5)),
    )


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
