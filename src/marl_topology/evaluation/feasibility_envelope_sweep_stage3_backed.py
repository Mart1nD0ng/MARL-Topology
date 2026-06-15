"""Stage 5.0k Stage 3-backed feasibility envelope sweep hardening."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean
from typing import Mapping

from marl_topology.channel import ChannelModelConfig
from marl_topology.geometry3d import Point3D
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

from .feasibility_envelope_sweep import STAGE5_0J_MINIMAL_SWEEP_STAGE_ID
from .feasibility_envelope_sweep_design import (
    REGISTERED_SWEEP_METRIC_FIELDS,
    REQUIRED_SWEEP_ROW_FIELDS,
    STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
    build_stage5_0i_feasibility_envelope_sweep_design,
)
from .requirement_feasibility_diagnosis import (
    SUPPORTED_FAILURE_REASONS,
    TAU_REQUIREMENT_MIN,
)


STAGE5_0K_STAGE3_BACKED_SWEEP_STAGE_ID = (
    "stage_5_0k_stage3_backed_feasibility_envelope_sweep_hardening"
)
STAGE5_0K_STAGE3_BACKED_SWEEP_IDS = (
    "bandwidth_sweep",
    "deadline_sweep",
    "resource_orthogonalization_sweep",
)
STAGE5_0K_REFERENCE_SCENARIO_ID = "stage5_0k_stage3_backed_reference"
STAGE5_0K_NODE_IDS = ("center", "edge_a", "edge_b", "edge_c")
STAGE5_0K_FAULT_TOLERANCE = 1
STAGE5_0K_PHASE_BUDGET_S = 0.01
STAGE5_0K_PARAMETER_SOURCE = "stage5_0k_stage3_backed_control"


@dataclass(frozen=True, slots=True)
class Stage50kSweepRow:
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
    stage3_backed: bool
    stage3_network_record_count: int
    min_network_delivery_probability: float
    mean_network_delivery_probability: float
    max_network_scheduled_latency_s: float
    total_network_energy_j: float
    interference_group_ids: tuple[str, ...]
    finite_blocklength_regime_id: str
    network_regime_id: str
    matrix_adapter_id: str
    protocol_accounting_model_id: str
    fault_filter_mode: str
    phase_budgets_s: Mapping[str, float]
    per_primary_reliability: Mapping[str, float]
    baseline_consensus_success_probability: float
    delta_consensus_success_probability: float
    latency_delta: float
    energy_delta: float
    monotonicity_check_passed: bool

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
            "finite_blocklength_regime_id",
            "network_regime_id",
            "matrix_adapter_id",
            "protocol_accounting_model_id",
            "fault_filter_mode",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "tau_requirement_min",
            "consensus_success_probability",
            "latency",
            "energy",
            "min_network_delivery_probability",
            "mean_network_delivery_probability",
            "max_network_scheduled_latency_s",
            "total_network_energy_j",
            "baseline_consensus_success_probability",
            "delta_consensus_success_probability",
            "latency_delta",
            "energy_delta",
        ):
            value = getattr(self, field_name)
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.tau_requirement_min != TAU_REQUIREMENT_MIN:
            raise ValueError("Stage 5.0k rows must keep tau_requirement_min fixed")
        for field_name in (
            "consensus_success_probability",
            "min_network_delivery_probability",
            "mean_network_delivery_probability",
            "baseline_consensus_success_probability",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.latency < 0.0 or self.energy < 0.0:
            raise ValueError("latency and energy must be nonnegative")
        if self.max_network_scheduled_latency_s < 0.0 or self.total_network_energy_j < 0.0:
            raise ValueError("network latency and energy diagnostics must be nonnegative")
        if self.selected_edge_count < 0 or self.stage3_network_record_count <= 0:
            raise ValueError("edge and Stage 3 record counts must be valid")
        if self.failure_reason_before is not None and self.failure_reason_before not in SUPPORTED_FAILURE_REASONS:
            raise ValueError("unsupported failure_reason_before")
        if self.failure_reason_after is not None and self.failure_reason_after not in SUPPORTED_FAILURE_REASONS:
            raise ValueError("unsupported failure_reason_after")
        if self.is_full_graph_baseline and self.is_oracle_candidate:
            raise ValueError("full graph baseline must not be oracle")
        if self.is_deployment_actor_input:
            raise ValueError("sweep rows must not be deployment actor inputs")
        if not self.stage3_backed:
            raise ValueError("Stage 5.0k rows must be backed by Stage 3 records")
        if self.finite_blocklength_regime_id != URLLC_FINITE_BLOCKLENGTH_REGIME_ID:
            raise ValueError("Stage 5.0k must use finite-blocklength link records")
        if self.network_regime_id != NETWORK_COMMUNICATION_REGIME_ID:
            raise ValueError("Stage 5.0k must use Stage 3 network records")
        if self.matrix_adapter_id != MESSAGE_MATRIX_ADAPTER_ID:
            raise ValueError("Stage 5.0k must use the Stage 4.3 matrix adapter")
        if set(self.phase_budgets_s) != {"pre_prepare", "prepare", "commit"}:
            raise ValueError("phase_budgets_s must include PBFT phases")
        if set(self.per_primary_reliability) != set(STAGE5_0K_NODE_IDS):
            raise ValueError("per_primary_reliability must contain reference node ids")

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
            "stage3_backed": self.stage3_backed,
            "stage3_network_record_count": self.stage3_network_record_count,
            "min_network_delivery_probability": self.min_network_delivery_probability,
            "mean_network_delivery_probability": self.mean_network_delivery_probability,
            "max_network_scheduled_latency_s": self.max_network_scheduled_latency_s,
            "total_network_energy_j": self.total_network_energy_j,
            "interference_group_ids": list(self.interference_group_ids),
            "finite_blocklength_regime_id": self.finite_blocklength_regime_id,
            "network_regime_id": self.network_regime_id,
            "matrix_adapter_id": self.matrix_adapter_id,
            "protocol_accounting_model_id": self.protocol_accounting_model_id,
            "fault_filter_mode": self.fault_filter_mode,
            "phase_budgets_s": dict(self.phase_budgets_s),
            "per_primary_reliability": dict(self.per_primary_reliability),
            "baseline_consensus_success_probability": self.baseline_consensus_success_probability,
            "delta_consensus_success_probability": self.delta_consensus_success_probability,
            "latency_delta": self.latency_delta,
            "energy_delta": self.energy_delta,
            "monotonicity_check_passed": self.monotonicity_check_passed,
        }
        missing = sorted(set(REQUIRED_SWEEP_ROW_FIELDS) - set(payload))
        if missing:
            raise ValueError(f"Stage 5.0k sweep row missing required fields: {missing}")
        require_registered_metrics(REGISTERED_SWEEP_METRIC_FIELDS)
        return payload


@dataclass(frozen=True, slots=True)
class _Stage3ControlCase:
    sweep_id: str
    scenario_family: str
    fixture_id: str
    topology_name: str
    controlled_parameter: str
    control_value_label: str
    control_unit: str
    channel_config: ChannelModelConfig
    link_config: LinkTransmissionConfig
    use_background_interference: bool = False
    orthogonal_resources: bool = False
    failure_reason_if_infeasible: str = "unknown"


@dataclass(frozen=True, slots=True)
class _Stage3Evaluation:
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


def build_stage5_0k_stage3_backed_feasibility_envelope_sweep() -> dict[str, object]:
    """Run the minimal Stage 3-backed hardening sweep."""

    require_registered_metrics(REGISTERED_SWEEP_METRIC_FIELDS)
    design = build_stage5_0i_feasibility_envelope_sweep_design()
    scene = _reference_scene()
    graph = CandidateGraph.from_scene(scene)
    rows = _build_rows(scene, graph)
    return {
        "stage": STAGE5_0K_STAGE3_BACKED_SWEEP_STAGE_ID,
        "source_stage": STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
        "source_alpha_stage": STAGE5_0J_MINIMAL_SWEEP_STAGE_ID,
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "sweep_scope": "minimal_stage3_backed_hardening_subset",
        "executed_sweep_ids": list(STAGE5_0K_STAGE3_BACKED_SWEEP_IDS),
        "deferred_sweep_ids": [
            sweep_id
            for sweep_id in design["required_sweep_ids"]
            if sweep_id not in STAGE5_0K_STAGE3_BACKED_SWEEP_IDS
        ],
        "scenario": {
            "scenario_id": scene.scenario_id,
            "node_ids": list(graph.node_ids),
            "candidate_edge_count": len(graph.edge_ids),
            "physics_regime": scene.physics_regime,
        },
        "sweep_rows": [row.to_payload() for row in rows],
        "sweep_summary": _sweep_summary(rows),
        "comparison_policy": design["comparison_policy"],
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
        "checks": _checks(rows, design),
    }


def _build_rows(scene: Scene3D, graph: CandidateGraph) -> tuple[Stage50kSweepRow, ...]:
    row_pairs: list[tuple[Stage50kSweepRow, Stage50kSweepRow]] = []
    for baseline, intervention in (
        _bandwidth_cases(),
        _deadline_cases(),
        _resource_orthogonalization_cases(),
    ):
        baseline_eval = _evaluate_case(scene, graph, baseline)
        intervention_eval = _evaluate_case(scene, graph, intervention)
        baseline_probability = baseline_eval.consensus_success_probability
        row_pairs.append(
            (
                _row_from_case(
                    graph,
                    baseline,
                    baseline_eval,
                    baseline_probability=baseline_probability,
                    baseline_latency=baseline_eval.protocol_latency_s,
                    baseline_energy=baseline_eval.protocol_energy_j,
                    extra_flags=("baseline",),
                ),
                _row_from_case(
                    graph,
                    intervention,
                    intervention_eval,
                    baseline_probability=baseline_probability,
                    baseline_latency=baseline_eval.protocol_latency_s,
                    baseline_energy=baseline_eval.protocol_energy_j,
                    extra_flags=("intervention", "single_axis_control"),
                ),
            )
        )
    return tuple(row for pair in row_pairs for row in pair)


def _bandwidth_cases() -> tuple[_Stage3ControlCase, _Stage3ControlCase]:
    return (
        _Stage3ControlCase(
            sweep_id="bandwidth_sweep",
            scenario_family="stage3_backed_near_threshold_link_budget",
            fixture_id="stage5_0k_bandwidth_stage3_fixture",
            topology_name="stage3_backed/bandwidth/full_graph",
            controlled_parameter="bandwidth_hz",
            control_value_label="bandwidth_15mhz_stage3",
            control_unit="Hz",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-8.0, bandwidth_hz=15e6),
            link_config=LinkTransmissionConfig(
                payload_bits=12_000,
                bandwidth_hz=15e6,
                fixed_transmission_time_s=0.0005,
                target_reliability=0.99,
                deadline_s=0.003,
            ),
            failure_reason_if_infeasible="link_budget_failure",
        ),
        _Stage3ControlCase(
            sweep_id="bandwidth_sweep",
            scenario_family="stage3_backed_near_threshold_link_budget",
            fixture_id="stage5_0k_bandwidth_stage3_fixture",
            topology_name="stage3_backed/bandwidth/full_graph",
            controlled_parameter="bandwidth_hz",
            control_value_label="bandwidth_20mhz_stage3",
            control_unit="Hz",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-8.0, bandwidth_hz=20e6),
            link_config=LinkTransmissionConfig(
                payload_bits=12_000,
                bandwidth_hz=20e6,
                fixed_transmission_time_s=0.0005,
                target_reliability=0.99,
                deadline_s=0.003,
            ),
            failure_reason_if_infeasible="link_budget_failure",
        ),
    )


def _deadline_cases() -> tuple[_Stage3ControlCase, _Stage3ControlCase]:
    return (
        _Stage3ControlCase(
            sweep_id="deadline_sweep",
            scenario_family="stage3_backed_deadline_tight_retransmission",
            fixture_id="stage5_0k_deadline_stage3_fixture",
            topology_name="stage3_backed/deadline/full_graph",
            controlled_parameter="deadline_s",
            control_value_label="deadline_1ms_stage3",
            control_unit="s",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-8.0, bandwidth_hz=20e6),
            link_config=LinkTransmissionConfig(
                payload_bits=12_000,
                bandwidth_hz=20e6,
                fixed_transmission_time_s=0.0005,
                target_reliability=0.99,
                deadline_s=0.001,
            ),
            failure_reason_if_infeasible="deadline_failure",
        ),
        _Stage3ControlCase(
            sweep_id="deadline_sweep",
            scenario_family="stage3_backed_deadline_tight_retransmission",
            fixture_id="stage5_0k_deadline_stage3_fixture",
            topology_name="stage3_backed/deadline/full_graph",
            controlled_parameter="deadline_s",
            control_value_label="deadline_3ms_stage3",
            control_unit="s",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-8.0, bandwidth_hz=20e6),
            link_config=LinkTransmissionConfig(
                payload_bits=12_000,
                bandwidth_hz=20e6,
                fixed_transmission_time_s=0.0005,
                target_reliability=0.99,
                deadline_s=0.003,
            ),
            failure_reason_if_infeasible="retransmission_insufficient",
        ),
    )


def _resource_orthogonalization_cases() -> tuple[_Stage3ControlCase, _Stage3ControlCase]:
    return (
        _Stage3ControlCase(
            sweep_id="resource_orthogonalization_sweep",
            scenario_family="stage3_backed_same_resource_interference",
            fixture_id="stage5_0k_resource_stage3_fixture",
            topology_name="stage3_backed/resource/full_graph",
            controlled_parameter="channel_resource_assignment",
            control_value_label="shared_resource_stage3",
            control_unit="resource_label",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-5.0, bandwidth_hz=20e6),
            link_config=LinkTransmissionConfig(
                payload_bits=12_000,
                bandwidth_hz=20e6,
                fixed_transmission_time_s=0.0005,
                target_reliability=0.99,
                deadline_s=0.003,
            ),
            use_background_interference=True,
            orthogonal_resources=False,
            failure_reason_if_infeasible="interference_failure",
        ),
        _Stage3ControlCase(
            sweep_id="resource_orthogonalization_sweep",
            scenario_family="stage3_backed_same_resource_interference",
            fixture_id="stage5_0k_resource_stage3_fixture",
            topology_name="stage3_backed/resource/full_graph",
            controlled_parameter="channel_resource_assignment",
            control_value_label="orthogonal_resources_stage3",
            control_unit="resource_label",
            channel_config=ChannelModelConfig(default_tx_power_dbm=-5.0, bandwidth_hz=20e6),
            link_config=LinkTransmissionConfig(
                payload_bits=12_000,
                bandwidth_hz=20e6,
                fixed_transmission_time_s=0.0005,
                target_reliability=0.99,
                deadline_s=0.003,
            ),
            use_background_interference=True,
            orthogonal_resources=True,
            failure_reason_if_infeasible="interference_failure",
        ),
    )


def _evaluate_case(
    scene: Scene3D,
    graph: CandidateGraph,
    case: _Stage3ControlCase,
) -> _Stage3Evaluation:
    selected_edge_ids = graph.edge_ids
    records = _directed_records(scene, graph, selected_edge_ids, case)
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
    return _Stage3Evaluation(
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
    case: _Stage3ControlCase,
) -> tuple[NetworkCommunicationRecord, ...]:
    resource_assignments = _resource_assignments(graph, case.orthogonal_resources)
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
    orthogonal_resources: bool,
) -> dict[str, str]:
    if not orthogonal_resources:
        return {}
    return {
        edge_id: f"resource_{index}"
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
    specs: list[NetworkTransmissionSpec] = []
    current_pair = {source_id, target_id}
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
    case: _Stage3ControlCase,
    evaluation: _Stage3Evaluation,
    *,
    baseline_probability: float,
    baseline_latency: float,
    baseline_energy: float,
    extra_flags: tuple[str, ...],
) -> Stage50kSweepRow:
    probability = evaluation.consensus_success_probability
    requirement_met = probability >= TAU_REQUIREMENT_MIN
    record_probabilities = tuple(record.network_delivery_probability for record in evaluation.records)
    max_scheduled_latency = max(record.network_scheduled_latency_s for record in evaluation.records)
    total_network_energy = sum(record.network_energy_j for record in evaluation.records)
    interference_group_ids = tuple(
        sorted({group_id for record in evaluation.records for group_id in record.interference_group_ids})
    )
    return Stage50kSweepRow(
        sweep_id=case.sweep_id,
        sweep_run_id=f"stage5_0k_{case.sweep_id}",
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
        parameter_source=f"{STAGE5_0K_PARAMETER_SOURCE}:{case.control_value_label}",
        diagnostic_flags=(
            "stage5_0k_stage3_backed",
            "finite_blocklength_active",
            "stage3_network_records",
            case.sweep_id,
        )
        + extra_flags,
        stage3_backed=True,
        stage3_network_record_count=len(evaluation.records),
        min_network_delivery_probability=min(record_probabilities),
        mean_network_delivery_probability=fmean(record_probabilities),
        max_network_scheduled_latency_s=max_scheduled_latency,
        total_network_energy_j=total_network_energy,
        interference_group_ids=interference_group_ids,
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
        monotonicity_check_passed=_monotonicity_passed(case.sweep_id, probability, baseline_probability),
    )


def _sweep_summary(rows: tuple[Stage50kSweepRow, ...]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for sweep_id in STAGE5_0K_STAGE3_BACKED_SWEEP_IDS:
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


def _checks(
    rows: tuple[Stage50kSweepRow, ...],
    design: Mapping[str, object],
) -> dict[str, object]:
    intervention_rows = [
        row for row in rows if "intervention" in row.diagnostic_flags
    ]
    row_sweep_ids = {row.sweep_id for row in rows}
    shared_resource_rows = [
        row for row in rows if row.control_value_label == "shared_resource_stage3"
    ]
    orthogonal_resource_rows = [
        row for row in rows if row.control_value_label == "orthogonal_resources_stage3"
    ]
    return {
        "tau_requirement_min_fixed": all(row.tau_requirement_min == TAU_REQUIREMENT_MIN for row in rows),
        "final_tau_selected": False,
        "final_tau_below_requirement_selected": False,
        "stage3_backed_sweeps_present": set(STAGE5_0K_STAGE3_BACKED_SWEEP_IDS).issubset(row_sweep_ids),
        "deferred_sweeps_declared": bool(
            set(design["required_sweep_ids"]) - set(STAGE5_0K_STAGE3_BACKED_SWEEP_IDS)
        ),
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
        "at_least_one_stage3_backed_infeasible_to_feasible_transition": any(
            row.delta_consensus_success_probability > 0.0
            and row.requirement_met
            and row.baseline_consensus_success_probability < TAU_REQUIREMENT_MIN
            for row in intervention_rows
        ),
        "bandwidth_sweep_improves_reliability": _intervention_delta(rows, "bandwidth_sweep") > 0.0,
        "deadline_sweep_improves_reliability": _intervention_delta(rows, "deadline_sweep") > 0.0,
        "deadline_sweep_records_latency_cost": _intervention_latency_delta(rows, "deadline_sweep") >= 0.0,
        "resource_orthogonalization_improves_reliability": (
            _intervention_delta(rows, "resource_orthogonalization_sweep") > 0.0
        ),
        "shared_resource_has_interference_groups": all(
            bool(row.interference_group_ids) for row in shared_resource_rows
        ),
        "orthogonal_resource_has_no_interference_groups": all(
            not row.interference_group_ids for row in orthogonal_resource_rows
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


def _intervention_latency_delta(rows: tuple[Stage50kSweepRow, ...], sweep_id: str) -> float:
    for row in rows:
        if row.sweep_id == sweep_id and "intervention" in row.diagnostic_flags:
            return row.latency_delta
    raise KeyError(f"missing intervention row for {sweep_id}")


def _monotonicity_passed(
    sweep_id: str,
    probability: float,
    baseline_probability: float,
) -> bool:
    if sweep_id in STAGE5_0K_STAGE3_BACKED_SWEEP_IDS:
        return probability + 1e-12 >= baseline_probability
    return True


def _reference_scene() -> Scene3D:
    return Scene3D(
        scenario_id=STAGE5_0K_REFERENCE_SCENARIO_ID,
        nodes=(
            Node3D("center", NodeKind.RSU, Point3D(0.0, 0.0, 8.0)),
            Node3D("edge_a", NodeKind.VEHICLE, Point3D(35.0, 0.0, 1.5)),
            Node3D("edge_b", NodeKind.VEHICLE, Point3D(70.0, 0.0, 1.5)),
            Node3D("edge_c", NodeKind.VEHICLE, Point3D(35.0, 35.0, 1.5)),
        ),
        physics_regime="stage5_0k_stage3_backed_urlcc_reference",
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
