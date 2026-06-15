"""Stage 4.8 communication/consensus boundary audit."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping

from marl_topology.channel import ChannelModelConfig
from marl_topology.geometry3d import Point3D
from marl_topology.link import LinkTransmissionConfig
from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.network import (
    NetworkCommunicationConfig,
    NetworkCommunicationRecord,
    NetworkTransmissionSpec,
    evaluate_network_communication,
)
from marl_topology.protocol import (
    FAULT_FILTER_NONE,
    FAULT_FILTER_REMOVE_LARGEST,
    PBFTExpectedInitiatorConfig,
    evaluate_expected_initiator_pbft_reliability,
)
from marl_topology.scenario import Node3D, NodeKind, Scene3D
from marl_topology.topology import CandidateGraph, canonical_edge_id


STAGE4_8_BOUNDARY_AUDIT_STAGE_ID = "stage_4_8_communication_consensus_boundary_audit"
STAGE4_8_REFERENCE_SCENARIO_ID = "stage4_8_boundary_reference"
STAGE4_8_REQUIRED_CASES = (
    "near_threshold_link",
    "deadline_tight_retransmission",
    "unreachable_reliability_target",
    "weak_edge_primary",
    "center_vs_edge_primary",
    "interference_full_graph_penalty",
    "sparse_resource_efficient",
    "failed_scheduled_message",
)


@dataclass(frozen=True, slots=True)
class Stage48BoundaryAuditConfig:
    reliability_threshold: float = 0.2
    fault_tolerance: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability_threshold <= 1.0:
            raise ValueError("reliability_threshold must be in [0, 1]")
        if self.fault_tolerance < 0:
            raise ValueError("fault_tolerance must be nonnegative")


@dataclass(frozen=True, slots=True)
class Stage48BoundaryAuditRow:
    topology_name: str
    selected_edge_count: int
    link_deadline_delivery_probability: float
    network_deadline_delivery_probability: float
    per_primary_reliability: Mapping[str, float]
    consensus_success_probability: float
    scheduled_latency_s: float
    successful_delivery_latency_s: float
    energy_j: float
    reliability_feasible: bool
    diagnostic_flags: tuple[str, ...]
    is_full_graph_baseline: bool = False
    is_oracle_candidate: bool = False
    is_deployment_actor_input: bool = False

    def __post_init__(self) -> None:
        if self.topology_name not in STAGE4_8_REQUIRED_CASES:
            raise ValueError("topology_name must be a Stage 4.8 boundary case")
        if self.selected_edge_count < 0:
            raise ValueError("selected_edge_count must be nonnegative")
        for value in (
            self.link_deadline_delivery_probability,
            self.network_deadline_delivery_probability,
            self.consensus_success_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("probabilities must be in [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in self.per_primary_reliability.values()):
            raise ValueError("per-primary reliability values must be in [0, 1]")
        if self.scheduled_latency_s < 0.0 or self.successful_delivery_latency_s < 0.0:
            raise ValueError("latency values must be nonnegative")
        if self.successful_delivery_latency_s > self.scheduled_latency_s:
            raise ValueError("successful delivery latency cannot exceed scheduled latency")
        if self.energy_j < 0.0:
            raise ValueError("energy_j must be nonnegative")
        if self.is_oracle_candidate or self.is_deployment_actor_input:
            raise ValueError("Stage 4.8 rows must not expose oracle/actor labels")

    def metrics(self) -> dict[str, object]:
        result = {
            "consensus_success": int(self.reliability_feasible),
            "consensus_success_probability": self.consensus_success_probability,
            "latency": self.scheduled_latency_s,
            "energy": self.energy_j,
            "topology_diagnostics": {
                "topology_name": self.topology_name,
                "selected_edge_count": self.selected_edge_count,
                "link_deadline_delivery_probability": self.link_deadline_delivery_probability,
                "network_deadline_delivery_probability": (
                    self.network_deadline_delivery_probability
                ),
                "per_primary_reliability": dict(self.per_primary_reliability),
                "scheduled_latency_s": self.scheduled_latency_s,
                "successful_delivery_latency_s": self.successful_delivery_latency_s,
                "diagnostic_flags": list(self.diagnostic_flags),
                "is_full_graph_baseline": self.is_full_graph_baseline,
                "is_oracle_candidate": self.is_oracle_candidate,
                "is_deployment_actor_input": self.is_deployment_actor_input,
            },
        }
        require_registered_metrics(result.keys())
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "topology_name": self.topology_name,
            "selected_edge_count": self.selected_edge_count,
            "link_deadline_delivery_probability": self.link_deadline_delivery_probability,
            "network_deadline_delivery_probability": (
                self.network_deadline_delivery_probability
            ),
            "per_primary_reliability": dict(self.per_primary_reliability),
            "consensus_success_probability": self.consensus_success_probability,
            "scheduled_latency_s": self.scheduled_latency_s,
            "successful_delivery_latency_s": self.successful_delivery_latency_s,
            "energy_j": self.energy_j,
            "reliability_feasible": self.reliability_feasible,
            "diagnostic_flags": list(self.diagnostic_flags),
            "is_full_graph_baseline": self.is_full_graph_baseline,
            "is_oracle_candidate": self.is_oracle_candidate,
            "is_deployment_actor_input": self.is_deployment_actor_input,
            "metric_rows": list(_metric_rows(self.topology_name, self.metrics())),
        }


def build_stage4_8_boundary_audit_report(
    config: Stage48BoundaryAuditConfig | None = None,
) -> dict[str, object]:
    if config is None:
        config = Stage48BoundaryAuditConfig()
    scene = _reference_scene()
    graph = CandidateGraph.from_scene(scene)
    rows = (
        _near_threshold_link_row(scene, graph, config),
        _deadline_tight_retransmission_row(scene, graph, config),
        _unreachable_reliability_target_row(scene, graph, config),
        _weak_edge_primary_row(config),
        _center_vs_edge_primary_row(config),
        _interference_full_graph_penalty_row(scene, graph, config),
        _sparse_resource_efficient_row(scene, graph, config),
        _failed_scheduled_message_row(scene, graph, config),
    )
    metric_table = tuple(
        metric_row for row in rows for metric_row in _metric_rows(row.topology_name, row.metrics())
    )
    require_registered_metrics(row["metric_name"] for row in metric_table)
    return {
        "stage": STAGE4_8_BOUNDARY_AUDIT_STAGE_ID,
        "scenario": {
            "scenario_id": scene.scenario_id,
            "node_ids": list(graph.node_ids),
            "candidate_edge_count": len(graph.edge_ids),
            "physics_regime": scene.physics_regime,
        },
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_rows_are_registered": True,
            "new_metric_names_introduced": [],
        },
        "audit_rows": [row.to_payload() for row in rows],
        "metric_table": metric_table,
        "checks": {
            "required_cases_present": set(STAGE4_8_REQUIRED_CASES)
            == {row.topology_name for row in rows},
            "non_saturated_consensus_present": any(
                0.0 < row.consensus_success_probability < 1.0 for row in rows
            ),
            "failed_scheduled_message_has_latency_and_energy": _row_by_name(
                rows,
                "failed_scheduled_message",
            ).scheduled_latency_s
            > 0.0
            and _row_by_name(rows, "failed_scheduled_message").energy_j > 0.0,
            "inverse_reliability_cap_is_diagnostic": "required_transmission_time_capped"
            in _row_by_name(rows, "unreachable_reliability_target").diagnostic_flags,
            "weak_primary_distinguished": _weak_primary_is_distinguished(
                _row_by_name(rows, "weak_edge_primary").per_primary_reliability
            ),
            "center_primary_stronger_than_edge": _center_primary_is_stronger(
                _row_by_name(rows, "center_vs_edge_primary").per_primary_reliability
            ),
            "full_graph_is_baseline_not_oracle": (
                _row_by_name(rows, "interference_full_graph_penalty").is_full_graph_baseline
                is True
                and _row_by_name(rows, "interference_full_graph_penalty").is_oracle_candidate
                is False
            ),
            "oracle_candidate_is_not_actor_input": all(
                not row.is_deployment_actor_input for row in rows
            ),
            "training_run": False,
            "reward_implemented": False,
            "v5_code_migrated": False,
        },
    }


def _reference_scene() -> Scene3D:
    return Scene3D(
        scenario_id=STAGE4_8_REFERENCE_SCENARIO_ID,
        nodes=(
            Node3D("center", NodeKind.RSU, Point3D(0.0, 0.0, 8.0)),
            Node3D("edge_a", NodeKind.VEHICLE, Point3D(35.0, 0.0, 1.5)),
            Node3D("edge_b", NodeKind.VEHICLE, Point3D(70.0, 0.0, 1.5)),
            Node3D("edge_c", NodeKind.VEHICLE, Point3D(35.0, 35.0, 1.5)),
        ),
        physics_regime="stage4_8_boundary_urlcc_reference",
    )


def _near_threshold_link_row(
    scene: Scene3D,
    graph: CandidateGraph,
    config: Stage48BoundaryAuditConfig,
) -> Stage48BoundaryAuditRow:
    record = _route_record(
        scene,
        graph,
        tx_power_dbm=-12.0,
        selected_edge_ids=(canonical_edge_id("center", "edge_a"),),
    )
    return _network_probability_row(
        "near_threshold_link",
        record,
        selected_edge_count=1,
        config=config,
        flags=("near_threshold_link", "finite_blocklength_active"),
    )


def _deadline_tight_retransmission_row(
    scene: Scene3D,
    graph: CandidateGraph,
    config: Stage48BoundaryAuditConfig,
) -> Stage48BoundaryAuditRow:
    record = _route_record(
        scene,
        graph,
        tx_power_dbm=-12.0,
        selected_edge_ids=(canonical_edge_id("center", "edge_a"),),
        deadline_s=0.0007,
    )
    return _network_probability_row(
        "deadline_tight_retransmission",
        record,
        selected_edge_count=1,
        config=config,
        flags=("deadline_tight_retransmission", "limited_retransmission_attempts"),
    )


def _unreachable_reliability_target_row(
    scene: Scene3D,
    graph: CandidateGraph,
    config: Stage48BoundaryAuditConfig,
) -> Stage48BoundaryAuditRow:
    record = _route_record(
        scene,
        graph,
        tx_power_dbm=-120.0,
        selected_edge_ids=(canonical_edge_id("center", "edge_a"),),
        payload_bits=10_000_000,
        bandwidth_hz=1e3,
        target_reliability=0.999999,
        use_inverse_reliability=True,
        max_required_transmission_time_s=0.001,
        deadline_s=0.003,
    )
    first_hop = record.hop_records[0]
    flags = [
        "unreachable_reliability_target",
        "required_transmission_time_capped",
    ]
    if first_hop.p2p_delivery_probability == 0.0:
        flags.append("zero_deadline_delivery")
    return _network_probability_row(
        "unreachable_reliability_target",
        record,
        selected_edge_count=1,
        config=config,
        flags=tuple(flags),
    )


def _weak_edge_primary_row(config: Stage48BoundaryAuditConfig) -> Stage48BoundaryAuditRow:
    node_ids = ("center", "edge_a", "edge_b", "edge_c")
    matrices = _matrix_with_sender_penalty(node_ids, weak_sender="edge_a", weak_probability=0.6)
    reliability = _pbft_reliability(node_ids, matrices, config, fault_filter_mode=FAULT_FILTER_NONE)
    return Stage48BoundaryAuditRow(
        topology_name="weak_edge_primary",
        selected_edge_count=3,
        link_deadline_delivery_probability=_mean_matrix_probability(matrices),
        network_deadline_delivery_probability=_mean_matrix_probability(matrices),
        per_primary_reliability=reliability.per_primary_reliability,
        consensus_success_probability=reliability.consensus_success_probability,
        scheduled_latency_s=0.001,
        successful_delivery_latency_s=0.001,
        energy_j=0.0003,
        reliability_feasible=(
            reliability.consensus_success_probability >= config.reliability_threshold
        ),
        diagnostic_flags=("weak_edge_primary", "synthetic_pbft_matrix"),
    )


def _center_vs_edge_primary_row(config: Stage48BoundaryAuditConfig) -> Stage48BoundaryAuditRow:
    node_ids = ("center", "edge_a", "edge_b", "edge_c")
    matrices = _center_edge_matrix(node_ids)
    reliability = _pbft_reliability(node_ids, matrices, config, fault_filter_mode=FAULT_FILTER_NONE)
    return Stage48BoundaryAuditRow(
        topology_name="center_vs_edge_primary",
        selected_edge_count=3,
        link_deadline_delivery_probability=_mean_matrix_probability(matrices),
        network_deadline_delivery_probability=_mean_matrix_probability(matrices),
        per_primary_reliability=reliability.per_primary_reliability,
        consensus_success_probability=reliability.consensus_success_probability,
        scheduled_latency_s=0.001,
        successful_delivery_latency_s=0.001,
        energy_j=0.0003,
        reliability_feasible=(
            reliability.consensus_success_probability >= config.reliability_threshold
        ),
        diagnostic_flags=("center_vs_edge_primary", "synthetic_pbft_matrix"),
    )


def _interference_full_graph_penalty_row(
    scene: Scene3D,
    graph: CandidateGraph,
    config: Stage48BoundaryAuditConfig,
) -> Stage48BoundaryAuditRow:
    record = _route_record(
        scene,
        graph,
        tx_power_dbm=-10.0,
        selected_edge_ids=graph.edge_ids,
        background_transmissions=(
            NetworkTransmissionSpec(
                edge_id=canonical_edge_id("edge_b", "edge_c"),
                tx_id="edge_b",
                rx_id="edge_c",
            ),
        ),
    )
    return _network_probability_row(
        "interference_full_graph_penalty",
        record,
        selected_edge_count=len(graph.edge_ids),
        config=config,
        flags=("interference_full_graph_penalty", "same_resource_interference"),
        is_full_graph_baseline=True,
    )


def _sparse_resource_efficient_row(
    scene: Scene3D,
    graph: CandidateGraph,
    config: Stage48BoundaryAuditConfig,
) -> Stage48BoundaryAuditRow:
    record = _route_record(
        scene,
        graph,
        tx_power_dbm=-10.0,
        selected_edge_ids=(canonical_edge_id("center", "edge_a"),),
    )
    return _network_probability_row(
        "sparse_resource_efficient",
        record,
        selected_edge_count=1,
        config=config,
        flags=("sparse_resource_efficient", "no_background_interference"),
    )


def _failed_scheduled_message_row(
    scene: Scene3D,
    graph: CandidateGraph,
    config: Stage48BoundaryAuditConfig,
) -> Stage48BoundaryAuditRow:
    record = _route_record(
        scene,
        graph,
        tx_power_dbm=-120.0,
        selected_edge_ids=(canonical_edge_id("center", "edge_a"),),
    )
    return _network_probability_row(
        "failed_scheduled_message",
        record,
        selected_edge_count=1,
        config=config,
        flags=("failed_scheduled_message", "scheduled_latency_energy_visible"),
    )


def _route_record(
    scene: Scene3D,
    graph: CandidateGraph,
    *,
    tx_power_dbm: float,
    selected_edge_ids: tuple[str, ...],
    payload_bits: int = 12_000,
    bandwidth_hz: float | None = None,
    target_reliability: float | None = None,
    use_inverse_reliability: bool = False,
    max_required_transmission_time_s: float = 10.0,
    deadline_s: float = 0.003,
    background_transmissions: tuple[NetworkTransmissionSpec, ...] = (),
) -> NetworkCommunicationRecord:
    return evaluate_network_communication(
        scene=scene,
        graph=graph,
        selected_edge_ids=selected_edge_ids,
        source_id="center",
        target_ids=("edge_a",),
        config=NetworkCommunicationConfig(
            channel_config=ChannelModelConfig(default_tx_power_dbm=tx_power_dbm),
            link_config=LinkTransmissionConfig(
                payload_bits=payload_bits,
                bandwidth_hz=bandwidth_hz,
                fixed_transmission_time_s=0.0005 if not use_inverse_reliability else None,
                target_reliability=target_reliability,
                use_inverse_reliability=use_inverse_reliability,
                deadline_s=deadline_s,
                max_required_transmission_time_s=max_required_transmission_time_s,
            ),
        ),
        background_transmissions=background_transmissions,
    )


def _network_probability_row(
    topology_name: str,
    record: NetworkCommunicationRecord,
    *,
    selected_edge_count: int,
    config: Stage48BoundaryAuditConfig,
    flags: tuple[str, ...],
    is_full_graph_baseline: bool = False,
) -> Stage48BoundaryAuditRow:
    p = record.network_delivery_probability
    matrices = _uniform_matrix(("center", "edge_a", "edge_b", "edge_c"), p)
    reliability = _pbft_reliability(
        ("center", "edge_a", "edge_b", "edge_c"),
        matrices,
        config,
        fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST,
    )
    link_probability = 0.0
    if record.hop_records:
        link_probability = fmean(hop.p2p_delivery_probability for hop in record.hop_records)
    return Stage48BoundaryAuditRow(
        topology_name=topology_name,
        selected_edge_count=selected_edge_count,
        link_deadline_delivery_probability=link_probability,
        network_deadline_delivery_probability=record.network_delivery_probability,
        per_primary_reliability=reliability.per_primary_reliability,
        consensus_success_probability=reliability.consensus_success_probability,
        scheduled_latency_s=record.network_scheduled_latency_s,
        successful_delivery_latency_s=record.network_successful_delivery_latency_s,
        energy_j=record.network_energy_j,
        reliability_feasible=(
            reliability.consensus_success_probability >= config.reliability_threshold
        ),
        diagnostic_flags=flags,
        is_full_graph_baseline=is_full_graph_baseline,
    )


def _pbft_reliability(
    node_ids: tuple[str, ...],
    matrices: Mapping[tuple[str, str], float],
    config: Stage48BoundaryAuditConfig,
    *,
    fault_filter_mode: str,
):
    return evaluate_expected_initiator_pbft_reliability(
        PBFTExpectedInitiatorConfig(
            node_ids=node_ids,
            fault_tolerance=config.fault_tolerance,
            fault_filter_mode=fault_filter_mode,
        ),
        pre_prepare_matrix=matrices,
        prepare_matrix=matrices,
        commit_matrix=matrices,
    )


def _uniform_matrix(node_ids: tuple[str, ...], probability: float) -> dict[tuple[str, str], float]:
    return {
        (source_id, target_id): probability
        for source_id in node_ids
        for target_id in node_ids
        if source_id != target_id
    }


def _matrix_with_sender_penalty(
    node_ids: tuple[str, ...],
    *,
    weak_sender: str,
    weak_probability: float,
) -> dict[tuple[str, str], float]:
    return {
        (source_id, target_id): weak_probability if source_id == weak_sender else 0.9
        for source_id in node_ids
        for target_id in node_ids
        if source_id != target_id
    }


def _center_edge_matrix(node_ids: tuple[str, ...]) -> dict[tuple[str, str], float]:
    return {
        (source_id, target_id): (
            0.95 if source_id == "center" or target_id == "center" else 0.55
        )
        for source_id in node_ids
        for target_id in node_ids
        if source_id != target_id
    }


def _mean_matrix_probability(matrix: Mapping[tuple[str, str], float]) -> float:
    return fmean(matrix.values())


def _metric_rows(
    topology_name: str,
    metrics: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    require_registered_metrics(metrics.keys())
    rows: list[Mapping[str, object]] = []
    for name, value in metrics.items():
        definition = REGISTERED_METRICS[name]
        rows.append(
            {
                "scenario_id": STAGE4_8_REFERENCE_SCENARIO_ID,
                "topology_id": topology_name,
                "metric_name": name,
                "metric_level": definition.level,
                "metric_value": value,
                "used_for": definition.used_for,
            }
        )
    return tuple(rows)


def _row_by_name(
    rows: tuple[Stage48BoundaryAuditRow, ...],
    name: str,
) -> Stage48BoundaryAuditRow:
    for row in rows:
        if row.topology_name == name:
            return row
    raise KeyError(f"missing Stage 4.8 row: {name}")


def _weak_primary_is_distinguished(per_primary: Mapping[str, float]) -> bool:
    weak_value = per_primary["edge_a"]
    return weak_value < per_primary["center"] and weak_value < per_primary["edge_b"]


def _center_primary_is_stronger(per_primary: Mapping[str, float]) -> bool:
    center_value = per_primary["center"]
    return all(center_value > value for node_id, value in per_primary.items() if node_id != "center")
