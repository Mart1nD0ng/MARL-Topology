"""Stage 4.5 baseline and oracle-candidate review over PBFT reliability."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

from marl_topology.channel import ChannelModelConfig
from marl_topology.geometry3d import Point3D
from marl_topology.link import LinkTransmissionConfig
from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.network import (
    NetworkCommunicationConfig,
    NetworkCommunicationRecord,
    evaluate_network_communication,
)
from marl_topology.protocol import (
    FAULT_FILTER_REMOVE_LARGEST,
    PBFTExpectedInitiatorConfig,
    PBFTExpectedInitiatorReliabilityRecord,
    PBFTMessageMatrices,
    PBFTPhaseBudgets,
    account_pbft_protocol_latency_energy,
    build_pbft_message_matrices_from_network_records,
    evaluate_expected_initiator_pbft_reliability,
)
from marl_topology.scenario import Node3D, NodeKind, Scene3D
from marl_topology.topology import CandidateGraph, canonical_edge_id


STAGE4_5_REVIEW_STAGE_ID = "stage_4_5_baseline_and_oracle_review"
STAGE4_5_REFERENCE_SCENARIO_ID = "stage4_5_pbft_reference"


@dataclass(frozen=True, slots=True)
class Stage45ReviewConfig:
    reliability_threshold: float = 0.95
    fault_tolerance: int = 1
    max_oracle_candidate_edges: int = 10
    phase_budgets: PBFTPhaseBudgets = PBFTPhaseBudgets(
        pre_prepare_budget_s=1.0,
        prepare_budget_s=1.0,
        commit_budget_s=1.0,
    )
    channel_config: ChannelModelConfig = ChannelModelConfig(default_tx_power_dbm=0.0)
    link_config: LinkTransmissionConfig = LinkTransmissionConfig(
        payload_bits=12_000,
        fixed_transmission_time_s=0.0005,
        deadline_s=0.003,
        target_reliability=None,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability_threshold <= 1.0:
            raise ValueError("reliability_threshold must be in [0, 1]")
        if self.fault_tolerance < 0:
            raise ValueError("fault_tolerance must be nonnegative")
        if self.max_oracle_candidate_edges < 0:
            raise ValueError("max_oracle_candidate_edges must be nonnegative")


@dataclass(frozen=True, slots=True)
class Stage45TopologyReviewRow:
    name: str
    family: str
    selected_edge_ids: tuple[str, ...]
    is_full_graph_baseline: bool
    is_oracle_candidate: bool
    is_deployment_actor_input: bool
    reliability_feasible: bool
    metrics: Mapping[str, object]
    metric_rows: tuple[Mapping[str, object], ...]
    diagnostics: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "selected_edge_ids": list(self.selected_edge_ids),
            "is_full_graph_baseline": self.is_full_graph_baseline,
            "is_oracle_candidate": self.is_oracle_candidate,
            "is_deployment_actor_input": self.is_deployment_actor_input,
            "reliability_feasible": self.reliability_feasible,
            "metrics": dict(self.metrics),
            "metric_rows": [dict(row) for row in self.metric_rows],
            "diagnostics": dict(self.diagnostics),
        }


def build_stage4_5_baseline_oracle_review(
    config: Stage45ReviewConfig | None = None,
) -> dict[str, object]:
    """Build a deterministic Stage 4.5 baseline and oracle-candidate review."""

    if config is None:
        config = Stage45ReviewConfig()
    scene = _stage4_5_reference_scene()
    graph = CandidateGraph.from_scene(scene)
    node_ids = graph.node_ids
    pbft_config = PBFTExpectedInitiatorConfig(
        node_ids=node_ids,
        fault_tolerance=config.fault_tolerance,
        fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST,
    )
    network_config = NetworkCommunicationConfig(
        channel_config=config.channel_config,
        link_config=config.link_config,
    )

    baseline_specs = _baseline_topology_specs(graph)
    baseline_rows = tuple(
        _evaluate_topology(
            scene=scene,
            graph=graph,
            selected_edge_ids=edge_ids,
            name=name,
            family="baseline",
            is_oracle_candidate=False,
            pbft_config=pbft_config,
            phase_budgets=config.phase_budgets,
            network_config=network_config,
            reliability_threshold=config.reliability_threshold,
        )
        for name, edge_ids in baseline_specs
    )
    oracle_candidate = _find_oracle_candidate(
        scene=scene,
        graph=graph,
        pbft_config=pbft_config,
        phase_budgets=config.phase_budgets,
        network_config=network_config,
        reliability_threshold=config.reliability_threshold,
        max_oracle_candidate_edges=config.max_oracle_candidate_edges,
    )
    rows = baseline_rows + ((oracle_candidate["row"],) if oracle_candidate["row"] else ())
    payload_rows = [row.to_payload() for row in rows]

    return {
        "stage": STAGE4_5_REVIEW_STAGE_ID,
        "scenario": {
            "scenario_id": scene.scenario_id,
            "node_ids": list(node_ids),
            "candidate_edge_ids": list(graph.edge_ids),
            "candidate_edge_count": len(graph.edge_ids),
            "physics_regime": scene.physics_regime,
        },
        "protocol": {
            "model_id": pbft_config.model_id,
            "fault_tolerance": pbft_config.fault_tolerance,
            "fault_filter_mode": pbft_config.fault_filter_mode,
            "primary_distribution": pbft_config.primary_distribution,
            "view_change_mode": pbft_config.view_change_mode,
            "reliability_threshold": config.reliability_threshold,
            "phase_budgets_s": {
                "pre_prepare": config.phase_budgets.pre_prepare_budget_s,
                "prepare": config.phase_budgets.prepare_budget_s,
                "commit": config.phase_budgets.commit_budget_s,
            },
        },
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_rows_are_registered": all(
                set(row.metrics) <= set(REGISTERED_METRICS) for row in rows
            ),
            "new_metric_names_introduced": [],
        },
        "topology_rows": payload_rows,
        "oracle_candidate": _oracle_candidate_payload(oracle_candidate),
        "checks": {
            "empty_baseline_fails_but_does_not_prove_infeasible": (
                _row_by_name(rows, "empty").reliability_feasible is False
                and oracle_candidate["status"] == "feasible"
            ),
            "full_graph_is_baseline_not_oracle": (
                _row_by_name(rows, "full").is_full_graph_baseline is True
                and _row_by_name(rows, "full").is_oracle_candidate is False
            ),
            "oracle_candidate_is_not_actor_input": (
                oracle_candidate["row"] is None
                or oracle_candidate["row"].is_deployment_actor_input is False
            ),
            "oracle_candidate_is_not_full_graph_baseline": (
                oracle_candidate["row"] is not None
                and oracle_candidate["row"].is_full_graph_baseline is False
            ),
            "reliability_constraint_separate_from_latency_energy": all(
                row.reliability_feasible
                == (
                    float(row.metrics["consensus_success_probability"])
                    >= config.reliability_threshold
                )
                for row in rows
            ),
            "training_run": False,
            "v5_code_migrated": False,
            "reward_implemented": False,
        },
    }


def _stage4_5_reference_scene() -> Scene3D:
    return Scene3D(
        scenario_id=STAGE4_5_REFERENCE_SCENARIO_ID,
        nodes=(
            Node3D("n0", NodeKind.RSU, Point3D(0.0, 0.0, 8.0)),
            Node3D("n1", NodeKind.VEHICLE, Point3D(35.0, 0.0, 1.5)),
            Node3D("n2", NodeKind.VEHICLE, Point3D(70.0, 0.0, 1.5)),
            Node3D("n3", NodeKind.VEHICLE, Point3D(35.0, 35.0, 1.5)),
        ),
        physics_regime="stage4_5_urlcc_pbft_reference",
    )


def _baseline_topology_specs(
    graph: CandidateGraph,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("empty", ()),
        ("sparse_star", _edge_ids(("n0", "n1"), ("n0", "n2"), ("n0", "n3"))),
        ("sparse_chain", _edge_ids(("n0", "n1"), ("n1", "n2"), ("n2", "n3"))),
        ("full", graph.edge_ids),
    )


def _edge_ids(*pairs: tuple[str, str]) -> tuple[str, ...]:
    return tuple(sorted(canonical_edge_id(left, right) for left, right in pairs))


def _evaluate_topology(
    *,
    scene: Scene3D,
    graph: CandidateGraph,
    selected_edge_ids: tuple[str, ...],
    name: str,
    family: str,
    is_oracle_candidate: bool,
    pbft_config: PBFTExpectedInitiatorConfig,
    phase_budgets: PBFTPhaseBudgets,
    network_config: NetworkCommunicationConfig,
    reliability_threshold: float,
) -> Stage45TopologyReviewRow:
    selected = tuple(sorted(set(selected_edge_ids)))
    records = _directed_message_records(scene, graph, selected, network_config)
    phase_records = {phase_name: records for phase_name in ("pre_prepare", "prepare", "commit")}
    matrices = build_pbft_message_matrices_from_network_records(
        graph.node_ids,
        phase_records,
        phase_budgets,
    )
    reliability = evaluate_expected_initiator_pbft_reliability(
        pbft_config,
        pre_prepare_matrix=matrices.pre_prepare_matrix,
        prepare_matrix=matrices.prepare_matrix,
        commit_matrix=matrices.commit_matrix,
    )
    accounting = account_pbft_protocol_latency_energy(
        node_ids=graph.node_ids,
        phase_records=phase_records,
        phase_budgets=phase_budgets,
    )
    latency = accounting.protocol_latency_s
    energy = accounting.protocol_energy_j
    reliability_feasible = reliability.consensus_success_probability >= reliability_threshold
    diagnostics = _diagnostics(
        graph=graph,
        selected=selected,
        matrices=matrices,
        reliability=reliability,
        network_records=records,
        reliability_threshold=reliability_threshold,
        protocol_accounting=accounting.diagnostics(),
    )
    metrics = {
        "consensus_success": int(reliability_feasible),
        "consensus_success_probability": reliability.consensus_success_probability,
        "latency": latency,
        "energy": energy,
        "topology_diagnostics": diagnostics,
    }
    require_registered_metrics(metrics.keys())
    return Stage45TopologyReviewRow(
        name=name,
        family=family,
        selected_edge_ids=selected,
        is_full_graph_baseline=graph.is_full_selection(set(selected)),
        is_oracle_candidate=is_oracle_candidate,
        is_deployment_actor_input=False,
        reliability_feasible=reliability_feasible,
        metrics=metrics,
        metric_rows=_metric_rows(scene.scenario_id, name, metrics),
        diagnostics=diagnostics,
    )


def _directed_message_records(
    scene: Scene3D,
    graph: CandidateGraph,
    selected_edge_ids: tuple[str, ...],
    network_config: NetworkCommunicationConfig,
) -> tuple[NetworkCommunicationRecord, ...]:
    return tuple(
        evaluate_network_communication(
            scene=scene,
            graph=graph,
            selected_edge_ids=selected_edge_ids,
            source_id=source_id,
            target_ids=(target_id,),
            config=network_config,
        )
        for source_id in graph.node_ids
        for target_id in graph.node_ids
        if source_id != target_id
    )


def _find_oracle_candidate(
    *,
    scene: Scene3D,
    graph: CandidateGraph,
    pbft_config: PBFTExpectedInitiatorConfig,
    phase_budgets: PBFTPhaseBudgets,
    network_config: NetworkCommunicationConfig,
    reliability_threshold: float,
    max_oracle_candidate_edges: int,
) -> dict[str, object]:
    if len(graph.edge_ids) > max_oracle_candidate_edges:
        return {
            "status": "unresolved",
            "reason": "candidate_edge_count_exceeds_search_limit",
            "searched_topology_count": 0,
            "is_exhaustive": False,
            "row": None,
        }

    searched = 0
    feasible: list[Stage45TopologyReviewRow] = []
    for size in range(len(graph.edge_ids) + 1):
        for selected in combinations(graph.edge_ids, size):
            searched += 1
            row = _evaluate_topology(
                scene=scene,
                graph=graph,
                selected_edge_ids=tuple(selected),
                name=f"oracle_candidate:{searched}",
                family="oracle_candidate_review",
                is_oracle_candidate=True,
                pbft_config=pbft_config,
                phase_budgets=phase_budgets,
                network_config=network_config,
                reliability_threshold=reliability_threshold,
            )
            if row.reliability_feasible and not row.is_full_graph_baseline:
                feasible.append(row)
        if feasible:
            break

    if not feasible:
        return {
            "status": "infeasible",
            "reason": "no_non_full_feasible_candidate_found",
            "searched_topology_count": searched,
            "is_exhaustive": True,
            "row": None,
        }

    best = min(
        feasible,
        key=lambda row: (
            len(row.selected_edge_ids),
            float(row.metrics["latency"]),
            float(row.metrics["energy"]),
            row.selected_edge_ids,
        ),
    )
    return {
        "status": "feasible",
        "reason": "non_full_reliability_feasible_candidate_found",
        "searched_topology_count": searched,
        "is_exhaustive": True,
        "row": best,
    }


def _diagnostics(
    *,
    graph: CandidateGraph,
    selected: tuple[str, ...],
    matrices: PBFTMessageMatrices,
    reliability: PBFTExpectedInitiatorReliabilityRecord,
    network_records: tuple[NetworkCommunicationRecord, ...],
    reliability_threshold: float,
    protocol_accounting: Mapping[str, object],
) -> dict[str, object]:
    return {
        "edge_count": len(selected),
        "node_count": len(graph.node_ids),
        "candidate_edge_count": len(graph.edge_ids),
        "is_full_graph_baseline": graph.is_full_selection(set(selected)),
        "reliability_threshold": reliability_threshold,
        "per_primary_reliability": dict(reliability.per_primary_reliability),
        "primary_distribution": dict(reliability.primary_distribution),
        "phase_budgets_s": dict(matrices.phase_budgets_s),
        "record_count_by_phase": dict(matrices.record_count_by_phase),
        "deadline_filtered_count_by_phase": dict(matrices.deadline_filtered_count_by_phase),
        "zero_delivery_count_by_phase": dict(matrices.zero_delivery_count_by_phase),
        "network_record_count": len(network_records),
        "network_records_have_oracle_label": any(record.is_oracle for record in network_records),
        "latency_aggregation": "stage4_6_sum_phase_max_clipped_to_budget",
        "energy_aggregation": "stage4_6_sum_scheduled_attempt_energy",
        "protocol_accounting": dict(protocol_accounting),
    }


def _metric_rows(
    scenario_id: str,
    topology_id: str,
    metrics: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    require_registered_metrics(metrics.keys())
    rows: list[Mapping[str, object]] = []
    for name, value in metrics.items():
        definition = REGISTERED_METRICS[name]
        rows.append(
            {
                "scenario_id": scenario_id,
                "topology_id": topology_id,
                "metric_name": name,
                "metric_level": definition.level,
                "metric_value": value,
                "used_for": definition.used_for,
            }
        )
    return tuple(rows)


def _oracle_candidate_payload(oracle_candidate: Mapping[str, object]) -> dict[str, object]:
    row = oracle_candidate["row"]
    return {
        "status": oracle_candidate["status"],
        "reason": oracle_candidate["reason"],
        "searched_topology_count": oracle_candidate["searched_topology_count"],
        "is_exhaustive": oracle_candidate["is_exhaustive"],
        "is_deployment_actor_input": False,
        "row": None if row is None else row.to_payload(),
    }


def _row_by_name(
    rows: tuple[Stage45TopologyReviewRow, ...],
    name: str,
) -> Stage45TopologyReviewRow:
    for row in rows:
        if row.name == name:
            return row
    raise KeyError(f"missing topology review row: {name}")
