"""Minimal topology evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Mapping

import networkx as nx

from marl_topology.link import LinkRecord
from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.objectives import aggregate_latency_energy
from marl_topology.protocol import ConsensusConfig, ConsensusResult
from marl_topology.topology.candidate_graph import CandidateGraph


@dataclass(frozen=True, slots=True)
class TopologyEvaluation:
    scenario_id: str
    topology_id: str
    selected_edge_ids: tuple[str, ...]
    metrics: Mapping[str, object]
    consensus: ConsensusResult
    diagnostics: Mapping[str, object]

    def metric_rows(self) -> tuple[dict[str, object], ...]:
        require_registered_metrics(self.metrics.keys())
        rows: list[dict[str, object]] = []
        for name, value in self.metrics.items():
            definition = REGISTERED_METRICS[name]
            rows.append(
                {
                    "scenario_id": self.scenario_id,
                    "topology_id": self.topology_id,
                    "metric_name": name,
                    "metric_level": definition.level,
                    "metric_value": value,
                    "used_for": definition.used_for,
                }
            )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class TopologyEvaluator:
    graph: CandidateGraph
    link_records: Mapping[str, LinkRecord]
    consensus_config: ConsensusConfig

    def __post_init__(self) -> None:
        missing = sorted(set(self.graph.edge_ids) - set(self.link_records))
        if missing:
            raise ValueError(f"missing link records for candidate edges: {missing}")
        if self.consensus_config.quorum_size > len(self.graph.node_ids):
            raise ValueError("quorum_size cannot exceed node count")

    def evaluate(
        self,
        selected_edge_ids: set[str] | frozenset[str] | tuple[str, ...],
        topology_id: str | None = None,
    ) -> TopologyEvaluation:
        selected = tuple(sorted(set(selected_edge_ids)))
        unknown = sorted(set(selected) - set(self.graph.edge_ids))
        if unknown:
            raise ValueError(f"selected unknown edge ids: {unknown}")

        selected_records = tuple(self.link_records[edge_id] for edge_id in selected)
        latency_energy = aggregate_latency_energy(selected_records)
        reachable_nodes, component_edges = self._leader_component(selected)
        probability = self._consensus_probability(reachable_nodes, component_edges)
        deadline_ok = (
            self.consensus_config.deadline_s is None
            or latency_energy.latency_s <= self.consensus_config.deadline_s
        )
        success = (
            reachable_nodes >= self.consensus_config.quorum_size
            and probability >= self.consensus_config.success_probability_threshold
            and deadline_ok
        )

        failure_reason = None
        if reachable_nodes < self.consensus_config.quorum_size:
            failure_reason = "quorum_unreachable"
        elif not deadline_ok:
            failure_reason = "deadline_exceeded"
        elif probability < self.consensus_config.success_probability_threshold:
            failure_reason = "probability_below_threshold"

        consensus = ConsensusResult(
            consensus_success=success,
            consensus_success_probability=probability,
            reachable_node_count=reachable_nodes,
            quorum_size=self.consensus_config.quorum_size,
            deadline_s=self.consensus_config.deadline_s,
            latency_s=latency_energy.latency_s,
            failure_reason=failure_reason,
        )
        diagnostics = {
            "edge_count": len(selected),
            "node_count": len(self.graph.node_ids),
            "reachable_node_count": reachable_nodes,
            "quorum_size": self.consensus_config.quorum_size,
            "is_full_graph_baseline": self.graph.is_full_selection(set(selected)),
            "physics_regime": self.graph.physics_regime,
            "protocol_variant": self.consensus_config.protocol_variant,
        }
        metrics = {
            "consensus_success": int(success),
            "consensus_success_probability": probability,
            "latency": latency_energy.latency_s,
            "energy": latency_energy.energy_j,
            "topology_diagnostics": diagnostics,
        }
        require_registered_metrics(metrics.keys())
        return TopologyEvaluation(
            scenario_id=self.graph.scenario_id,
            topology_id=topology_id or topology_id_for_edges(selected),
            selected_edge_ids=selected,
            metrics=metrics,
            consensus=consensus,
            diagnostics=diagnostics,
        )

    def _leader_component(self, selected_edge_ids: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
        if not self.graph.node_ids:
            return 0, ()
        leader = self.graph.node_ids[0]
        graph = nx.Graph()
        graph.add_nodes_from(self.graph.node_ids)
        for edge_id in selected_edge_ids:
            edge = self.graph.get_edge(edge_id)
            graph.add_edge(edge.node_u, edge.node_v, edge_id=edge.edge_id)
        if leader not in graph:
            return 0, ()
        component_nodes = set(nx.node_connected_component(graph, leader))
        component_edges = tuple(
            sorted(
                edge_id
                for edge_id in selected_edge_ids
                if self.graph.get_edge(edge_id).node_u in component_nodes
                and self.graph.get_edge(edge_id).node_v in component_nodes
            )
        )
        return len(component_nodes), component_edges

    def _consensus_probability(self, reachable_nodes: int, component_edges: tuple[str, ...]) -> float:
        if reachable_nodes < self.consensus_config.quorum_size:
            return 0.0
        if self.consensus_config.quorum_size <= 1 and not component_edges:
            return 1.0
        if not component_edges:
            return 0.0
        return min(self.link_records[edge_id].link_success_probability for edge_id in component_edges)


def topology_id_for_edges(edge_ids: tuple[str, ...]) -> str:
    if not edge_ids:
        return "topology:empty"
    digest = sha1("|".join(edge_ids).encode("utf-8")).hexdigest()[:12]
    return f"topology:{digest}"
