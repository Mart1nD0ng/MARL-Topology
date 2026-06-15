"""Simple non-learning topology baselines."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Mapping

import networkx as nx

from marl_topology.link import LinkRecord
from marl_topology.topology.candidate_graph import CandidateGraph


@dataclass(frozen=True, slots=True)
class BaselineTopology:
    name: str
    edge_ids: tuple[str, ...]
    is_oracle: bool = False


class PolicyBaselines:
    """Policy-free baselines for regression.

    These methods do not implement actors and do not use oracle labels.
    """

    @staticmethod
    def empty(graph: CandidateGraph) -> BaselineTopology:
        return BaselineTopology(name="empty", edge_ids=())

    @staticmethod
    def full(graph: CandidateGraph) -> BaselineTopology:
        return BaselineTopology(name="full", edge_ids=graph.edge_ids, is_oracle=False)

    @staticmethod
    def random(graph: CandidateGraph, seed: int, edge_probability: float = 0.5) -> BaselineTopology:
        if not 0.0 <= edge_probability <= 1.0:
            raise ValueError("edge_probability must be in [0, 1]")
        rng = Random(seed)
        selected = tuple(edge_id for edge_id in graph.edge_ids if rng.random() < edge_probability)
        return BaselineTopology(name=f"random_seed_{seed}", edge_ids=selected)

    @staticmethod
    def greedy_reliability(
        graph: CandidateGraph,
        link_records: Mapping[str, LinkRecord],
    ) -> BaselineTopology:
        weighted = nx.Graph()
        weighted.add_nodes_from(graph.node_ids)
        for edge in graph.edges:
            weighted.add_edge(
                edge.node_u,
                edge.node_v,
                edge_id=edge.edge_id,
                weight=link_records[edge.edge_id].link_success_probability,
            )
        selected: list[str] = []
        for component_nodes in nx.connected_components(weighted):
            subgraph = weighted.subgraph(component_nodes)
            tree = nx.maximum_spanning_tree(subgraph, weight="weight")
            selected.extend(data["edge_id"] for _, _, data in tree.edges(data=True))
        return BaselineTopology(name="greedy_reliability", edge_ids=tuple(sorted(selected)))
