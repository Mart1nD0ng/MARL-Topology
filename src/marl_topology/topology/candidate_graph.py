"""Candidate communication graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from marl_topology.geometry3d import distance_3d
from marl_topology.scenario import Scene3D


def canonical_edge_id(node_a: str, node_b: str) -> str:
    if node_a == node_b:
        raise ValueError("self edges are not valid candidate edges")
    left, right = sorted((node_a, node_b))
    return f"{left}--{right}"


@dataclass(frozen=True, slots=True)
class CandidateEdge:
    edge_id: str
    node_u: str
    node_v: str
    distance_3d_m: float

    def __post_init__(self) -> None:
        expected = canonical_edge_id(self.node_u, self.node_v)
        if self.edge_id != expected:
            raise ValueError(f"edge_id must be canonical: {expected}")
        if self.distance_3d_m < 0:
            raise ValueError("distance_3d_m must be nonnegative")


@dataclass(frozen=True, slots=True)
class CandidateGraph:
    scenario_id: str
    node_ids: tuple[str, ...]
    edges: tuple[CandidateEdge, ...]
    physics_regime: str

    def __post_init__(self) -> None:
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ValueError("node_ids must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("candidate edge ids must be unique")
        object.__setattr__(self, "node_ids", tuple(sorted(self.node_ids)))
        object.__setattr__(self, "edges", tuple(sorted(self.edges, key=lambda e: e.edge_id)))

    @classmethod
    def from_scene(cls, scene: Scene3D, max_distance_m: float | None = None) -> "CandidateGraph":
        if max_distance_m is not None and max_distance_m < 0:
            raise ValueError("max_distance_m must be nonnegative")
        edges: list[CandidateEdge] = []
        for left, right in combinations(scene.nodes, 2):
            distance_m = distance_3d(left.position, right.position)
            if max_distance_m is None or distance_m <= max_distance_m:
                edges.append(
                    CandidateEdge(
                        edge_id=canonical_edge_id(left.node_id, right.node_id),
                        node_u=left.node_id,
                        node_v=right.node_id,
                        distance_3d_m=distance_m,
                    )
                )
        return cls(
            scenario_id=scene.scenario_id,
            node_ids=scene.node_ids,
            edges=tuple(edges),
            physics_regime=scene.physics_regime,
        )

    @property
    def edge_ids(self) -> tuple[str, ...]:
        return tuple(edge.edge_id for edge in self.edges)

    def get_edge(self, edge_id: str) -> CandidateEdge:
        for edge in self.edges:
            if edge.edge_id == edge_id:
                return edge
        raise KeyError(f"unknown edge_id: {edge_id}")

    def is_full_selection(self, selected_edge_ids: set[str] | frozenset[str]) -> bool:
        return set(self.edge_ids) == set(selected_edge_ids)
