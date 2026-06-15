"""Deterministic Stage 2 link model.

This is a deliberately simple distance-based abstraction, not a 3D physical
simulator. It exists to make the goal skeleton executable while keeping units
and monotonic sanity testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from marl_topology.topology.candidate_graph import CandidateEdge, CandidateGraph

SPEED_OF_LIGHT_MPS = 299_792_458.0


@dataclass(frozen=True, slots=True)
class LinkRecord:
    edge_id: str
    tx_id: str
    rx_id: str
    distance_3d_m: float
    link_success_probability: float
    latency_s: float
    energy_j: float
    physics_regime: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.link_success_probability <= 1.0:
            raise ValueError("link_success_probability must be in [0, 1]")
        if self.latency_s < 0:
            raise ValueError("latency_s must be nonnegative")
        if self.energy_j < 0:
            raise ValueError("energy_j must be nonnegative")


@dataclass(frozen=True, slots=True)
class SimpleLinkModel:
    reference_distance_m: float = 100.0
    base_latency_s: float = 0.001
    energy_per_m_j: float = 0.001
    fixed_tx_energy_j: float = 0.01

    def __post_init__(self) -> None:
        if self.reference_distance_m <= 0:
            raise ValueError("reference_distance_m must be positive")
        if self.base_latency_s < 0:
            raise ValueError("base_latency_s must be nonnegative")
        if self.energy_per_m_j < 0:
            raise ValueError("energy_per_m_j must be nonnegative")
        if self.fixed_tx_energy_j < 0:
            raise ValueError("fixed_tx_energy_j must be nonnegative")

    def link_success_probability(self, distance_3d_m: float) -> float:
        if distance_3d_m < 0:
            raise ValueError("distance_3d_m must be nonnegative")
        return exp(-distance_3d_m / self.reference_distance_m)

    def evaluate_edge(self, edge: CandidateEdge, physics_regime: str) -> LinkRecord:
        distance_m = edge.distance_3d_m
        return LinkRecord(
            edge_id=edge.edge_id,
            tx_id=edge.node_u,
            rx_id=edge.node_v,
            distance_3d_m=distance_m,
            link_success_probability=self.link_success_probability(distance_m),
            latency_s=self.base_latency_s + distance_m / SPEED_OF_LIGHT_MPS,
            energy_j=self.fixed_tx_energy_j + self.energy_per_m_j * distance_m,
            physics_regime=physics_regime,
        )

    def evaluate_graph(self, graph: CandidateGraph) -> dict[str, LinkRecord]:
        return {
            edge.edge_id: self.evaluate_edge(edge, graph.physics_regime)
            for edge in graph.edges
        }
