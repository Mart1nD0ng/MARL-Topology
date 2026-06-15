"""Deterministic Stage 3.4 network communication fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from marl_topology.geometry3d import Point3D
from marl_topology.scenario import Node3D, NodeKind, Scene3D
from marl_topology.topology import CandidateGraph, canonical_edge_id

from .communication import (
    NetworkCommunicationConfig,
    NetworkCommunicationRecord,
    NetworkTransmissionSpec,
    evaluate_network_communication,
)


@dataclass(frozen=True, slots=True)
class NetworkCommunicationFixture:
    fixture_id: str
    description: str
    scene: Scene3D
    selected_edge_ids: tuple[str, ...]
    source_id: str
    target_ids: tuple[str, ...]
    config: NetworkCommunicationConfig = field(default_factory=NetworkCommunicationConfig)
    resource_assignments: Mapping[str, str] = field(default_factory=dict)
    background_transmissions: tuple[NetworkTransmissionSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.fixture_id != self.scene.scenario_id:
            raise ValueError("fixture_id must match scene.scenario_id")
        self.scene.get_node(self.source_id)
        for target_id in self.target_ids:
            self.scene.get_node(target_id)

    def candidate_graph(self) -> CandidateGraph:
        return CandidateGraph.from_scene(self.scene)

    def evaluate(self) -> NetworkCommunicationRecord:
        return evaluate_network_communication(
            scene=self.scene,
            graph=self.candidate_graph(),
            selected_edge_ids=self.selected_edge_ids,
            source_id=self.source_id,
            target_ids=self.target_ids,
            config=self.config,
            resource_assignments=self.resource_assignments,
            background_transmissions=self.background_transmissions,
        )


def iter_network_communication_fixtures() -> tuple[NetworkCommunicationFixture, ...]:
    return (
        multi_hop_delivery_fixture(),
        disconnected_reachability_fixture(),
        two_transmitters_interference_network_fixture(),
        orthogonal_channel_no_interference_network_fixture(),
        resource_redundant_topology_fixture(),
    )


def get_network_communication_fixture(fixture_id: str) -> NetworkCommunicationFixture:
    for fixture in iter_network_communication_fixtures():
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown network communication fixture: {fixture_id}")


def multi_hop_delivery_fixture() -> NetworkCommunicationFixture:
    scene = _line_scene("multi_hop_delivery")
    return NetworkCommunicationFixture(
        fixture_id="multi_hop_delivery",
        description="Route primitive aggregates two selected hops.",
        scene=scene,
        selected_edge_ids=(
            canonical_edge_id("rsu_0", "veh_0"),
            canonical_edge_id("veh_0", "veh_1"),
        ),
        source_id="rsu_0",
        target_ids=("veh_1",),
    )


def disconnected_reachability_fixture() -> NetworkCommunicationFixture:
    scene = _line_scene("disconnected_reachability")
    return NetworkCommunicationFixture(
        fixture_id="disconnected_reachability",
        description="Selected topology cannot reach target.",
        scene=scene,
        selected_edge_ids=(canonical_edge_id("rsu_0", "veh_0"),),
        source_id="rsu_0",
        target_ids=("veh_2",),
    )


def two_transmitters_interference_network_fixture() -> NetworkCommunicationFixture:
    scene = _line_scene("two_transmitters_interference_network")
    return NetworkCommunicationFixture(
        fixture_id="two_transmitters_interference_network",
        description="Background same-resource transmission interferes with route hop.",
        scene=scene,
        selected_edge_ids=(canonical_edge_id("rsu_0", "veh_0"),),
        source_id="rsu_0",
        target_ids=("veh_0",),
        background_transmissions=(
            NetworkTransmissionSpec(
                edge_id=canonical_edge_id("rsu_1", "veh_2"),
                tx_id="rsu_1",
                rx_id="veh_2",
                resource_id="resource_0",
            ),
        ),
    )


def orthogonal_channel_no_interference_network_fixture() -> NetworkCommunicationFixture:
    scene = _line_scene("orthogonal_channel_no_interference_network")
    return NetworkCommunicationFixture(
        fixture_id="orthogonal_channel_no_interference_network",
        description="Background orthogonal-resource transmission is excluded.",
        scene=scene,
        selected_edge_ids=(canonical_edge_id("rsu_0", "veh_0"),),
        source_id="rsu_0",
        target_ids=("veh_0",),
        background_transmissions=(
            NetworkTransmissionSpec(
                edge_id=canonical_edge_id("rsu_1", "veh_2"),
                tx_id="rsu_1",
                rx_id="veh_2",
                resource_id="resource_1",
            ),
        ),
    )


def resource_redundant_topology_fixture() -> NetworkCommunicationFixture:
    scene = _line_scene("resource_redundant_topology")
    return NetworkCommunicationFixture(
        fixture_id="resource_redundant_topology",
        description="Broadcast primitive activates redundant selected edges.",
        scene=scene,
        selected_edge_ids=(
            canonical_edge_id("rsu_0", "veh_0"),
            canonical_edge_id("veh_0", "veh_1"),
            canonical_edge_id("rsu_0", "veh_1"),
        ),
        source_id="rsu_0",
        target_ids=("veh_0", "veh_1"),
        config=NetworkCommunicationConfig(primitive="broadcast"),
    )


def _line_scene(scenario_id: str) -> Scene3D:
    return Scene3D(
        scenario_id=scenario_id,
        nodes=(
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 8.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(40.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(80.0, 0.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(140.0, 0.0, 1.5)),
            Node3D("rsu_1", NodeKind.RSU, Point3D(42.0, 25.0, 8.0)),
        ),
        physics_regime="stage3_network_communication_v1",
    )
