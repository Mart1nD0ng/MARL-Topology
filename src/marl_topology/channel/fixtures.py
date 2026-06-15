"""Deterministic Stage 3.2 channel fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from marl_topology.geometry3d import Point3D
from marl_topology.scenario import Node3D, NodeKind, Scene3D, get_geometry_visibility_fixture

from .model import ActiveTransmission, STAGE3_CHANNEL_REGIME_ID


@dataclass(frozen=True, slots=True)
class ChannelModelFixture:
    fixture_id: str
    description: str
    scene: Scene3D
    tx_id: str
    rx_id: str
    resource_id: str
    active_transmissions: tuple[ActiveTransmission, ...]
    expected_interference_tx_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fixture_id != self.scene.scenario_id:
            raise ValueError("fixture_id must match scene.scenario_id")
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        self.scene.get_node(self.tx_id)
        self.scene.get_node(self.rx_id)


def iter_channel_model_fixtures() -> tuple[ChannelModelFixture, ...]:
    return (
        free_space_close_channel_fixture(),
        free_space_far_channel_fixture(),
        blocked_by_building_channel_fixture(),
        two_transmitters_interference_fixture(),
        orthogonal_channel_no_interference_fixture(),
    )


def get_channel_model_fixture(fixture_id: str) -> ChannelModelFixture:
    for fixture in iter_channel_model_fixtures():
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown channel model fixture: {fixture_id}")


def free_space_close_channel_fixture() -> ChannelModelFixture:
    geometry_fixture = get_geometry_visibility_fixture("free_space_close")
    return ChannelModelFixture(
        fixture_id=geometry_fixture.fixture_id,
        description="Channel fixture for close free-space LoS.",
        scene=geometry_fixture.scene,
        tx_id=geometry_fixture.tx_id,
        rx_id=geometry_fixture.rx_id,
        resource_id="resource_0",
        active_transmissions=(),
        expected_interference_tx_ids=(),
    )


def free_space_far_channel_fixture() -> ChannelModelFixture:
    geometry_fixture = get_geometry_visibility_fixture("free_space_far")
    return ChannelModelFixture(
        fixture_id=geometry_fixture.fixture_id,
        description="Channel fixture for far free-space LoS.",
        scene=geometry_fixture.scene,
        tx_id=geometry_fixture.tx_id,
        rx_id=geometry_fixture.rx_id,
        resource_id="resource_0",
        active_transmissions=(),
        expected_interference_tx_ids=(),
    )


def blocked_by_building_channel_fixture() -> ChannelModelFixture:
    geometry_fixture = get_geometry_visibility_fixture("blocked_by_building")
    return ChannelModelFixture(
        fixture_id=geometry_fixture.fixture_id,
        description="Channel fixture for NLoS building blockage.",
        scene=geometry_fixture.scene,
        tx_id=geometry_fixture.tx_id,
        rx_id=geometry_fixture.rx_id,
        resource_id="resource_0",
        active_transmissions=(),
        expected_interference_tx_ids=(),
    )


def two_transmitters_interference_fixture() -> ChannelModelFixture:
    scene = _interference_scene("two_transmitters_interference")
    return ChannelModelFixture(
        fixture_id="two_transmitters_interference",
        description="Same-resource active transmitter contributes interference.",
        scene=scene,
        tx_id="rsu_0",
        rx_id="veh_0",
        resource_id="resource_0",
        active_transmissions=(
            ActiveTransmission("rsu_1", "veh_1", resource_id="resource_0"),
        ),
        expected_interference_tx_ids=("rsu_1",),
    )


def orthogonal_channel_no_interference_fixture() -> ChannelModelFixture:
    scene = _interference_scene("orthogonal_channel_no_interference")
    return ChannelModelFixture(
        fixture_id="orthogonal_channel_no_interference",
        description="Orthogonal active transmitter is excluded from interference.",
        scene=scene,
        tx_id="rsu_0",
        rx_id="veh_0",
        resource_id="resource_0",
        active_transmissions=(
            ActiveTransmission("rsu_1", "veh_1", resource_id="resource_1"),
        ),
        expected_interference_tx_ids=(),
    )


def _interference_scene(scenario_id: str) -> Scene3D:
    return Scene3D(
        scenario_id=scenario_id,
        nodes=(
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 8.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(60.0, 0.0, 1.5)),
            Node3D("rsu_1", NodeKind.RSU, Point3D(60.0, 35.0, 8.0)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(90.0, 35.0, 1.5)),
        ),
        physics_regime=STAGE3_CHANNEL_REGIME_ID,
    )
