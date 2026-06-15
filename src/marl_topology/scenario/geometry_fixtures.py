"""Stage 3.1 deterministic geometry visibility fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from marl_topology.geometry3d import BuildingBox, Point3D
from marl_topology.scenario.scene import Lane, Node3D, NodeKind, RoadSegment, Scene3D


STAGE3_GEOMETRY_REGIME = "stage3_axis_aligned_boxes"


@dataclass(frozen=True, slots=True)
class GeometryVisibilityFixture:
    fixture_id: str
    description: str
    scene: Scene3D
    tx_id: str
    rx_id: str
    expected_los_state: str
    expected_blocker_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fixture_id != self.scene.scenario_id:
            raise ValueError("fixture_id must match scene.scenario_id")
        if self.expected_los_state not in {"los", "nlos"}:
            raise ValueError("expected_los_state must be los or nlos")
        self.scene.get_node(self.tx_id)
        self.scene.get_node(self.rx_id)


def iter_geometry_visibility_fixtures() -> tuple[GeometryVisibilityFixture, ...]:
    return (
        free_space_close_fixture(),
        free_space_far_fixture(),
        blocked_by_building_fixture(),
        urban_gap_los_fixture(),
        urban_canyon_nlos_fixture(),
        rsu_high_los_fixture(),
        low_rsu_blocked_variant(),
    )


def get_geometry_visibility_fixture(fixture_id: str) -> GeometryVisibilityFixture:
    for fixture in iter_geometry_visibility_fixtures():
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown geometry visibility fixture: {fixture_id}")


def free_space_close_fixture() -> GeometryVisibilityFixture:
    return GeometryVisibilityFixture(
        fixture_id="free_space_close",
        description="Close free-space LoS geometry fixture.",
        scene=_scene(
            "free_space_close",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 8.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(30.0, 0.0, 1.5)),
            ),
        ),
        tx_id="rsu_0",
        rx_id="veh_0",
        expected_los_state="los",
        expected_blocker_ids=(),
    )


def free_space_far_fixture() -> GeometryVisibilityFixture:
    return GeometryVisibilityFixture(
        fixture_id="free_space_far",
        description="Far free-space LoS geometry fixture.",
        scene=_scene(
            "free_space_far",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 8.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(300.0, 0.0, 1.5)),
            ),
        ),
        tx_id="rsu_0",
        rx_id="veh_0",
        expected_los_state="los",
        expected_blocker_ids=(),
    )


def blocked_by_building_fixture() -> GeometryVisibilityFixture:
    return GeometryVisibilityFixture(
        fixture_id="blocked_by_building",
        description="A building box blocks the ray between RSU and vehicle.",
        scene=_scene(
            "blocked_by_building",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 4.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(100.0, 0.0, 1.5)),
            ),
            buildings=(
                BuildingBox.from_footprint_height("building_blocker", 40.0, 60.0, -10.0, 10.0, 20.0),
            ),
        ),
        tx_id="rsu_0",
        rx_id="veh_0",
        expected_los_state="nlos",
        expected_blocker_ids=("building_blocker",),
    )


def urban_gap_los_fixture() -> GeometryVisibilityFixture:
    return GeometryVisibilityFixture(
        fixture_id="urban_gap_los",
        description="Two buildings leave a clear corridor along the communication ray.",
        scene=_scene(
            "urban_gap_los",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 6.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(120.0, 0.0, 1.5)),
            ),
            buildings=(
                BuildingBox.from_footprint_height("north_wall", 40.0, 80.0, 5.0, 25.0, 35.0),
                BuildingBox.from_footprint_height("south_wall", 40.0, 80.0, -25.0, -5.0, 35.0),
            ),
        ),
        tx_id="rsu_0",
        rx_id="veh_0",
        expected_los_state="los",
        expected_blocker_ids=(),
    )


def urban_canyon_nlos_fixture() -> GeometryVisibilityFixture:
    return GeometryVisibilityFixture(
        fixture_id="urban_canyon_nlos",
        description="Tall building in an urban canyon blocks the low RSU ray.",
        scene=_scene(
            "urban_canyon_nlos",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(120.0, 0.0, 1.5)),
            ),
            buildings=(
                BuildingBox.from_footprint_height("canyon_blocker", 50.0, 70.0, -4.0, 4.0, 40.0),
                BuildingBox.from_footprint_height("north_canyon", 45.0, 80.0, 8.0, 20.0, 40.0),
                BuildingBox.from_footprint_height("south_canyon", 45.0, 80.0, -20.0, -8.0, 40.0),
            ),
        ),
        tx_id="rsu_0",
        rx_id="veh_0",
        expected_los_state="nlos",
        expected_blocker_ids=("canyon_blocker",),
    )


def rsu_high_los_fixture() -> GeometryVisibilityFixture:
    return GeometryVisibilityFixture(
        fixture_id="rsu_high_los",
        description="Higher RSU ray clears a low blocker that would block a low RSU.",
        scene=_scene(
            "rsu_high_los",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 30.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(100.0, 0.0, 1.5)),
            ),
            buildings=(
                BuildingBox.from_footprint_height("low_blocker", 40.0, 60.0, -10.0, 10.0, 10.0),
            ),
        ),
        tx_id="rsu_0",
        rx_id="veh_0",
        expected_los_state="los",
        expected_blocker_ids=(),
    )


def low_rsu_blocked_variant() -> GeometryVisibilityFixture:
    return GeometryVisibilityFixture(
        fixture_id="rsu_low_nlos",
        description="Low RSU variant used to verify RSU height effect.",
        scene=_scene(
            "rsu_low_nlos",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(100.0, 0.0, 1.5)),
            ),
            buildings=(
                BuildingBox.from_footprint_height("low_blocker", 40.0, 60.0, -10.0, 10.0, 10.0),
            ),
        ),
        tx_id="rsu_0",
        rx_id="veh_0",
        expected_los_state="nlos",
        expected_blocker_ids=("low_blocker",),
    )


def _scene(
    scenario_id: str,
    nodes: tuple[Node3D, ...],
    buildings: tuple[BuildingBox, ...] = (),
) -> Scene3D:
    road = RoadSegment(
        road_id="road_0",
        start=Point3D(-20.0, 0.0, 0.0),
        end=Point3D(320.0, 0.0, 0.0),
    )
    lane = Lane(
        lane_id="lane_0",
        road_id=road.road_id,
        centerline=(road.start, road.end),
    )
    return Scene3D(
        scenario_id=scenario_id,
        nodes=nodes,
        buildings=buildings,
        roads=(road,),
        lanes=(lane,),
        physics_regime=STAGE3_GEOMETRY_REGIME,
    )
