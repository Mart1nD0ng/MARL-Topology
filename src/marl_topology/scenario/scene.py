"""Deterministic 3D scene container for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from marl_topology.geometry3d import BuildingBox, Point3D


class NodeKind(str, Enum):
    VEHICLE = "vehicle"
    RSU = "rsu"
    BASE_STATION = "base_station"


@dataclass(frozen=True, slots=True)
class RoadSegment:
    road_id: str
    start: Point3D
    end: Point3D

    def __post_init__(self) -> None:
        if not self.road_id:
            raise ValueError("road_id must be non-empty")
        if self.start == self.end:
            raise ValueError("road segment endpoints must differ")


@dataclass(frozen=True, slots=True)
class Lane:
    lane_id: str
    road_id: str
    centerline: tuple[Point3D, ...]
    width_m: float = 3.5

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("lane_id must be non-empty")
        if not self.road_id:
            raise ValueError("road_id must be non-empty")
        if len(self.centerline) < 2:
            raise ValueError("lane centerline must contain at least two points")
        if self.width_m <= 0:
            raise ValueError("width_m must be positive")


@dataclass(frozen=True, slots=True)
class Node3D:
    node_id: str
    kind: NodeKind
    position: Point3D

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")


@dataclass(frozen=True, slots=True)
class NodeMotion:
    """Constant-velocity motion for one node, in m/s. Immutable, like Node3D.

    The minimal mobility model: a node's position advances by p' = p + v*dt each
    rollout step (see ``advance_scene``). RSUs / base stations are fixed (zero
    velocity). Constant velocity is deliberately the simplest model that makes
    links change *predictably* over time -- the temporal signal a recurrent/GNN
    actor can exploit. Lane-constrained or car-following motion is a later
    refinement; this primitive is free-space constant velocity.
    """

    node_id: str
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if len(self.velocity_mps) != 3:
            raise ValueError("velocity_mps must be a 3-tuple (vx, vy, vz) in m/s")
        for component in self.velocity_mps:
            float(component)  # raises TypeError on a non-numeric component

    @property
    def is_stationary(self) -> bool:
        return self.velocity_mps == (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class Scene3D:
    """Minimal scene state.

    The default physics regime is deliberately explicit. It allows later tests
    to reject cross-regime policy claims before a full simulator exists.
    """

    scenario_id: str
    nodes: tuple[Node3D, ...]
    buildings: tuple[BuildingBox, ...] = field(default_factory=tuple)
    roads: tuple[RoadSegment, ...] = field(default_factory=tuple)
    lanes: tuple[Lane, ...] = field(default_factory=tuple)
    physics_regime: str = "stage2_deterministic_distance"
    distance_unit: str = "m"

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if self.distance_unit != "m":
            raise ValueError("Stage 2 Scene3D requires meter distance units")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node_id values must be unique")
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda n: n.node_id)))
        object.__setattr__(
            self,
            "buildings",
            tuple(sorted(self.buildings, key=lambda b: b.building_id)),
        )
        road_ids = [road.road_id for road in self.roads]
        if len(road_ids) != len(set(road_ids)):
            raise ValueError("road_id values must be unique")
        object.__setattr__(self, "roads", tuple(sorted(self.roads, key=lambda r: r.road_id)))
        lane_ids = [lane.lane_id for lane in self.lanes]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("lane_id values must be unique")
        known_roads = set(road_ids)
        unknown_lane_roads = sorted({lane.road_id for lane in self.lanes} - known_roads)
        if unknown_lane_roads:
            raise ValueError(f"lane references unknown road ids: {unknown_lane_roads}")
        object.__setattr__(self, "lanes", tuple(sorted(self.lanes, key=lambda lane: lane.lane_id)))

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    def get_node(self, node_id: str) -> Node3D:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"unknown node_id: {node_id}")

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "Scene3D":
        scenario_id = str(config.get("scenario_id", "scenario"))
        physics_regime = str(config.get("physics_regime", "stage2_deterministic_distance"))
        raw_nodes = config.get("nodes", [])
        if not isinstance(raw_nodes, Sequence):
            raise TypeError("nodes must be a sequence")

        nodes = []
        for raw in raw_nodes:
            if not isinstance(raw, Mapping):
                raise TypeError("each node must be a mapping")
            position = raw.get("position_m")
            if not isinstance(position, Sequence) or len(position) != 3:
                raise ValueError("position_m must contain three coordinates")
            nodes.append(
                Node3D(
                    node_id=str(raw["node_id"]),
                    kind=NodeKind(str(raw["kind"])),
                    position=Point3D(float(position[0]), float(position[1]), float(position[2])),
                )
            )

        buildings = []
        raw_buildings = config.get("buildings", [])
        if not isinstance(raw_buildings, Sequence):
            raise TypeError("buildings must be a sequence")
        for raw in raw_buildings:
            if not isinstance(raw, Mapping):
                raise TypeError("each building must be a mapping")
            min_corner = raw.get("min_corner_m")
            max_corner = raw.get("max_corner_m")
            if not isinstance(min_corner, Sequence) or len(min_corner) != 3:
                raise ValueError("min_corner_m must contain three coordinates")
            if not isinstance(max_corner, Sequence) or len(max_corner) != 3:
                raise ValueError("max_corner_m must contain three coordinates")
            buildings.append(
                BuildingBox(
                    building_id=str(raw["building_id"]),
                    min_corner=Point3D(
                        float(min_corner[0]),
                        float(min_corner[1]),
                        float(min_corner[2]),
                    ),
                    max_corner=Point3D(
                        float(max_corner[0]),
                        float(max_corner[1]),
                        float(max_corner[2]),
                    ),
                    material_class=str(raw.get("material_class", "generic")),
                )
            )

        return cls(
            scenario_id=scenario_id,
            nodes=tuple(nodes),
            buildings=tuple(buildings),
            physics_regime=physics_regime,
        )


def make_demo_scene(nodes: Iterable[Node3D] | None = None) -> Scene3D:
    if nodes is None:
        nodes = (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 40.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(30.0, 0.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(65.0, 0.0, 1.5)),
        )
    return Scene3D(scenario_id="demo_stage2", nodes=tuple(nodes))


def advance_scene(
    scene: Scene3D,
    motions: Mapping[str, NodeMotion],
    dt_s: float,
) -> Scene3D:
    """Return a NEW Scene3D with each node's position advanced by ``p' = p + v*dt``.

    PURE / immutable: the input ``scene`` is never mutated (Node3D / Scene3D are
    frozen). A node with no ``NodeMotion`` entry -- or a stationary one (RSU /
    base station) -- keeps its position exactly, so geometry only changes for the
    moving vehicles. Everything else (buildings, roads, lanes, regime, scenario_id)
    is carried through unchanged: advancing time does not change the *scenario*, only
    the geometry within it. ``motions`` keys that are not in the scene are ignored.

    This is the env-side primitive that makes a rollout step a real time step, so a
    temporal actor has a predictable link-evolution signal to learn from. It is
    deterministic (no RNG).
    """

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    advanced: list[Node3D] = []
    for node in scene.nodes:
        motion = motions.get(node.node_id)
        if motion is None or motion.is_stationary:
            advanced.append(node)
            continue
        vx, vy, vz = motion.velocity_mps
        moved = Point3D(
            node.position.x_m + vx * dt_s,
            node.position.y_m + vy * dt_s,
            node.position.z_m + vz * dt_s,
        )
        advanced.append(Node3D(node.node_id, node.kind, moved))

    return Scene3D(
        scenario_id=scene.scenario_id,
        nodes=tuple(advanced),
        buildings=scene.buildings,
        roads=scene.roads,
        lanes=scene.lanes,
        physics_regime=scene.physics_regime,
        distance_unit=scene.distance_unit,
    )
