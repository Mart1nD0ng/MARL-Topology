"""Urban grid (Manhattan) scene generator with real NLOS building blockage.

The production FSPL generator places nodes in free space (no buildings) so every link is
line-of-sight and the 20 dB NLOS penalty never fires -- unrealistic for a *city* topology
problem. This builds a fixed-scale city grid: square building blocks separated by streets,
RSUs at intersections, vehicles on the lanes. Links along the same street are LOS; links
across a block are blocked by the (tall) building -> NLOS. The downstream channel/visibility
chain already consumes ``scene.buildings``, so NLOS triggers automatically.

Geometry is FIXED in absolute meters (an urban canyon scale), NOT rescaled to the FSPL
reliable range -- so blockage, not just distance, controls feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from marl_topology.geometry3d import BuildingBox, Point3D
from marl_topology.scenario.scene import Lane, Node3D, NodeKind, RoadSegment, Scene3D

RSU_HEIGHT_M = 5.0
VEHICLE_HEIGHT_M = 1.5


@dataclass(frozen=True, slots=True)
class UrbanGridConfig:
    blocks_per_side: int = 3          # G x G building blocks
    block_size_m: float = 60.0        # building footprint side
    street_width_m: float = 20.0      # street (gap) between blocks
    building_height_min_m: float = 18.0
    building_height_max_m: float = 36.0  # >> RSU 5 m, so blocks form canyons
    rsu_count: int = 1
    vehicle_count: int = 5
    # Probability a vehicle is placed on a street an RSU sits on (so it has line-of-sight
    # to that RSU). 1.0 -> vehicles cluster into RSU-covered streets (feasible); 0.0 ->
    # scattered across the grid (most NLOS-isolated -> infeasible). This is the feasibility
    # gradient knob for all-nodes PBFT under urban NLOS, where every node must connect.
    vehicle_los_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.blocks_per_side < 1:
            raise ValueError("blocks_per_side must be >= 1")
        if self.block_size_m <= 0 or self.street_width_m <= 0:
            raise ValueError("block_size_m and street_width_m must be positive")
        if self.building_height_min_m <= RSU_HEIGHT_M:
            raise ValueError("buildings must be taller than the RSU to create NLOS canyons")
        if self.rsu_count < 1 or self.vehicle_count < 1:
            raise ValueError("need at least one RSU and one vehicle")

    @property
    def pitch_m(self) -> float:
        return self.block_size_m + self.street_width_m

    @property
    def extent_m(self) -> float:
        return self.blocks_per_side * self.pitch_m


def build_urban_grid_scene(scenario_id: str, config: UrbanGridConfig, rng: Random) -> Scene3D:
    """Build one city-grid scene: G*G building blocks, street grid, RSUs at intersections,
    vehicles on lanes. Deterministic given ``rng``."""
    pitch = config.pitch_m
    g = config.blocks_per_side
    half = config.street_width_m / 2.0
    streets = [k * pitch for k in range(g + 1)]  # street centerline coordinates
    extent = config.extent_m

    # buildings occupy each block BETWEEN streets (inset by half a street width).
    buildings = []
    for i in range(g):
        for j in range(g):
            x0, x1 = streets[i] + half, streets[i + 1] - half
            y0, y1 = streets[j] + half, streets[j + 1] - half
            height = rng.uniform(config.building_height_min_m, config.building_height_max_m)
            buildings.append(
                BuildingBox.from_footprint_height(f"blk_{i}_{j}", x0, x1, y0, y1, height)
            )

    # streets: one horizontal + one vertical road through each grid line, with a lane.
    roads, lanes = [], []
    for k, s in enumerate(streets):
        roads.append(RoadSegment(f"road_h_{k}", Point3D(0.0, s, 0.0), Point3D(extent, s, 0.0)))
        lanes.append(
            Lane(f"lane_h_{k}", f"road_h_{k}", (Point3D(0.0, s, 0.0), Point3D(extent, s, 0.0)), config.street_width_m)
        )
        roads.append(RoadSegment(f"road_v_{k}", Point3D(s, 0.0, 0.0), Point3D(s, extent, 0.0)))
        lanes.append(
            Lane(f"lane_v_{k}", f"road_v_{k}", (Point3D(s, 0.0, 0.0), Point3D(s, extent, 0.0)), config.street_width_m)
        )

    # RSUs at distinct intersections (spread); vehicles at random points along streets.
    intersections = [(streets[i], streets[j]) for i in range(g + 1) for j in range(g + 1)]
    rsu_xy = rng.sample(intersections, min(config.rsu_count, len(intersections)))
    nodes = [
        Node3D(f"rsu_{idx}", NodeKind.RSU, Point3D(x, y, RSU_HEIGHT_M))
        for idx, (x, y) in enumerate(rsu_xy)
    ]
    # the streets an RSU sits on: a vehicle on one of these has LOS along the street to it.
    rsu_streets = []  # ("h", y_coord) horizontal street, ("v", x_coord) vertical street
    for x, y in rsu_xy:
        rsu_streets.append(("h", y))
        rsu_streets.append(("v", x))
    for idx in range(config.vehicle_count):
        along = rng.uniform(0.0, extent)
        if rsu_streets and rng.random() < config.vehicle_los_bias:
            orient, coord = rng.choice(rsu_streets)  # place on an RSU-covered street
            x, y = (along, coord) if orient == "h" else (coord, along)
        else:
            s = rng.choice(streets)
            x, y = (along, s) if rng.random() < 0.5 else (s, along)
        nodes.append(Node3D(f"veh_{idx}", NodeKind.VEHICLE, Point3D(x, y, VEHICLE_HEIGHT_M)))

    return Scene3D(
        scenario_id=scenario_id,
        nodes=tuple(nodes),
        buildings=tuple(buildings),
        roads=tuple(roads),
        lanes=tuple(lanes),
    )
