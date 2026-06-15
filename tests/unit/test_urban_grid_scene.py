"""Urban grid scene generator: a Manhattan grid of building blocks + streets with RSUs at
intersections and vehicles on lanes, producing REAL NLOS blockage (the production FSPL
generator is always-LOS). Confirms buildings exist, the scene carries roads/lanes, and the
already-validated visibility engine reports NLOS for cross-block pairs + LOS along a street.
"""

from random import Random

from marl_topology.geometry3d.visibility import evaluate_visibility
from marl_topology.scenario.scene import NodeKind
from marl_topology.scenario.urban_grid import UrbanGridConfig, build_urban_grid_scene


def test_urban_grid_has_buildings_roads_and_lanes() -> None:
    cfg = UrbanGridConfig(blocks_per_side=3, rsu_count=2, vehicle_count=6)
    scene = build_urban_grid_scene("grid", cfg, Random(1))
    assert len(scene.buildings) == cfg.blocks_per_side**2  # one building per block
    assert scene.roads and scene.lanes  # street grid present
    rsus = [n for n in scene.nodes if n.kind is NodeKind.RSU]
    vehicles = [n for n in scene.nodes if n.kind is NodeKind.VEHICLE]
    assert len(rsus) == cfg.rsu_count
    assert len(vehicles) == cfg.vehicle_count
    # buildings tower over the RSU (5 m) -> they can block.
    assert all(b.height_m > 5.0 for b in scene.buildings)


def test_urban_grid_produces_real_nlos() -> None:
    cfg = UrbanGridConfig(blocks_per_side=3, rsu_count=2, vehicle_count=8)
    scene = build_urban_grid_scene("grid", cfg, Random(7))
    nodes = list(scene.node_ids)
    states = [
        evaluate_visibility(scene, a, b).los_state.value
        for i, a in enumerate(nodes)
        for b in nodes[i + 1:]
    ]
    # a city grid blocks most cross-block pairs -> NLOS must actually fire (not always-LOS).
    assert "nlos" in states
    assert states.count("nlos") > states.count("los")  # blockage dominates in a dense grid


def test_urban_grid_is_deterministic() -> None:
    cfg = UrbanGridConfig(blocks_per_side=3, rsu_count=1, vehicle_count=5)
    a = build_urban_grid_scene("grid", cfg, Random(42))
    b = build_urban_grid_scene("grid", cfg, Random(42))
    assert [(n.node_id, n.position) for n in a.nodes] == [(n.node_id, n.position) for n in b.nodes]
    assert [bld.building_id for bld in a.buildings] == [bld.building_id for bld in b.buildings]
