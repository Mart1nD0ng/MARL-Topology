from marl_topology.geometry3d import Point3D, distance_3d
from marl_topology.scenario import Node3D, NodeKind, Scene3D
from marl_topology.topology import CandidateGraph, canonical_edge_id


def test_node_identity_is_stable_and_sorted() -> None:
    scene = Scene3D(
        scenario_id="stable_nodes",
        nodes=(
            Node3D("veh_b", NodeKind.VEHICLE, Point3D(1.0, 0.0, 0.0)),
            Node3D("rsu_a", NodeKind.RSU, Point3D(0.0, 0.0, 0.0)),
        ),
    )

    assert scene.node_ids == ("rsu_a", "veh_b")
    assert scene.get_node("veh_b").kind == NodeKind.VEHICLE


def test_candidate_edge_identity_is_stable() -> None:
    scene = Scene3D.from_config(
        {
            "scenario_id": "stable_edges",
            "nodes": [
                {"node_id": "veh_1", "kind": "vehicle", "position_m": [10.0, 0.0, 0.0]},
                {"node_id": "veh_0", "kind": "vehicle", "position_m": [0.0, 0.0, 0.0]},
                {"node_id": "rsu_0", "kind": "rsu", "position_m": [0.0, 10.0, 0.0]},
            ],
        }
    )
    graph_a = CandidateGraph.from_scene(scene)
    graph_b = CandidateGraph.from_scene(scene)

    assert graph_a.node_ids == ("rsu_0", "veh_0", "veh_1")
    assert graph_a.edge_ids == graph_b.edge_ids
    assert canonical_edge_id("veh_1", "veh_0") == "veh_0--veh_1"
    assert len(graph_a.edge_ids) == len(set(graph_a.edge_ids))


def test_distance_3d_sanity() -> None:
    assert distance_3d(Point3D(0.0, 0.0, 0.0), Point3D(3.0, 4.0, 12.0)) == 13.0
