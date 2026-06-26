"""D5: local motion features in the dynamic actor observation (Plan §7, Contract §3.5).

Adds each node's OWN velocity/heading and per-link relative-velocity / distance-delta / CSI-delta so a
memoryless actor can predict the next frame's link evolution (a fair test of Markovness). Opt-in
(motion_features flag); the shared graph_payload featurizer is untouched (static path byte-identical).
The deployed actor stays local/neighbour-only; the centralized critic gets all nodes' velocities
(training-only global view).
"""

from __future__ import annotations

import torch

from marl_topology.data.stage31_scenario_generator import PhysicsRegime, measure_reliable_range_m
from marl_topology.geometry3d import Point3D
from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic
from marl_topology.scenario.scene import Node3D, NodeKind, NodeMotion, Scene3D
from marl_topology.training.decentralized_distillation import feature_standardization
from marl_topology.training.dynamic_frames import dynamic_scene_from_motion
from marl_topology.training.graph_mappo import critic_scene_value
from marl_topology.training.two_timescale_env import ReconfigCost

_R = measure_reliable_range_m(PhysicsRegime())


def _scene(motions, motion_features, nodes=None, num_frames=3):
    if nodes is None:
        nodes = (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.2 * _R, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.2 * _R, 1.5)),
        )
    base = Scene3D(scenario_id="mf", nodes=nodes)
    return dynamic_scene_from_motion(
        base, motions, PhysicsRegime(), quorum_size=3, num_frames=num_frames, dt_s=1.0,
        reliable_range_m=_R, reconfig=ReconfigCost(e_edge=0.05, l_edge=0.0),
        motion_features=motion_features)


def _node_row(scene, node_id, t=0):
    obs = scene.observation(t, [])
    order = list(scene.context(t).graph.node_ids)
    return obs["nf"][order.index(node_id)]


def _edge_row(scene, u, v, t=0):
    obs = scene.observation(t, [])
    for idx, e in enumerate(scene.context(t).graph.edges):
        if {e.node_u, e.node_v} == {u, v}:
            return obs["ef"][idx]
    raise AssertionError(f"edge {u}--{v} not found")


def test_dynamic_actor_observation_contains_velocity() -> None:
    motions = (NodeMotion("veh_0", (10.0, 0.0, 0.0)),)
    on, off = _scene(motions, True), _scene(motions, False)
    nf_on, nf_off = on.observation(0, [])["nf"], off.observation(0, [])["nf"]
    ef_on, ef_off = on.observation(0, [])["ef"], off.observation(0, [])["ef"]
    base_n, base_e = nf_off.shape[1], ef_off.shape[1]
    assert nf_on.shape[1] == base_n + 5      # [vx, vy, speed, heading_sin, heading_cos]
    assert ef_on.shape[1] == base_e + 4      # [rel_vel_along, distance_delta, csi_delta, csi_age]
    vx, vy, speed, hs, hc = _node_row(on, "veh_0")[base_n:base_n + 5].tolist()
    assert (vx, vy) == (10.0, 0.0)
    assert abs(speed - 10.0) < 1e-6
    assert abs(hc - 1.0) < 1e-6 and abs(hs - 0.0) < 1e-6     # heading along +x


def test_relative_velocity_feature_changes_sign() -> None:
    nodes = (
        Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.3 * _R, 0.0, 1.5)),
        Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.0, 1.5)),
        Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.1 * _R, 5.0)),
    )
    base_e = _scene((NodeMotion("veh_0", (-10.0, 0.0, 0.0)),), False, nodes=nodes).observation(0, [])["ef"].shape[1]
    approach = _edge_row(_scene((NodeMotion("veh_0", (-10.0, 0.0, 0.0)),), True, nodes=nodes), "veh_0", "veh_1")
    depart = _edge_row(_scene((NodeMotion("veh_0", (10.0, 0.0, 0.0)),), True, nodes=nodes), "veh_0", "veh_1")
    rv_approach = float(approach[base_e])    # relative_velocity_along_link (first appended edge feature)
    rv_depart = float(depart[base_e])
    assert rv_approach < 0.0 < rv_depart     # approaching closes the link (<0); departing opens it (>0)


def test_memoryless_with_velocity_can_distinguish_approach_vs_depart() -> None:
    nodes = (
        Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.3 * _R, 0.0, 1.5)),
        Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.0, 1.5)),
        Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.1 * _R, 5.0)),
    )
    approach = _scene((NodeMotion("veh_0", (-10.0, 0.0, 0.0)),), True, nodes=nodes).observation(0, [])
    depart = _scene((NodeMotion("veh_0", (10.0, 0.0, 0.0)),), True, nodes=nodes).observation(0, [])
    base_e = _scene((NodeMotion("veh_0", (-10.0, 0.0, 0.0)),), False, nodes=nodes).observation(0, [])["ef"].shape[1]
    # frame 0 geometry identical -> current CSI (the base edge features) identical...
    assert torch.allclose(approach["ef"][:, :base_e], depart["ef"][:, :base_e])
    # ...but the motion-augmented observations DIFFER -> a memoryless actor can tell them apart.
    assert not torch.allclose(approach["ef"], depart["ef"])


def test_critic_can_read_training_only_velocity() -> None:
    on = _scene((NodeMotion("veh_0", (10.0, 0.0, 0.0)),), True)
    obs = on.observation(0, [])
    nf, ef, ei = obs["nf"], obs["ef"], obs["ei"]
    assert nf.shape[1] >= 13                 # 8 base node features + 5 motion -> critic sees all velocities
    mean, std = feature_standardization([obs])
    critic = CentralizedGraphCritic(nf.shape[1], ef.shape[1], hidden=16, rounds=2)
    v = critic_scene_value(critic, nf, ef, ei, node_mean=mean[0], node_std=std[0],
                           edge_mean=mean[1], edge_std=std[1])
    assert v == v                            # finite value (the critic ingests the motion features)


def test_actor_motion_features_are_local_no_global_fields() -> None:
    # changing a NON-incident node's velocity must not change another node/edge's motion features.
    nodes = (
        Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
        Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.2 * _R, 0.0, 1.5)),
        Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.2 * _R, 1.5)),
        Node3D("veh_2", NodeKind.VEHICLE, Point3D(0.3 * _R, 0.3 * _R, 1.5)),
    )
    s1 = _scene((NodeMotion("veh_2", (20.0, 0.0, 0.0)),), True, nodes=nodes)
    s2 = _scene((NodeMotion("veh_2", (-20.0, 0.0, 0.0)),), True, nodes=nodes)
    # veh_0 has zero velocity in both -> its node motion features are identical regardless of veh_2.
    assert torch.allclose(_node_row(s1, "veh_0"), _node_row(s2, "veh_0"))
    # the veh_0<->veh_1 edge does not involve veh_2 -> its motion features are identical too.
    assert torch.allclose(_edge_row(s1, "veh_0", "veh_1"), _edge_row(s2, "veh_0", "veh_1"))
