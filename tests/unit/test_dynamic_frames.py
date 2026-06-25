"""Dynamic multi-frame channel data (two-timescale substrate, Spec S3.3-3.5).

Pins the invariants the dynamic episode rollout relies on:

  * the node SET and the all-pairs candidate edge-id SET are INVARIANT across frames (vehicles
    move but never appear/disappear; edges are all node pairs) -> the action space, and hence the
    reconfiguration cost |E_t triangle E_{t-1}|, is well defined frame-to-frame;
  * the per-frame CHANNEL evolves with vehicle motion (a vehicle driving out of range lowers its
    link success probabilities) -> a genuine, PREDICTABLE temporal signal (unlike i.i.d. fading);
  * the observation encodes the current channel (edge features), the previous topology
    (previous_selected_edges -> a per-edge flag + per-node previous degree), and the step index;
  * frame 0 reproduces the static scene exactly (the T=1 limit).

Uses a controlled scene (one vehicle drives radially out of range) so the temporal signal is
deterministic, not dataset-dependent.
"""

from __future__ import annotations

import torch

from marl_topology.data.stage31_scenario_generator import (
    PhysicsRegime,
    measure_reliable_range_m,
    roll_trajectory,
)
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, NodeMotion, Scene3D
from marl_topology.training.dynamic_frames import DynamicScene, dynamic_scene_from_trajectory


def _controlled_trajectory(num_frames: int = 5):
    """A feasible cluster around an RSU; veh_2 drives radially out of range (links degrade)."""
    regime = PhysicsRegime()
    r = measure_reliable_range_m(regime)
    scene = Scene3D(
        scenario_id="dyn_ctrl",
        nodes=(
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.20 * r, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.20 * r, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(0.20 * r, 0.20 * r, 1.5)),
        ),
    )
    motions = (NodeMotion("veh_2", (0.5 * r, 0.0, 0.0)),)   # veh_2 drives out
    return roll_trajectory(scene, motions, regime, quorum_size=3, tau=0.9,
                           num_frames=num_frames, dt_s=1.0)


def test_node_and_edge_id_sets_invariant_across_frames() -> None:
    dyn = dynamic_scene_from_trajectory(_controlled_trajectory())
    assert isinstance(dyn, DynamicScene)
    node_sets = {tuple(dyn.context(t).graph.node_ids) for t in range(dyn.n_frames)}
    edge_sets = {tuple(sorted(dyn.edge_ids_at(t))) for t in range(dyn.n_frames)}
    assert len(node_sets) == 1, "node set must be invariant (vehicles move, never appear/disappear)"
    assert len(edge_sets) == 1, "all-pairs candidate edge-id set must be invariant -> action space fixed"
    # the shared, frame-invariant action space
    assert tuple(sorted(dyn.edge_ids)) == next(iter(edge_sets))


def test_channel_evolves_with_motion() -> None:
    # a vehicle driving out of range must lower at least one incident link's success probability;
    # the per-frame edge features (link_success_probability) therefore change across frames.
    dyn = dynamic_scene_from_trajectory(_controlled_trajectory())
    s0 = dyn.observation(0, previous_selected_edges=[])
    sl = dyn.observation(dyn.n_frames - 1, previous_selected_edges=[])
    link_p0 = s0["ef"][:, 0]
    link_pl = sl["ef"][:, 0]
    assert not torch.allclose(link_p0, link_pl), "channel must evolve as vehicles move (temporal signal)"
    # and it should DEGRADE somewhere (a link left the reliable range): some edge's prob drops.
    assert float((link_pl - link_p0).min()) < -1e-6


def test_observation_encodes_previous_topology_and_step() -> None:
    dyn = dynamic_scene_from_trajectory(_controlled_trajectory())
    edge_ids = dyn.edge_ids
    prev = [edge_ids[0], edge_ids[2]]
    s = dyn.observation(3, previous_selected_edges=prev)
    # the previous-edge indicator is edge feature index 4 (graph_payload schema)
    prev_flag = s["ef"][:, 4]
    on = {edge_ids[i] for i in range(len(edge_ids)) if float(prev_flag[i]) > 0.5}
    assert on == set(prev), "previous-topology flag must mark exactly the previously selected edges"
    # step index is node feature index 7
    assert torch.allclose(s["nf"][:, 7], torch.full((s["nf"].shape[0],), 3.0))
    # at t=0 with no history the flag is all-zero
    s0 = dyn.observation(0, previous_selected_edges=[])
    assert float(s0["ef"][:, 4].abs().max()) == 0.0


def test_frame0_reproduces_static_scene_evaluation() -> None:
    traj = _controlled_trajectory()
    dyn = dynamic_scene_from_trajectory(traj)
    ctx0 = dyn.context(0)
    full = [e.edge_id for e in ctx0.graph.edges]
    c = float(ctx0.evaluator.evaluate(set(full)).metrics["consensus_success_probability"])
    assert abs(c - float(traj.frames[0].measurement["full_graph_psucc"])) < 1e-6


def test_reconfig_cost_well_defined_across_frames() -> None:
    # because edge ids are frame-invariant, any per-frame topology pair has a symmetric-difference
    # reconfiguration cost (the two_timescale_env semantics).
    dyn = dynamic_scene_from_trajectory(_controlled_trajectory())
    edge_ids = dyn.edge_ids
    a = frozenset(edge_ids[:3])
    b = frozenset(edge_ids[1:4])
    assert dyn.reconfig.cost(a, b) == (dyn.reconfig.e_edge + dyn.reconfig.l_edge) * len(a ^ b)
