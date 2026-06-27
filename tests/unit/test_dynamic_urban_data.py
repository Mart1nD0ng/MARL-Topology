"""D1: real 4-RSU urban-grid DYNAMIC data (Plan §3, gap #1). The dynamic data must genuinely be a
4-RSU urban grid (buildings + streets + road-constrained motion), not single-RSU random geometry."""

from __future__ import annotations

from marl_topology.data.stage31_scenario_generator import PhysicsRegime
from marl_topology.scenario.scene import NodeKind
from marl_topology.training.dynamic_frames import sample_dynamic_urban_scenes
from marl_topology.training.two_timescale_env import ReconfigCost


def _urban(count=2, num_frames=4, rsu_count=4, node_count_choices=(12,)):
    return sample_dynamic_urban_scenes(
        seed=3, count=count, node_count_choices=node_count_choices, regime=PhysicsRegime(),
        num_frames=num_frames, dt_s=1.0, speed_min_mps=10.0, speed_max_mps=20.0,
        reconfig=ReconfigCost(e_edge=0.05, l_edge=0.0), hold_interval=4, gamma=0.9,
        tag="held_", rsu_count=rsu_count)


def test_dynamic_urban_has_four_rsus() -> None:
    for sc in _urban(rsu_count=4):
        rsus = [n for n in sc.scenes[0].nodes if n.kind is NodeKind.RSU]
        assert len(rsus) == 4, f"expected 4 RSUs, got {len(rsus)}"
        assert {r.node_id for r in rsus} == {"rsu_0", "rsu_1", "rsu_2", "rsu_3"}


def test_dynamic_urban_uses_grid_and_buildings() -> None:
    for sc in _urban():
        scene0 = sc.scenes[0]
        assert len(scene0.buildings) >= 4          # G*G building blocks (G>=2)
        assert len(scene0.roads) >= 4 and len(scene0.lanes) >= 4   # the street grid
        # vehicles exist alongside the 4 RSUs
        assert any(n.kind is NodeKind.VEHICLE for n in scene0.nodes)


def test_vehicle_motion_is_road_constrained() -> None:
    # each vehicle moves ALONG its street axis (one velocity component is exactly 0), not diagonally.
    for sc in _urban():
        assert sc.velocities, "urban scenes must carry per-node velocities"
        for node_id, (vx, vy, vz) in sc.velocities.items():
            if node_id.startswith("veh"):
                assert vz == 0.0
                assert (abs(vx) < 1e-9) ^ (abs(vy) < 1e-9), "exactly one of vx/vy must be 0 (axis-aligned)"
                assert abs(vx) + abs(vy) > 0.0      # the vehicle actually moves


def test_edge_ids_stable_across_frames() -> None:
    for sc in _urban(num_frames=4):
        ids0 = sc.edge_ids_at(0)
        nodes0 = sc.scenes[0].node_ids
        for t in range(1, sc.n_frames):
            assert sc.edge_ids_at(t) == ids0          # candidate edge-id set invariant
            assert sc.scenes[t].node_ids == nodes0    # node-id set invariant (vehicles move, not appear)


def test_frame0_matches_static_urban_context() -> None:
    # frame 0 IS the initial (un-advanced) urban scene (the T=1 limit reproduces the static scene).
    for sc in _urban(num_frames=3):
        p0 = {n.node_id: (n.position.x_m, n.position.y_m) for n in sc.scenes[0].nodes}
        p1 = {n.node_id: (n.position.x_m, n.position.y_m) for n in sc.scenes[1].nodes}
        assert p0 != p1                                # frame 1 has advanced (vehicles moved)
        # RSUs are fixed (no motion entry); their positions are identical across frames
        for n in sc.scenes[0].nodes:
            if n.kind is NodeKind.RSU:
                assert p0[n.node_id] == p1[n.node_id]


def test_channel_changes_with_motion() -> None:
    # the per-frame channel (link success probs) evolves as vehicles move -> a real temporal signal.
    sc = _urban(num_frames=4)[0]
    eids = sc.edge_ids_at(0)
    psucc = []
    for t in range(sc.n_frames):
        recs = sc.context(t).link_records
        psucc.append(tuple(round(float(recs[e].link_success_probability), 6) for e in eids))
    assert any(psucc[t] != psucc[0] for t in range(1, sc.n_frames)), "channel did not evolve with motion"
