"""Trajectory generation (Part A2): the same scene evolved over time, measured per
frame on the real Stage 21 stack. Additive -- the static-scene generator is unchanged.
"""

import pytest

from marl_topology.data.stage31_scenario_generator import (
    PhysicsRegime,
    ProductionScenarioConfig,
    ProductionTrajectorySpec,
    generate_production_trajectories,
    measure_reliable_range_m,
    roll_trajectory,
)
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, NodeMotion, Scene3D


def test_roll_trajectory_structure_and_evolving_feasibility() -> None:
    regime = PhysicsRegime()
    r = measure_reliable_range_m(regime)
    # A feasible cluster around the RSU; veh_2 drives radially out of range.
    scene = Scene3D(
        scenario_id="traj_ctrl",
        nodes=(
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.25 * r, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.25 * r, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(0.25 * r, 0.25 * r, 1.5)),
        ),
    )
    motions = (NodeMotion("veh_2", (0.4 * r, 0.0, 0.0)),)
    traj = roll_trajectory(scene, motions, regime, quorum_size=3, tau=0.9, num_frames=4, dt_s=1.0)

    # structure
    assert isinstance(traj, ProductionTrajectorySpec)
    assert traj.num_frames == 4
    assert [f.time_index for f in traj.frames] == [0, 1, 2, 3]
    assert [f.time_s for f in traj.frames] == [0.0, 1.0, 2.0, 3.0]
    assert traj.sequence_id == "traj_ctrl"

    # frame 0 is the initial scene; RSU is fixed; veh_2 moves monotonically out
    assert traj.frames[0].scene.get_node("veh_2").position == scene.get_node("veh_2").position
    assert len({f.scene.get_node("rsu_0").position for f in traj.frames}) == 1
    veh2_x = [f.scene.get_node("veh_2").position.x_m for f in traj.frames]
    assert veh2_x == sorted(veh2_x) and veh2_x[0] < veh2_x[-1]

    # the temporal signal: the per-frame measurement is NOT constant -- the geometry
    # change is reflected in measured feasibility, and full-graph reliability degrades
    # as veh_2 leaves range.
    assert traj.frames[0].measurement != traj.frames[-1].measurement
    assert traj.frames[0].measurement["full_graph_psucc"] >= traj.frames[-1].measurement["full_graph_psucc"]


def test_roll_trajectory_rejects_bad_inputs() -> None:
    regime = PhysicsRegime()
    scene = Scene3D(
        scenario_id="t",
        nodes=(
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(10.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 10.0, 1.5)),
        ),
    )
    with pytest.raises(Exception):
        roll_trajectory(scene, (), regime, 3, 0.9, num_frames=0, dt_s=1.0)


def test_generate_production_trajectories_deterministic_and_moving() -> None:
    cfg = ProductionScenarioConfig(seed=31, scenario_count=4)
    a = generate_production_trajectories(cfg, num_frames=3, dt_s=1.0)
    b = generate_production_trajectories(cfg, num_frames=3, dt_s=1.0)

    assert len(a) == 4
    assert all(t.num_frames == 3 for t in a)
    # deterministic given the seed (same scenes + same sampled velocities)
    assert [t.sequence_id for t in a] == [t.sequence_id for t in b]
    assert [t.motions for t in a] == [t.motions for t in b]

    # every trajectory has at least one vehicle that actually moved across its window
    for traj in a:
        first, last = traj.frames[0].scene, traj.frames[-1].scene
        moved = any(
            first.get_node(nid).position != last.get_node(nid).position
            for nid in first.node_ids
            if first.get_node(nid).kind is NodeKind.VEHICLE
        )
        assert moved
        # the RSU never moves
        assert first.get_node("rsu_0").position == last.get_node("rsu_0").position
