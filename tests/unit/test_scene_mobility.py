"""Env mobility primitives (Part A1 of the temporal-actor work): NodeMotion and
advance_scene. Pure, deterministic constant-velocity position integration that turns
a rollout step into a real time step -- so a temporal actor has a predictable
link-evolution signal to learn from. Touches nothing in the training loop yet.
"""

import pytest

from marl_topology.geometry3d import Point3D
from marl_topology.scenario import NodeMotion, advance_scene  # re-exported from package
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D


def _scene() -> Scene3D:
    return Scene3D(
        scenario_id="mob_test",
        nodes=(
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(10.0, 0.0, 1.5)),
        ),
    )


def test_vehicle_advances_by_v_times_dt() -> None:
    out = advance_scene(_scene(), {"veh_0": NodeMotion("veh_0", (2.0, -1.0, 0.0))}, dt_s=3.0)
    veh = out.get_node("veh_0").position
    assert veh.x_m == pytest.approx(16.0)  # 10 + 2*3
    assert veh.y_m == pytest.approx(-3.0)  # 0 + (-1)*3
    assert veh.z_m == pytest.approx(1.5)  # vz=0


def test_rsu_and_stationary_nodes_stay_put() -> None:
    scene = _scene()
    # RSU has no motion entry; veh_0 has an explicitly stationary motion.
    out = advance_scene(scene, {"veh_0": NodeMotion("veh_0", (0.0, 0.0, 0.0))}, dt_s=5.0)
    assert out.get_node("rsu_0").position == scene.get_node("rsu_0").position
    assert out.get_node("veh_0").position == scene.get_node("veh_0").position


def test_advance_is_pure_input_unchanged() -> None:
    scene = _scene()
    before = scene.get_node("veh_0").position
    advance_scene(scene, {"veh_0": NodeMotion("veh_0", (1.0, 0.0, 0.0))}, dt_s=1.0)
    assert scene.get_node("veh_0").position == before  # original scene untouched


def test_links_change_predictably_over_time() -> None:
    # The whole point of Part A1: as the vehicle drives toward the RSU, the 3D link
    # distance shrinks monotonically -- the predictable signal a temporal actor learns.
    scene = _scene()
    motion = {"veh_0": NodeMotion("veh_0", (-2.0, 0.0, 0.0))}  # toward x=0 (under the RSU)
    distances = []
    frame = scene
    for _ in range(4):
        frame = advance_scene(frame, motion, dt_s=1.0)
        veh = frame.get_node("veh_0").position
        rsu = frame.get_node("rsu_0").position
        distances.append(veh.distance_to(rsu))
    assert distances == sorted(distances, reverse=True)  # strictly shrinking
    assert distances[0] > distances[-1]


def test_determinism() -> None:
    scene = _scene()
    motions = {"veh_0": NodeMotion("veh_0", (1.5, 0.5, 0.0))}
    a = advance_scene(scene, motions, 2.0).get_node("veh_0").position
    b = advance_scene(scene, motions, 2.0).get_node("veh_0").position
    assert a == b


def test_unknown_motion_keys_are_ignored() -> None:
    scene = _scene()
    out = advance_scene(scene, {"ghost": NodeMotion("ghost", (9.0, 9.0, 9.0))}, dt_s=1.0)
    assert out.get_node("veh_0").position == scene.get_node("veh_0").position


def test_dt_must_be_positive() -> None:
    scene = _scene()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            advance_scene(scene, {}, dt_s=bad)


def test_node_motion_validation_and_stationary_flag() -> None:
    with pytest.raises(ValueError):
        NodeMotion("", (1.0, 0.0, 0.0))  # empty node_id
    with pytest.raises(ValueError):
        NodeMotion("v", (1.0, 0.0))  # wrong-length velocity
    assert NodeMotion("v", (0.0, 0.0, 0.0)).is_stationary is True
    assert NodeMotion("v", (1.0, 0.0, 0.0)).is_stationary is False
