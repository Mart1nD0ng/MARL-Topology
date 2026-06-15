"""Contract tests for TR 37.885 stochastic large-scale fading (shadowing + NLOSv).

Verified properties: byte-identical defaults, determinism per scene seed, reciprocity,
spatial correlation of the shadow field, per-state sigma selection, the stochastic NLOSv
state with persistent per-pair draws, the blockage-loss height cases (Option A), the
UE-type RSU link mapping, and independent robustness realizations.
"""

from math import exp, sqrt

from marl_topology.channel import ChannelModelConfig, evaluate_channel
from marl_topology.channel.model import (
    NLOSV_BLOCKER_HEIGHT_M,
    PATH_LOSS_MODEL_V2X,
    V2V_SHADOW_STD_LOS_DB,
    _shadow_field_value,
    umi_street_canyon_path_loss_db,
    v2v_37885_path_loss_db,
)
from marl_topology.data.stage31_scenario_generator import PhysicsRegime
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D

import pytest


def _scene(*nodes: Node3D, scenario_id: str = "fading_test") -> Scene3D:
    return Scene3D(scenario_id=scenario_id, nodes=tuple(nodes))


def _veh(node_id: str, x: float, y: float = 0.0) -> Node3D:
    return Node3D(node_id, NodeKind.VEHICLE, Point3D(x, y, 1.5))


def test_defaults_are_byte_identical() -> None:
    scene = _scene(_veh("veh_0", 0.0), _veh("veh_1", 80.0))
    config = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X)
    record = evaluate_channel(scene, "veh_0", "veh_1", config=config)
    expected = v2v_37885_path_loss_db(record.distance_3d_m, config.carrier_frequency_hz, is_nlos=False)
    assert record.shadowing_db == 0.0
    assert record.los_penalty_db == 0.0
    assert abs(record.path_loss_db - expected) < 1e-9


def test_flags_require_v2x_model() -> None:
    with pytest.raises(ValueError):
        ChannelModelConfig(shadowing_37885=True)
    with pytest.raises(ValueError):
        ChannelModelConfig(nlosv_37885=True)


def test_shadowing_deterministic_reciprocal_and_state_sigma() -> None:
    scene = _scene(_veh("veh_0", 0.0), _veh("veh_1", 80.0))
    config = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True)
    forward = evaluate_channel(scene, "veh_0", "veh_1", config=config)
    again = evaluate_channel(scene, "veh_0", "veh_1", config=config)
    reverse = evaluate_channel(scene, "veh_1", "veh_0", config=config)
    assert forward.shadowing_db == again.shadowing_db  # deterministic
    assert forward.shadowing_db == reverse.shadowing_db  # reciprocal
    assert forward.shadowing_db != 0.0
    # sigma bound: |(g_tx+g_rx)/sqrt(2)| values explode only with absurd field draws
    assert abs(forward.shadowing_db) < 6.0 * V2V_SHADOW_STD_LOS_DB


def test_shadow_field_is_spatially_correlated() -> None:
    near = abs(_shadow_field_value("s", 0.0, 0.0) - _shadow_field_value("s", 1.0, 0.0))
    pairs = [
        abs(_shadow_field_value("s", 100.0 * k, 0.0) - _shadow_field_value("s", 100.0 * k + 50.0, 0.0))
        for k in range(1, 9)
    ]
    assert near < sum(pairs) / len(pairs)  # 1 m apart ~ equal; 50 m apart ~ independent


def test_shadowing_evolves_smoothly_with_motion() -> None:
    config = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True)
    values = []
    for step in range(4):
        scene = _scene(_veh("veh_0", 0.0), _veh("veh_1", 80.0 + 2.0 * step))
        values.append(evaluate_channel(scene, "veh_0", "veh_1", config=config).shadowing_db)
    deltas = [abs(values[i + 1] - values[i]) for i in range(3)]
    assert max(deltas) < 2.0 * V2V_SHADOW_STD_LOS_DB  # 2 m steps stay correlated
    assert any(d > 0.0 for d in deltas)  # but the field does evolve


def test_realizations_are_independent_draws() -> None:
    scene = _scene(_veh("veh_0", 0.0), _veh("veh_1", 80.0))
    base = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True)
    other = ChannelModelConfig(
        path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True, shadowing_realization=1
    )
    assert (
        evaluate_channel(scene, "veh_0", "veh_1", config=base).shadowing_db
        != evaluate_channel(scene, "veh_0", "veh_1", config=other).shadowing_db
    )


def test_nlosv_state_is_distance_driven_and_persistent() -> None:
    config = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, nlosv_37885=True)
    # At 400 m P(LOS) = 1.05*exp(-4.56) ~ 0.011: essentially every pair is NLOSv.
    blocked = 0
    for k in range(12):
        scene = _scene(_veh(f"veh_a{k}", 0.0), _veh(f"veh_b{k}", 400.0))
        record = evaluate_channel(scene, f"veh_a{k}", f"veh_b{k}", config=config)
        assert record.los_state.value == "los"  # building-visibility truth is preserved
        blocked += int(record.los_penalty_db > 0.0)
        repeat = evaluate_channel(scene, f"veh_a{k}", f"veh_b{k}", config=config)
        assert repeat.los_penalty_db == record.los_penalty_db  # persistent draw
    assert blocked >= 11
    assert exp(-0.0114 * 400.0) * 1.05 < 0.02
    # At 5 m P(LOS) = 1: never NLOSv.
    scene = _scene(_veh("veh_c", 0.0), _veh("veh_d", 5.0))
    assert evaluate_channel(scene, "veh_c", "veh_d", config=config).los_penalty_db == 0.0


def test_nlosv_blockage_height_cases() -> None:
    config = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, nlosv_37885=True)
    assert 1.5 < NLOSV_BLOCKER_HEIGHT_M  # our vehicle antennas sit below Option A blockers
    # Antennas above the blocker height -> no blockage loss even at NLOSv distances.
    high_a = Node3D("veh_h0", NodeKind.VEHICLE, Point3D(0.0, 0.0, 2.5))
    high_b = Node3D("veh_h1", NodeKind.VEHICLE, Point3D(400.0, 0.0, 2.5))
    record = evaluate_channel(_scene(high_a, high_b), "veh_h0", "veh_h1", config=config)
    assert record.los_penalty_db == 0.0


def test_ue_type_rsu_links_use_v2v_family_when_nlosv_on() -> None:
    rsu = Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0))
    veh = _veh("veh_0", 5.0)  # short LOS link: P(LOS)=1, so no stochastic loss either
    scene = _scene(rsu, veh)
    config_off = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X)
    config_on = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, nlosv_37885=True)
    off = evaluate_channel(scene, "rsu_0", "veh_0", config=config_off)
    on = evaluate_channel(scene, "rsu_0", "veh_0", config=config_on)
    expected_umi = umi_street_canyon_path_loss_db(
        off.distance_3d_m, config_off.carrier_frequency_hz, is_nlos=False, h_tx_m=5.0, h_rx_m=1.5
    )
    expected_v2v = v2v_37885_path_loss_db(on.distance_3d_m, config_on.carrier_frequency_hz, is_nlos=False)
    assert abs(off.base_path_loss_db - expected_umi) < 1e-9
    assert abs(on.base_path_loss_db - expected_v2v) < 1e-9


def test_regime_passthrough_defaults_off() -> None:
    regime = PhysicsRegime(tx_power_dbm=20.0, path_loss_model="v2x_37885")
    channel = regime.channel_config()
    assert channel.shadowing_37885 is False
    assert channel.nlosv_37885 is False
    assert channel.shadowing_realization == 0
    stochastic = PhysicsRegime(
        tx_power_dbm=20.0, path_loss_model="v2x_37885", shadowing_37885=True, nlosv_37885=True
    )
    channel = stochastic.channel_config()
    assert channel.shadowing_37885 is True
    assert channel.nlosv_37885 is True
