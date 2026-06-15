import pytest

from marl_topology.channel import (
    ChannelModelConfig,
    STAGE3_CHANNEL_REGIME_ID,
    evaluate_channel,
    free_space_path_loss_db,
    get_channel_model_fixture,
    iter_channel_model_fixtures,
    noise_power_dbm,
)
from marl_topology.geometry3d import LoSState
from marl_topology.scenario import Scene3D


def test_channel_fixtures_produce_expected_interference_sets() -> None:
    for fixture in iter_channel_model_fixtures():
        record = evaluate_channel(
            fixture.scene,
            fixture.tx_id,
            fixture.rx_id,
            active_transmissions=fixture.active_transmissions,
            resource_id=fixture.resource_id,
        )

        assert record.scenario_id == fixture.fixture_id
        assert record.interference_tx_ids == fixture.expected_interference_tx_ids
        assert record.channel_model_id == STAGE3_CHANNEL_REGIME_ID


def test_path_loss_increases_with_distance_under_free_space_los() -> None:
    close = get_channel_model_fixture("free_space_close")
    far = get_channel_model_fixture("free_space_far")

    close_record = evaluate_channel(close.scene, close.tx_id, close.rx_id)
    far_record = evaluate_channel(far.scene, far.tx_id, far.rx_id)

    assert far_record.distance_3d_m > close_record.distance_3d_m
    assert far_record.path_loss_db > close_record.path_loss_db
    assert far_record.rx_power_dbm < close_record.rx_power_dbm
    assert far_record.sinr_db < close_record.sinr_db


def test_nlos_is_worse_than_los_for_matching_nodes_and_power() -> None:
    nlos_fixture = get_channel_model_fixture("blocked_by_building")
    nlos_record = evaluate_channel(nlos_fixture.scene, nlos_fixture.tx_id, nlos_fixture.rx_id)
    los_scene = Scene3D(
        scenario_id="blocked_by_building_los_counterfactual",
        nodes=nlos_fixture.scene.nodes,
        roads=nlos_fixture.scene.roads,
        lanes=nlos_fixture.scene.lanes,
        physics_regime=nlos_fixture.scene.physics_regime,
    )
    los_record = evaluate_channel(los_scene, nlos_fixture.tx_id, nlos_fixture.rx_id)

    assert los_record.los_state == LoSState.LOS
    assert nlos_record.los_state == LoSState.NLOS
    assert nlos_record.path_loss_db > los_record.path_loss_db
    assert nlos_record.rx_power_dbm < los_record.rx_power_dbm
    assert nlos_record.sinr_db < los_record.sinr_db


def test_tx_power_increase_raises_rx_power_and_sinr() -> None:
    fixture = get_channel_model_fixture("free_space_close")

    low_power = evaluate_channel(fixture.scene, fixture.tx_id, fixture.rx_id, tx_power_dbm=10.0)
    high_power = evaluate_channel(fixture.scene, fixture.tx_id, fixture.rx_id, tx_power_dbm=25.0)

    assert high_power.rx_power_dbm > low_power.rx_power_dbm
    assert high_power.sinr_db > low_power.sinr_db


def test_same_resource_interference_lowers_sinr() -> None:
    fixture = get_channel_model_fixture("two_transmitters_interference")

    clean = evaluate_channel(fixture.scene, fixture.tx_id, fixture.rx_id)
    interfered = evaluate_channel(
        fixture.scene,
        fixture.tx_id,
        fixture.rx_id,
        active_transmissions=fixture.active_transmissions,
        resource_id=fixture.resource_id,
    )

    assert interfered.interference_tx_ids == ("rsu_1",)
    assert interfered.interference_power_dbm > clean.interference_power_dbm
    assert interfered.sinr_db < clean.sinr_db


def test_orthogonal_resource_transmission_does_not_interfere() -> None:
    fixture = get_channel_model_fixture("orthogonal_channel_no_interference")
    config = ChannelModelConfig()

    clean = evaluate_channel(fixture.scene, fixture.tx_id, fixture.rx_id, config=config)
    orthogonal = evaluate_channel(
        fixture.scene,
        fixture.tx_id,
        fixture.rx_id,
        config=config,
        active_transmissions=fixture.active_transmissions,
        resource_id=fixture.resource_id,
    )

    assert orthogonal.interference_tx_ids == ()
    assert orthogonal.interference_power_dbm == config.no_interference_power_dbm
    assert orthogonal.sinr_db == clean.sinr_db


def test_shadowing_requires_seed_and_is_deterministic_when_seeded() -> None:
    fixture = get_channel_model_fixture("free_space_close")
    config = ChannelModelConfig(shadowing_std_db=2.0)

    with pytest.raises(ValueError, match="shadowing_seed"):
        evaluate_channel(fixture.scene, fixture.tx_id, fixture.rx_id, config=config)

    first = evaluate_channel(
        fixture.scene,
        fixture.tx_id,
        fixture.rx_id,
        config=config,
        shadowing_seed=123,
    )
    second = evaluate_channel(
        fixture.scene,
        fixture.tx_id,
        fixture.rx_id,
        config=config,
        shadowing_seed=123,
    )

    assert first.shadowing_seed == 123
    assert first.shadowing_db == second.shadowing_db
    assert first.path_loss_db == second.path_loss_db


def test_power_helpers_and_noise_are_monotonic() -> None:
    assert free_space_path_loss_db(20.0, 5.9e9) > free_space_path_loss_db(10.0, 5.9e9)
    assert noise_power_dbm(20e6, 7.0) > noise_power_dbm(10e6, 7.0)
