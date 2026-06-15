"""Contract tests for the TR 37.885 V2X path-loss variant and the wired RSU backhaul."""

from random import Random

from marl_topology.channel import (
    ChannelModelConfig,
    PATH_LOSS_MODEL_V2X,
    evaluate_channel,
    umi_street_canyon_path_loss_db,
    v2v_37885_path_loss_db,
)
from marl_topology.data.stage21_objective_stack_evidence import (
    Stage21ObjectiveStackConfig,
    Stage21ObjectiveStackEvaluator,
)
from marl_topology.data.stage31_scenario_generator import PhysicsRegime, build_stack_config
from marl_topology.geometry3d import Point3D
from marl_topology.protocol import PBFTPhaseBudgets, build_pbft_message_matrices_from_network_records
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D
from marl_topology.scenario.urban_grid import UrbanGridConfig, build_urban_grid_scene
from marl_topology.topology import CandidateGraph


def _mixed_scene() -> Scene3D:
    nodes = (
        Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
        Node3D("veh_0", NodeKind.VEHICLE, Point3D(100.0, 0.0, 1.5)),
        Node3D("veh_1", NodeKind.VEHICLE, Point3D(200.0, 0.0, 1.5)),
    )
    return Scene3D(scenario_id="v2x_test", nodes=nodes)


def test_v2x_variant_uses_v2v_for_vehicle_pairs_and_umi_for_v2i() -> None:
    scene = _mixed_scene()
    config = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X)
    v2v = evaluate_channel(scene, "veh_0", "veh_1", config=config)
    expected_v2v = v2v_37885_path_loss_db(v2v.distance_3d_m, config.carrier_frequency_hz, is_nlos=False)
    assert abs(v2v.path_loss_db - expected_v2v) < 1e-9
    v2i = evaluate_channel(scene, "rsu_0", "veh_0", config=config)
    expected_v2i = umi_street_canyon_path_loss_db(
        v2i.distance_3d_m, config.carrier_frequency_hz, is_nlos=False, h_tx_m=5.0, h_rx_m=1.5
    )
    assert abs(v2i.path_loss_db - expected_v2i) < 1e-9


def test_v2v_37885_nlos_harsher_and_monotone() -> None:
    for d in (50.0, 150.0, 400.0):
        los = v2v_37885_path_loss_db(d, 5.9e9, is_nlos=False)
        nlos = v2v_37885_path_loss_db(d, 5.9e9, is_nlos=True)
        assert nlos > los
    assert v2v_37885_path_loss_db(300.0, 5.9e9, is_nlos=False) > v2v_37885_path_loss_db(
        100.0, 5.9e9, is_nlos=False
    )


def test_perfect_pairs_feed_multi_hop_reach() -> None:
    # No network records at all: only the wired pair delivers, and with relay_hops=2 a
    # node-pair path THROUGH the wire is still impossible (no wireless hops exist), so the
    # matrices contain exactly the wired entries.
    node_ids = ("rsu_0", "rsu_1", "veh_0")
    budgets = PBFTPhaseBudgets(0.01, 0.01, 0.01)
    pairs = frozenset({("rsu_0", "rsu_1"), ("rsu_1", "rsu_0")})
    matrices = build_pbft_message_matrices_from_network_records(
        node_ids, {}, budgets, relay_hops=2, perfect_pairs=pairs
    )
    assert matrices.pre_prepare_matrix[("rsu_0", "rsu_1")] == 1.0
    assert matrices.pre_prepare_matrix[("rsu_1", "rsu_0")] == 1.0
    assert ("rsu_0", "veh_0") not in matrices.pre_prepare_matrix


def test_wired_backhaul_lifts_multi_rsu_consensus() -> None:
    grid = UrbanGridConfig(
        blocks_per_side=2, block_size_m=150.0, street_width_m=24.0, rsu_count=3, vehicle_count=5
    )
    scene = build_urban_grid_scene("backhaul_test", grid, Random(5))
    graph = CandidateGraph.from_scene(scene, max_distance_m=None)
    regime_off = PhysicsRegime(
        tx_power_dbm=20.0, use_background_interference=True, orthogonal_resources=False,
        scheduled_mac=True, relay_hops=3, target_reliability=None,
    )
    config_off = build_stack_config(regime_off)
    config_on = Stage21ObjectiveStackConfig(
        channel_config=config_off.channel_config,
        link_config=config_off.link_config,
        use_background_interference=config_off.use_background_interference,
        orthogonal_resources=config_off.orthogonal_resources,
        relay_hops=config_off.relay_hops,
        scheduled_mac=config_off.scheduled_mac,
        wired_rsu_backhaul=True,
    )
    selected = set(graph.edge_ids)  # full graph; only the backhaul differs
    p_off = float(
        Stage21ObjectiveStackEvaluator(scene=scene, graph=graph, config=config_off)
        .evaluate(selected).metrics["consensus_success_probability"]
    )
    p_on = float(
        Stage21ObjectiveStackEvaluator(scene=scene, graph=graph, config=config_on)
        .evaluate(selected).metrics["consensus_success_probability"]
    )
    assert p_on >= p_off  # a reliable backbone can only help


def test_wired_backhaul_default_off_is_unchanged() -> None:
    regime = PhysicsRegime(tx_power_dbm=20.0, scheduled_mac=True, relay_hops=2, target_reliability=None)
    config = build_stack_config(regime)
    assert config.wired_rsu_backhaul is False
