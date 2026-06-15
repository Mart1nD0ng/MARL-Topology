"""Contract tests for the 3GPP UMi path-loss variant and the STDMA scheduler.

Covers two audit-flagged gaps: (1) the new opt-in path_loss_model must keep the default
byte-identical and the UMi variant physically sane; (2) the STDMA scheduler previously had
no dedicated unit test for its core invariants."""

from random import Random

from marl_topology.channel import (
    ChannelModelConfig,
    PATH_LOSS_MODEL_FSPL,
    PATH_LOSS_MODEL_UMI,
    evaluate_channel,
    umi_street_canyon_path_loss_db,
)
from marl_topology.geometry3d import Point3D
from marl_topology.protocol import StdmaScheduleConfig, build_stdma_schedule
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D
from marl_topology.scenario.urban_grid import UrbanGridConfig, build_urban_grid_scene
from marl_topology.topology import CandidateGraph


def _line_scene() -> Scene3D:
    nodes = (
        Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
        Node3D("veh_0", NodeKind.VEHICLE, Point3D(80.0, 0.0, 1.5)),
        Node3D("veh_1", NodeKind.VEHICLE, Point3D(160.0, 0.0, 1.5)),
    )
    return Scene3D(scenario_id="umi_test", nodes=nodes)


def test_default_path_loss_model_is_fspl_and_unchanged() -> None:
    scene = _line_scene()
    default_config = ChannelModelConfig()
    explicit_fspl = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_FSPL)
    rec_default = evaluate_channel(scene, "rsu_0", "veh_0", config=default_config)
    rec_fspl = evaluate_channel(scene, "rsu_0", "veh_0", config=explicit_fspl)
    assert default_config.path_loss_model == PATH_LOSS_MODEL_FSPL
    assert rec_default.path_loss_db == rec_fspl.path_loss_db
    assert rec_default.sinr_db == rec_fspl.sinr_db


def test_umi_variant_changes_path_loss_and_is_monotone_in_distance() -> None:
    scene = _line_scene()
    umi = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_UMI)
    near = evaluate_channel(scene, "rsu_0", "veh_0", config=umi)
    far = evaluate_channel(scene, "rsu_0", "veh_1", config=umi)
    assert far.path_loss_db > near.path_loss_db
    fspl = evaluate_channel(scene, "rsu_0", "veh_0", config=ChannelModelConfig())
    assert near.path_loss_db != fspl.path_loss_db


def test_umi_nlos_at_least_los_and_distance_dependent() -> None:
    common = dict(carrier_frequency_hz=5.9e9, h_tx_m=5.0, h_rx_m=1.5)
    for d in (30.0, 100.0, 300.0):
        los = umi_street_canyon_path_loss_db(d, is_nlos=False, **common)
        nlos = umi_street_canyon_path_loss_db(d, is_nlos=True, **common)
        assert nlos >= los
    gap_near = (
        umi_street_canyon_path_loss_db(50.0, is_nlos=True, **common)
        - umi_street_canyon_path_loss_db(50.0, is_nlos=False, **common)
    )
    gap_far = (
        umi_street_canyon_path_loss_db(400.0, is_nlos=True, **common)
        - umi_street_canyon_path_loss_db(400.0, is_nlos=False, **common)
    )
    assert gap_near != gap_far  # distance-dependent NLOS exponent, not a flat penalty


def test_stdma_schedule_core_invariants() -> None:
    grid = UrbanGridConfig(
        blocks_per_side=2, block_size_m=150.0, street_width_m=24.0, rsu_count=1, vehicle_count=5
    )
    scene = build_urban_grid_scene("stdma_test", grid, Random(3))
    graph = CandidateGraph.from_scene(scene, max_distance_m=None)
    selected = tuple(sorted(graph.edge_ids))[:8]
    schedule = build_stdma_schedule(
        scene=scene,
        graph=graph,
        selected_edge_ids=selected,
        channel_config=ChannelModelConfig(default_tx_power_dbm=20.0, bandwidth_hz=20e6),
        config=StdmaScheduleConfig(),
    )
    # every selected edge gets exactly one slot, slots are contiguous from 0
    assert set(schedule.slot_of_edge) == set(selected)
    assert schedule.num_slots >= 1
    assert max(schedule.slot_of_edge.values()) + 1 == schedule.num_slots
    # half-duplex: co-slot edges never share a node
    by_slot: dict[int, list[str]] = {}
    for edge_id, slot in schedule.slot_of_edge.items():
        by_slot.setdefault(slot, []).append(edge_id)
    for members in by_slot.values():
        seen_nodes: set[str] = set()
        for edge_id in members:
            edge = graph.get_edge(edge_id)
            assert edge.node_u not in seen_nodes and edge.node_v not in seen_nodes
            seen_nodes.update((edge.node_u, edge.node_v))
    # slot-0 routes incur zero waiting latency; later slots wait k * slot_duration
    slot0_edge = next(e for e, s in schedule.slot_of_edge.items() if s == 0)
    assert schedule.route_schedule_latency_s((slot0_edge,)) == 0.0
    if schedule.num_slots > 1:
        late_edge = next(e for e, s in schedule.slot_of_edge.items() if s == schedule.num_slots - 1)
        expected = (schedule.num_slots - 1) * schedule.slot_duration_s
        assert schedule.route_schedule_latency_s((late_edge,)) == expected


def test_stdma_empty_selection_yields_empty_schedule() -> None:
    grid = UrbanGridConfig(
        blocks_per_side=2, block_size_m=150.0, street_width_m=24.0, rsu_count=1, vehicle_count=3
    )
    scene = build_urban_grid_scene("stdma_empty", grid, Random(4))
    graph = CandidateGraph.from_scene(scene, max_distance_m=None)
    schedule = build_stdma_schedule(
        scene=scene,
        graph=graph,
        selected_edge_ids=(),
        channel_config=ChannelModelConfig(),
        config=StdmaScheduleConfig(),
    )
    assert schedule.num_slots == 0
    assert schedule.slot_of_edge == {}
