from marl_topology.channel import ChannelModelConfig
from marl_topology.link import LinkTransmissionConfig
from marl_topology.network import (
    NetworkCommunicationConfig,
    evaluate_network_communication,
    get_network_communication_fixture,
    iter_network_communication_fixtures,
)


def test_network_fixtures_are_evaluable() -> None:
    for fixture in iter_network_communication_fixtures():
        record = fixture.evaluate()

        assert record.network_model_id == "stage3_network_communication_v1"
        assert 0.0 <= record.network_delivery_probability <= 1.0
        assert record.network_latency_s >= 0.0
        assert record.network_scheduled_latency_s >= record.network_successful_delivery_latency_s
        assert record.network_latency_s == record.network_successful_delivery_latency_s
        assert record.network_energy_j >= 0.0
        assert record.is_oracle is False


def test_multi_hop_route_aggregates_latency_energy_and_delivery() -> None:
    record = get_network_communication_fixture("multi_hop_delivery").evaluate()

    assert record.primitive == "route"
    assert record.hop_count == 2
    assert record.route_edge_ids == ("rsu_0--veh_0", "veh_0--veh_1")
    assert record.network_latency_s == sum(hop.p2p_latency_s for hop in record.hop_records)
    assert record.network_scheduled_latency_s == sum(
        hop.p2p_latency_s for hop in record.hop_records
    )
    assert record.network_successful_delivery_latency_s == record.network_latency_s
    assert record.network_energy_j == sum(hop.p2p_energy_j for hop in record.hop_records)
    expected_probability = 1.0
    for hop in record.hop_records:
        expected_probability *= hop.p2p_delivery_probability
    assert record.network_delivery_probability == expected_probability


def test_disconnected_topology_lowers_delivery_and_reachability() -> None:
    record = get_network_communication_fixture("disconnected_reachability").evaluate()

    assert "veh_2" not in record.reachable_node_ids
    assert record.hop_count == 0
    assert record.network_delivery_probability == 0.0
    assert record.network_latency_s == 0.0
    assert record.network_scheduled_latency_s == 0.0
    assert record.network_successful_delivery_latency_s == 0.0
    assert record.network_energy_j == 0.0


def test_same_resource_background_transmission_lowers_delivery() -> None:
    same = get_network_communication_fixture("two_transmitters_interference_network").evaluate()
    orthogonal = get_network_communication_fixture(
        "orthogonal_channel_no_interference_network"
    ).evaluate()

    assert same.interference_group_ids == ("resource_0:active_tx=2",)
    assert same.hop_records[0].interference_tx_ids == ("rsu_1",)
    assert orthogonal.interference_group_ids == ()
    assert orthogonal.hop_records[0].interference_tx_ids == ()
    assert same.network_delivery_probability < orthogonal.network_delivery_probability


def test_resource_redundant_topology_increases_energy_without_oracle_label() -> None:
    redundant = get_network_communication_fixture("resource_redundant_topology")
    redundant_record = redundant.evaluate()
    minimal_record = evaluate_network_communication(
        scene=redundant.scene,
        graph=redundant.candidate_graph(),
        selected_edge_ids=("rsu_0--veh_0", "veh_0--veh_1"),
        source_id=redundant.source_id,
        target_ids=redundant.target_ids,
        config=NetworkCommunicationConfig(primitive="broadcast"),
    )

    assert redundant_record.hop_count > minimal_record.hop_count
    assert redundant_record.network_energy_j > minimal_record.network_energy_j
    assert redundant_record.is_oracle is False
    assert minimal_record.is_oracle is False


def test_full_graph_is_baseline_not_oracle() -> None:
    fixture = get_network_communication_fixture("multi_hop_delivery")
    graph = fixture.candidate_graph()
    record = evaluate_network_communication(
        scene=fixture.scene,
        graph=graph,
        selected_edge_ids=graph.edge_ids,
        source_id=fixture.source_id,
        target_ids=fixture.target_ids,
    )

    assert record.is_full_graph_baseline is True
    assert record.is_oracle is False


def test_unselected_edges_do_not_generate_transmission_energy() -> None:
    fixture = get_network_communication_fixture("multi_hop_delivery")
    graph = fixture.candidate_graph()
    record = evaluate_network_communication(
        scene=fixture.scene,
        graph=graph,
        selected_edge_ids=(),
        source_id=fixture.source_id,
        target_ids=fixture.target_ids,
    )

    assert record.selected_edge_ids == ()
    assert record.active_transmission_ids == ()
    assert record.network_energy_j == 0.0
    assert record.network_delivery_probability == 0.0


def test_failed_scheduled_message_keeps_scheduled_latency_and_energy_visible() -> None:
    fixture = get_network_communication_fixture("multi_hop_delivery")
    graph = fixture.candidate_graph()
    record = evaluate_network_communication(
        scene=fixture.scene,
        graph=graph,
        selected_edge_ids=fixture.selected_edge_ids,
        source_id=fixture.source_id,
        target_ids=fixture.target_ids,
        config=NetworkCommunicationConfig(
            channel_config=ChannelModelConfig(default_tx_power_dbm=-120.0),
            link_config=LinkTransmissionConfig(
                fixed_transmission_time_s=0.001,
                deadline_s=0.003,
                target_reliability=None,
            ),
        ),
    )

    assert record.network_delivery_probability == 0.0
    assert record.network_latency_s == 0.0
    assert record.network_successful_delivery_latency_s == 0.0
    assert record.network_scheduled_latency_s > 0.0
    assert record.network_energy_j > 0.0
