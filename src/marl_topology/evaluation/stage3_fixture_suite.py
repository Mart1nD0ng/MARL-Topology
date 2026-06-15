"""Stage 3.5 micro-fixture suite review helpers."""

from __future__ import annotations

from marl_topology.channel import evaluate_channel, iter_channel_model_fixtures
from marl_topology.geometry3d import evaluate_visibility
from marl_topology.link import iter_link_transmission_fixtures
from marl_topology.network import iter_network_communication_fixtures
from marl_topology.scenario import iter_geometry_visibility_fixtures


REQUIRED_STAGE3_MICRO_FIXTURES = (
    "free_space_close",
    "free_space_far",
    "blocked_by_building",
    "urban_gap_los",
    "urban_canyon_nlos",
    "rsu_high_los",
    "two_transmitters_interference",
    "orthogonal_channel_no_interference",
    "multi_hop_delivery",
    "resource_redundant_topology",
)


def build_stage3_micro_fixture_suite_review() -> dict[str, object]:
    geometry_rows = _geometry_rows()
    channel_rows = _channel_rows()
    link_rows = _link_rows()
    network_rows = _network_rows()
    all_fixture_ids = _all_fixture_ids(geometry_rows, channel_rows, link_rows, network_rows)
    covered_required = _covered_required_fixture_ids(
        geometry_rows,
        channel_rows,
        network_rows,
    )
    missing_required = sorted(set(REQUIRED_STAGE3_MICRO_FIXTURES) - covered_required)

    return {
        "stage": "stage_3_5_micro_fixture_suite_review",
        "required_micro_fixture_ids": list(REQUIRED_STAGE3_MICRO_FIXTURES),
        "fixture_counts": {
            "geometry": len(geometry_rows),
            "channel": len(channel_rows),
            "link_transmission": len(link_rows),
            "network": len(network_rows),
            "total_rows": len(geometry_rows) + len(channel_rows) + len(link_rows) + len(network_rows),
        },
        "geometry_rows": geometry_rows,
        "channel_rows": channel_rows,
        "link_transmission_rows": link_rows,
        "network_rows": network_rows,
        "checks": {
            "required_micro_fixtures_covered": not missing_required,
            "missing_required_micro_fixture_ids": missing_required,
            "geometry_expectations_match": all(row["expectation_matches"] for row in geometry_rows),
            "channel_expectations_match": all(row["expectation_matches"] for row in channel_rows),
            "link_records_nonnegative": all(row["latency_energy_nonnegative"] for row in link_rows),
            "network_records_nonnegative": all(row["latency_energy_nonnegative"] for row in network_rows),
            "network_records_not_oracle": all(row["is_oracle"] is False for row in network_rows),
            "training_run": False,
            "v5_code_migrated": False,
            "reward_implemented": False,
            "pbft_application_semantics_defined": False,
            "new_stage3_registry_entries": [],
        },
        "boundary": {
            "records_are_stage3_communication_only": True,
            "not_pbft_consensus": True,
            "not_reward": True,
            "not_training_data": True,
            "not_oracle": True,
        },
        "all_fixture_ids": all_fixture_ids,
    }


def _geometry_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fixture in iter_geometry_visibility_fixtures():
        record = evaluate_visibility(fixture.scene, fixture.tx_id, fixture.rx_id)
        blocker_ids = tuple(blocker.building_id for blocker in record.blocker_records)
        rows.append(
            {
                "layer": "geometry",
                "fixture_id": fixture.fixture_id,
                "scenario_id": record.scenario_id,
                "distance_3d_m": record.distance_3d_m,
                "los_state": record.los_state.value,
                "blocker_ids": list(blocker_ids),
                "expected_los_state": fixture.expected_los_state,
                "expected_blocker_ids": list(fixture.expected_blocker_ids),
                "expectation_matches": (
                    record.los_state.value == fixture.expected_los_state
                    and blocker_ids == fixture.expected_blocker_ids
                ),
            }
        )
    return rows


def _channel_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fixture in iter_channel_model_fixtures():
        record = evaluate_channel(
            fixture.scene,
            fixture.tx_id,
            fixture.rx_id,
            active_transmissions=fixture.active_transmissions,
            resource_id=fixture.resource_id,
        )
        rows.append(
            {
                "layer": "channel",
                "fixture_id": fixture.fixture_id,
                "scenario_id": record.scenario_id,
                "los_state": record.los_state.value,
                "path_loss_db": record.path_loss_db,
                "sinr_db": record.sinr_db,
                "interference_tx_ids": list(record.interference_tx_ids),
                "expected_interference_tx_ids": list(fixture.expected_interference_tx_ids),
                "expectation_matches": (
                    record.interference_tx_ids == fixture.expected_interference_tx_ids
                ),
            }
        )
    return rows


def _link_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fixture in iter_link_transmission_fixtures():
        record = fixture.evaluate()
        rows.append(
            {
                "layer": "link_transmission",
                "fixture_id": fixture.fixture_id,
                "scenario_id": record.scenario_id,
                "transmission_attempted": record.transmission_attempted,
                "payload_bits": record.payload_bits,
                "bandwidth_hz": record.bandwidth_hz,
                "p2p_latency_s": record.p2p_latency_s,
                "p2p_energy_j": record.p2p_energy_j,
                "packet_success_probability": record.packet_success_probability,
                "latency_energy_nonnegative": (
                    record.p2p_latency_s >= 0.0 and record.p2p_energy_j >= 0.0
                ),
            }
        )
    return rows


def _network_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fixture in iter_network_communication_fixtures():
        record = fixture.evaluate()
        rows.append(
            {
                "layer": "network",
                "fixture_id": fixture.fixture_id,
                "scenario_id": record.scenario_id,
                "primitive": record.primitive,
                "selected_edge_count": len(record.selected_edge_ids),
                "hop_count": record.hop_count,
                "network_delivery_probability": record.network_delivery_probability,
                "network_latency_s": record.network_latency_s,
                "network_energy_j": record.network_energy_j,
                "interference_group_ids": list(record.interference_group_ids),
                "is_full_graph_baseline": record.is_full_graph_baseline,
                "is_oracle": record.is_oracle,
                "latency_energy_nonnegative": (
                    record.network_latency_s >= 0.0 and record.network_energy_j >= 0.0
                ),
            }
        )
    return rows


def _covered_required_fixture_ids(
    geometry_rows: list[dict[str, object]],
    channel_rows: list[dict[str, object]],
    network_rows: list[dict[str, object]],
) -> set[str]:
    ids = {str(row["fixture_id"]) for row in geometry_rows + channel_rows + network_rows}
    if "two_transmitters_interference_network" in ids:
        ids.add("two_transmitters_interference")
    if "orthogonal_channel_no_interference_network" in ids:
        ids.add("orthogonal_channel_no_interference")
    return ids


def _all_fixture_ids(
    *row_groups: list[dict[str, object]],
) -> dict[str, list[str]]:
    return {
        rows[0]["layer"] if rows else f"empty_{index}": [str(row["fixture_id"]) for row in rows]
        for index, rows in enumerate(row_groups)
    }
