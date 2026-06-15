from pathlib import Path

import pytest

from marl_topology.network import NetworkCommunicationRecord, get_network_communication_fixture
from marl_topology.protocol import (
    MESSAGE_MATRIX_ADAPTER_ID,
    FAULT_FILTER_REMOVE_LARGEST,
    PBFTPhaseBudgets,
    PBFTThreePhaseConfig,
    build_pbft_message_matrices_from_network_records,
    evaluate_pbft_reliability_from_network_records,
)


ROOT = Path(__file__).resolve().parents[2]
NODES = ("n0", "n1", "n2", "n3")


def _record(
    source_id: str,
    target_id: str,
    probability: float,
    latency_s: float,
    *,
    scenario_id: str = "synthetic_stage4_3",
) -> NetworkCommunicationRecord:
    return NetworkCommunicationRecord(
        scenario_id=scenario_id,
        network_model_id="stage3_network_communication_v1",
        primitive="route",
        source_id=source_id,
        target_ids=(target_id,),
        selected_edge_ids=(),
        active_transmission_ids=(),
        interference_group_ids=(),
        reachable_node_ids=(source_id, target_id),
        route_node_ids=(source_id, target_id),
        route_edge_ids=(),
        hop_records=(),
        network_delivery_probability=probability,
        network_latency_s=0.0 if probability == 0.0 else latency_s,
        network_scheduled_latency_s=latency_s,
        network_successful_delivery_latency_s=0.0 if probability == 0.0 else latency_s,
        network_energy_j=0.0,
        hop_count=0,
        is_full_graph_baseline=False,
    )


def _complete_records(
    probability: float,
    latency_s: float,
    node_ids: tuple[str, ...] = NODES,
) -> tuple[NetworkCommunicationRecord, ...]:
    return tuple(
        _record(sender_id, receiver_id, probability, latency_s)
        for sender_id in node_ids
        for receiver_id in node_ids
        if sender_id != receiver_id
    )


def test_build_message_matrices_from_stage3_records() -> None:
    records = _complete_records(probability=0.8, latency_s=0.01)
    budgets = PBFTPhaseBudgets(
        pre_prepare_budget_s=0.1,
        prepare_budget_s=0.1,
        commit_budget_s=0.1,
    )

    matrices = build_pbft_message_matrices_from_network_records(
        NODES,
        {
            "pre_prepare": records,
            "prepare": records,
            "commit": records,
        },
        budgets,
    )

    assert matrices.adapter_id == MESSAGE_MATRIX_ADAPTER_ID
    assert matrices.pre_prepare_matrix[("n0", "n1")] == pytest.approx(0.8)
    assert matrices.prepare_matrix[("n2", "n3")] == pytest.approx(0.8)
    assert matrices.commit_matrix[("n3", "n0")] == pytest.approx(0.8)
    assert matrices.exports_consensus_metric is False
    assert not hasattr(matrices, "consensus_success_probability")


def test_late_stage3_record_is_zeroed_by_phase_budget() -> None:
    late = _record("n0", "n1", probability=0.9, latency_s=2.0)
    budgets = PBFTPhaseBudgets(
        pre_prepare_budget_s=0.1,
        prepare_budget_s=2.0,
        commit_budget_s=2.0,
    )

    matrices = build_pbft_message_matrices_from_network_records(
        NODES,
        {"pre_prepare": (late,)},
        budgets,
    )

    assert matrices.pre_prepare_matrix[("n0", "n1")] == 0.0
    assert matrices.deadline_filtered_count_by_phase["pre_prepare"] == 1
    assert matrices.zero_delivery_count_by_phase["pre_prepare"] == 1


def test_disconnected_stage3_record_lowers_matrix_entry() -> None:
    disconnected = get_network_communication_fixture("disconnected_reachability").evaluate()
    delivered = get_network_communication_fixture("multi_hop_delivery").evaluate()
    node_ids = ("rsu_0", "veh_0", "veh_1", "veh_2", "rsu_1")
    budgets = PBFTPhaseBudgets(1.0, 1.0, 1.0)

    matrices = build_pbft_message_matrices_from_network_records(
        node_ids,
        {"pre_prepare": (disconnected, delivered)},
        budgets,
    )

    assert matrices.pre_prepare_matrix[("rsu_0", "veh_2")] == 0.0
    assert matrices.pre_prepare_matrix[("rsu_0", "veh_1")] > 0.0
    assert matrices.zero_delivery_count_by_phase["pre_prepare"] >= 1


def test_same_resource_interference_lowers_matrix_entry_against_orthogonal_resource() -> None:
    same = get_network_communication_fixture("two_transmitters_interference_network").evaluate()
    orthogonal = get_network_communication_fixture(
        "orthogonal_channel_no_interference_network"
    ).evaluate()
    node_ids = ("rsu_0", "veh_0", "veh_1", "veh_2", "rsu_1")
    budgets = PBFTPhaseBudgets(1.0, 1.0, 1.0)

    same_matrix = build_pbft_message_matrices_from_network_records(
        node_ids,
        {"pre_prepare": (same,)},
        budgets,
    )
    orthogonal_matrix = build_pbft_message_matrices_from_network_records(
        node_ids,
        {"pre_prepare": (orthogonal,)},
        budgets,
    )

    key = ("rsu_0", "veh_0")
    assert same_matrix.pre_prepare_matrix[key] < orthogonal_matrix.pre_prepare_matrix[key]


def test_failed_record_zero_latency_is_not_consensus_success() -> None:
    failed = _record("n0", "n1", probability=0.0, latency_s=0.0)
    budgets = PBFTPhaseBudgets(1.0, 1.0, 1.0)

    matrices = build_pbft_message_matrices_from_network_records(
        NODES,
        {"pre_prepare": (failed,)},
        budgets,
    )

    assert matrices.pre_prepare_matrix[("n0", "n1")] == 0.0
    assert matrices.zero_delivery_count_by_phase["pre_prepare"] == 1
    assert matrices.exports_consensus_metric is False


def test_evaluate_pbft_reliability_from_network_records() -> None:
    records = _complete_records(probability=1.0, latency_s=0.01)
    budgets = PBFTPhaseBudgets(1.0, 1.0, 1.0)
    config = PBFTThreePhaseConfig(
        node_ids=NODES,
        primary_id="n0",
        fault_tolerance=1,
        fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST,
    )

    record = evaluate_pbft_reliability_from_network_records(
        config,
        {
            "pre_prepare": records,
            "prepare": records,
            "commit": records,
        },
        budgets,
    )

    assert record.consensus_success_probability == pytest.approx(1.0)


def test_adapter_rejects_invalid_budget_and_nodes() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        PBFTPhaseBudgets(-1.0, 1.0, 1.0)

    record = _record("n0", "missing", probability=0.9, latency_s=0.01)
    with pytest.raises(ValueError, match="target_ids"):
        build_pbft_message_matrices_from_network_records(
            NODES,
            {"pre_prepare": (record,)},
            PBFTPhaseBudgets(1.0, 1.0, 1.0),
        )


def test_adapter_source_avoids_reward_training_and_v5_routes() -> None:
    source = (
        ROOT / "src" / "marl_topology" / "protocol" / "message_matrix_adapter.py"
    ).read_text(encoding="utf-8")
    banned_terms = [
        "import v5",
        "from v5",
        "torch",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "backward(",
        "train_loop",
        "torch.save",
        "P_eff",
        "reward",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.3 adapter source uses forbidden routes: {hits}"
