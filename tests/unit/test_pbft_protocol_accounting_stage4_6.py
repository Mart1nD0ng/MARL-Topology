from pathlib import Path

import pytest

from marl_topology.evaluation import build_stage4_5_baseline_oracle_review
from marl_topology.network import NetworkCommunicationRecord
from marl_topology.protocol import (
    PBFT_PROTOCOL_ACCOUNTING_MODEL_ID,
    PBFTPhaseBudgets,
    account_pbft_protocol_latency_energy,
)


ROOT = Path(__file__).resolve().parents[2]
NODES = ("n0", "n1", "n2", "n3")


def _record(
    source_id: str,
    target_id: str,
    probability: float,
    latency_s: float,
    energy_j: float,
    *,
    is_oracle: bool = False,
) -> NetworkCommunicationRecord:
    return NetworkCommunicationRecord(
        scenario_id="synthetic_stage4_6",
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
        network_energy_j=energy_j,
        hop_count=0,
        is_full_graph_baseline=False,
        is_oracle=is_oracle,
    )


def test_protocol_accounting_sums_phase_latency_and_scheduled_energy() -> None:
    budgets = PBFTPhaseBudgets(
        pre_prepare_budget_s=0.10,
        prepare_budget_s=0.10,
        commit_budget_s=0.10,
    )
    record = account_pbft_protocol_latency_energy(
        node_ids=NODES,
        phase_records={
            "pre_prepare": (
                _record("n0", "n1", 1.0, 0.01, 1.0),
                _record("n0", "n2", 1.0, 0.02, 2.0),
            ),
            "prepare": (_record("n1", "n2", 1.0, 0.50, 4.0),),
            "commit": (),
        },
        phase_budgets=budgets,
    )

    assert record.accounting_model_id == PBFT_PROTOCOL_ACCOUNTING_MODEL_ID
    assert record.phase_accounting["pre_prepare"].phase_latency_s == pytest.approx(0.02)
    assert record.phase_accounting["prepare"].phase_latency_s == pytest.approx(0.10)
    assert record.protocol_latency_s == pytest.approx(0.12)
    assert record.protocol_energy_j == pytest.approx(7.0)
    assert record.phase_accounting["prepare"].deadline_filtered_message_count == 1


def test_protocol_accounting_is_not_consensus_probability_or_reward() -> None:
    record = account_pbft_protocol_latency_energy(
        node_ids=NODES,
        phase_records={"pre_prepare": (_record("n0", "n1", 0.0, 0.0, 0.5),)},
        phase_budgets=PBFTPhaseBudgets(1.0, 1.0, 1.0),
    )

    assert record.exports_consensus_probability is False
    assert record.reward_logic_implemented is False
    assert not hasattr(record, "consensus_success_probability")
    assert record.phase_accounting["pre_prepare"].zero_delivery_message_count == 1
    assert record.protocol_energy_j == pytest.approx(0.5)


def test_failed_scheduled_record_uses_scheduled_latency_for_accounting() -> None:
    record = account_pbft_protocol_latency_energy(
        node_ids=NODES,
        phase_records={"pre_prepare": (_record("n0", "n1", 0.0, 0.25, 0.5),)},
        phase_budgets=PBFTPhaseBudgets(1.0, 1.0, 1.0),
    )

    assert record.phase_accounting["pre_prepare"].zero_delivery_message_count == 1
    assert record.phase_accounting["pre_prepare"].phase_latency_s == pytest.approx(0.25)
    assert record.protocol_latency_s == pytest.approx(0.25)
    assert record.protocol_energy_j == pytest.approx(0.5)


def test_protocol_accounting_metric_rows_are_registered() -> None:
    record = account_pbft_protocol_latency_energy(
        node_ids=NODES,
        phase_records={"pre_prepare": (_record("n0", "n1", 1.0, 0.01, 0.5),)},
        phase_budgets=PBFTPhaseBudgets(1.0, 1.0, 1.0),
    )

    rows = record.metric_rows(scenario_id="synthetic_stage4_6", topology_id="topology")
    assert {row["metric_name"] for row in rows} == {
        "latency",
        "energy",
        "topology_diagnostics",
    }


def test_protocol_accounting_rejects_oracle_and_self_message_records() -> None:
    with pytest.raises(ValueError, match="oracle"):
        account_pbft_protocol_latency_energy(
            node_ids=NODES,
            phase_records={
                "pre_prepare": (_record("n0", "n1", 1.0, 0.01, 0.5, is_oracle=True),)
            },
            phase_budgets=PBFTPhaseBudgets(1.0, 1.0, 1.0),
        )

    with pytest.raises(ValueError, match="self-message"):
        account_pbft_protocol_latency_energy(
            node_ids=NODES,
            phase_records={"pre_prepare": (_record("n0", "n0", 1.0, 0.01, 0.5),)},
            phase_budgets=PBFTPhaseBudgets(1.0, 1.0, 1.0),
        )


def test_stage4_5_uses_stage4_6_protocol_accounting() -> None:
    report = build_stage4_5_baseline_oracle_review()

    for row in report["topology_rows"]:
        diagnostics = row["diagnostics"]
        accounting = diagnostics["protocol_accounting"]
        assert accounting["protocol_accounting_model_id"] == PBFT_PROTOCOL_ACCOUNTING_MODEL_ID
        assert diagnostics["latency_aggregation"] == "stage4_6_sum_phase_max_clipped_to_budget"
        assert diagnostics["energy_aggregation"] == "stage4_6_sum_scheduled_attempt_energy"
        assert row["metrics"]["latency"] >= 0.0
        assert row["metrics"]["energy"] >= 0.0


def test_protocol_accounting_source_avoids_forbidden_routes() -> None:
    source = (ROOT / "src" / "marl_topology" / "protocol" / "pbft_accounting.py").read_text(
        encoding="utf-8"
    )
    banned_terms = [
        "D:\\PhD_works\\v5",
        "import v5",
        "from v5",
        "P_eff",
        "train_loop",
        "optimizer",
        "backward(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "def reward",
        "class Reward",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.6 accounting source uses forbidden routes: {hits}"
