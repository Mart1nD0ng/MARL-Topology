from pathlib import Path

from marl_topology.evaluation import (
    STAGE4_8_BOUNDARY_AUDIT_STAGE_ID,
    STAGE4_8_REQUIRED_CASES,
    build_stage4_8_boundary_audit_report,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows_by_name(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["topology_name"]): row for row in report["audit_rows"]}


def test_stage4_8_report_contains_required_boundary_cases() -> None:
    report = build_stage4_8_boundary_audit_report()
    rows = _rows_by_name(report)

    assert report["stage"] == STAGE4_8_BOUNDARY_AUDIT_STAGE_ID
    assert set(rows) == set(STAGE4_8_REQUIRED_CASES)
    assert report["checks"]["required_cases_present"] is True
    assert report["checks"]["non_saturated_consensus_present"] is True
    assert any(
        0.0 < row["consensus_success_probability"] < 1.0
        for row in report["audit_rows"]
    )


def test_failed_scheduled_message_keeps_latency_and_energy_visible() -> None:
    report = build_stage4_8_boundary_audit_report()
    row = _rows_by_name(report)["failed_scheduled_message"]

    assert row["network_deadline_delivery_probability"] == 0.0
    assert row["scheduled_latency_s"] > 0.0
    assert row["successful_delivery_latency_s"] == 0.0
    assert row["energy_j"] > 0.0
    assert "scheduled_latency_energy_visible" in row["diagnostic_flags"]
    assert report["checks"]["failed_scheduled_message_has_latency_and_energy"] is True


def test_inverse_reliability_cap_is_a_diagnostic() -> None:
    report = build_stage4_8_boundary_audit_report()
    row = _rows_by_name(report)["unreachable_reliability_target"]

    assert "required_transmission_time_capped" in row["diagnostic_flags"]
    assert "zero_deadline_delivery" in row["diagnostic_flags"]
    assert row["network_deadline_delivery_probability"] == 0.0
    assert report["checks"]["inverse_reliability_cap_is_diagnostic"] is True


def test_expected_initiator_distinguishes_weak_and_central_primaries() -> None:
    report = build_stage4_8_boundary_audit_report()
    rows = _rows_by_name(report)

    weak = rows["weak_edge_primary"]["per_primary_reliability"]
    assert weak["edge_a"] < weak["center"]
    assert weak["edge_a"] < weak["edge_b"]
    assert report["checks"]["weak_primary_distinguished"] is True

    center_case = rows["center_vs_edge_primary"]["per_primary_reliability"]
    assert all(
        center_case["center"] > value
        for node_id, value in center_case.items()
        if node_id != "center"
    )
    assert report["checks"]["center_primary_stronger_than_edge"] is True


def test_full_graph_is_only_a_baseline_and_metrics_are_registered() -> None:
    report = build_stage4_8_boundary_audit_report()
    full = _rows_by_name(report)["interference_full_graph_penalty"]

    assert full["is_full_graph_baseline"] is True
    assert full["is_oracle_candidate"] is False
    assert report["checks"]["full_graph_is_baseline_not_oracle"] is True
    assert report["checks"]["oracle_candidate_is_not_actor_input"] is True
    assert report["metric_governance"]["new_metric_names_introduced"] == []
    assert {row["metric_name"] for row in report["metric_table"]} == {
        "consensus_success",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    }


def test_stage4_8_source_has_no_forbidden_routes() -> None:
    source = (
        ROOT / "src" / "marl_topology" / "evaluation" / "stage4_boundary_audit.py"
    ).read_text(encoding="utf-8")
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
        "Monte Carlo",
        "from random",
        "random.",
        "itertools.combinations",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.8 audit source uses forbidden routes: {hits}"
