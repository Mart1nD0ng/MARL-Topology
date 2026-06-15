import pytest

from marl_topology.evaluation import (
    STAGE5_0D_TAU_CALIBRATION_REPORT_STAGE_ID,
    TauCandidate,
    build_tau_consensus_calibration_report,
)


def test_tau_report_requires_owner_declared_candidates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_tau_consensus_calibration_report(())


def test_tau_report_rejects_out_of_range_and_selected_candidates() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_tau_consensus_calibration_report((1.2,))
    with pytest.raises(ValueError, match="does not select"):
        build_tau_consensus_calibration_report(
            (TauCandidate(tau_candidate=0.5, is_selected=True),)
        )


def test_tau_report_is_report_only_and_does_not_select_tau() -> None:
    report = build_tau_consensus_calibration_report((0.15, 0.55))

    assert report["stage"] == STAGE5_0D_TAU_CALIBRATION_REPORT_STAGE_ID
    assert report["calibration_ready"] is False
    assert report["tau_selected"] is False
    assert report["final_tau_consensus"] is None
    assert report["owner_decision_packet"]["recommended_tau_candidate"] is None
    assert report["owner_decision_packet"]["owner_selected_tau"] is None
    assert report["owner_decision_packet"]["owner_decision_status"] == "required_not_provided"

    for row in report["candidate_tau_rows"]:
        assert row["is_default"] is False
        assert row["is_selected"] is False

    checks = report["checks"]
    assert checks["candidate_tau_values_supplied"] is True
    assert checks["no_default_tau_candidates"] is True
    assert checks["no_final_tau_selected"] is True
    assert checks["stage4_8_threshold_not_used_as_default"] is True
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False


def test_tau_report_feasibility_counts_match_detail_rows() -> None:
    report = build_tau_consensus_calibration_report((0.15, 0.55))

    detail_rows = report["topology_by_tau_detail"]
    for summary in report["tau_feasibility_summary"]:
        tau = summary["tau_candidate"]
        tau_detail = [row for row in detail_rows if row["tau_candidate"] == tau]
        feasible = [row for row in tau_detail if row["reliability_feasible"]]
        non_full_feasible = [
            row for row in feasible if not row["is_full_graph_baseline"]
        ]
        full_feasible = [row for row in feasible if row["is_full_graph_baseline"]]

        assert summary["feasible_topology_count"] == len(feasible)
        assert summary["infeasible_topology_count"] == len(tau_detail) - len(feasible)
        assert summary["feasible_non_full_topology_count"] == len(non_full_feasible)
        assert summary["full_graph_feasible"] is bool(full_feasible)
        assert summary["full_graph_only_feasible"] is (
            bool(full_feasible) and len(feasible) == len(full_feasible)
        )


def test_tau_report_preserves_full_graph_and_actor_boundaries() -> None:
    report = build_tau_consensus_calibration_report((0.15, 0.55))
    topology_rows = report["topology_evaluation_rows"]

    assert any(row["is_full_graph_baseline"] for row in topology_rows)
    assert all(
        not row["is_oracle_candidate"]
        for row in topology_rows
        if row["is_full_graph_baseline"]
    )
    assert all(not row["is_deployment_actor_input"] for row in topology_rows)
    assert report["checks"]["full_graph_not_oracle"] is True
    assert report["checks"]["oracle_labels_not_actor_inputs"] is True


def test_tau_report_metric_governance_adds_no_new_metrics() -> None:
    report = build_tau_consensus_calibration_report((0.15, 0.55))
    metric_governance = report["metric_governance"]

    assert metric_governance["metric_valued_fields"] == [
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    ]
    assert metric_governance["metric_valued_fields_registered"] is True
    assert metric_governance["new_metric_names_introduced"] == []
    assert report["source_scope"] == "stage4_8_smoke_test_only_not_calibration_set"
