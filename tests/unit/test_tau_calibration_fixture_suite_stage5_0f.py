from marl_topology.evaluation import (
    STAGE5_0F_REQUIRED_FAMILIES,
    STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID,
    build_stage5_0f_tau_calibration_fixture_suite_report,
    build_tau_consensus_calibration_report,
)


REQUIRED_ROW_FIELDS = {
    "scenario_set_id",
    "scenario_family",
    "scenario_id",
    "fixture_id",
    "topology_name",
    "topology_family",
    "selected_edge_count",
    "is_full_graph_baseline",
    "is_oracle_candidate",
    "is_deployment_actor_input",
    "consensus_success_probability",
    "latency",
    "energy",
    "per_primary_reliability",
    "diagnostic_flags",
}


def test_stage5_0f_fixture_suite_required_families_and_variants_exist() -> None:
    report = build_stage5_0f_tau_calibration_fixture_suite_report()
    rows = report["topology_evaluation_rows"]

    assert report["stage"] == STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID
    family_ids = {row["scenario_family"] for row in rows}
    assert set(STAGE5_0F_REQUIRED_FAMILIES).issubset(family_ids)

    for family_id in STAGE5_0F_REQUIRED_FAMILIES:
        family_rows = [row for row in rows if row["scenario_family"] == family_id]
        topology_families = {row["topology_family"] for row in family_rows}
        assert len(family_rows) >= 3
        assert "weak_or_disconnected_baseline" in topology_families
        assert "sparse_candidate" in topology_families
        assert any(row["is_full_graph_baseline"] for row in family_rows)


def test_stage5_0f_fixture_rows_have_required_fields_and_valid_values() -> None:
    report = build_stage5_0f_tau_calibration_fixture_suite_report()

    for row in report["topology_evaluation_rows"]:
        assert REQUIRED_ROW_FIELDS.issubset(row)
        assert row["selected_edge_count"] >= 0
        assert 0.0 <= row["consensus_success_probability"] <= 1.0
        assert row["latency"] >= 0.0
        assert row["energy"] >= 0.0
        assert row["is_deployment_actor_input"] is False
        if row["is_full_graph_baseline"]:
            assert row["is_oracle_candidate"] is False
        assert set(row["per_primary_reliability"]) == {"center", "edge_a", "edge_b", "edge_c"}


def test_stage5_0f_fixture_suite_exit_criteria_are_visible() -> None:
    report = build_stage5_0f_tau_calibration_fixture_suite_report()
    rows = report["topology_evaluation_rows"]
    checks = report["checks"]

    assert checks["non_saturated_consensus_present"] is True
    assert checks["sparse_better_than_full_graph_for_some_family"] is True
    assert checks["failed_scheduled_message_has_latency_or_energy"] is True
    assert checks["weak_primary_spread_present"] is True
    assert checks["full_graph_not_oracle"] is True
    assert checks["oracle_labels_not_actor_inputs"] is True
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False

    assert any(0.0 < row["consensus_success_probability"] < 1.0 for row in rows)
    assert any("weak_edge_primary" in row["diagnostic_flags"] for row in rows)


def test_stage5_0d_report_builder_accepts_stage5_0f_fixture_source() -> None:
    source = build_stage5_0f_tau_calibration_fixture_suite_report()
    report = build_tau_consensus_calibration_report(
        (0.05, 0.50),
        tau_source="owner_declared_test",
        source_report=source,
    )

    assert report["source_stage"] == STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID
    assert report["source_kind"] == "stage5_0f_alpha_fixture_suite"
    assert report["calibration_ready"] is True
    assert report["tau_selected"] is False
    assert report["final_tau_consensus"] is None
    assert report["owner_decision_packet"]["recommended_tau_candidate"] is None
    assert report["owner_decision_packet"]["owner_selected_tau"] is None
    assert report["checks"]["source_is_alpha_fixture_suite"] is True
    assert report["checks"]["sparse_better_than_full_graph_for_some_family"] is True
    assert len(report["scenario_summary"]) >= len(STAGE5_0F_REQUIRED_FAMILIES)
    assert any(
        row["topology_family"] == "sparse_candidate" and row["reliability_feasible"]
        for row in report["topology_by_tau_detail"]
        if row["tau_candidate"] == 0.05
    )
