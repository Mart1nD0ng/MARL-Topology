from marl_topology.evaluation import (
    REQUIRED_SWEEP_ROW_FIELDS,
    STAGE5_0L_RANGE_REVIEW_STAGE_ID,
    STAGE5_0L_SWEEP_IDS,
    TAU_REQUIREMENT_MIN,
    build_stage5_0l_stage3_backed_sweep_range_review,
)


def test_stage5_0l_report_keeps_tau_fixed_and_no_selection() -> None:
    report = build_stage5_0l_stage3_backed_sweep_range_review()

    assert report["stage"] == STAGE5_0L_RANGE_REVIEW_STAGE_ID
    assert report["tau_requirement_min"] == TAU_REQUIREMENT_MIN == 0.9
    checks = report["checks"]
    assert checks["tau_requirement_min_fixed"] is True
    assert checks["final_tau_selected"] is False
    assert checks["final_tau_below_requirement_selected"] is False
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False


def test_stage5_0l_executes_range_expansion_sweeps() -> None:
    report = build_stage5_0l_stage3_backed_sweep_range_review()

    assert tuple(report["executed_sweep_ids"]) == STAGE5_0L_SWEEP_IDS
    assert set(report["deferred_sweep_ids"]) == {
        "bandwidth_sweep",
        "deadline_sweep",
        "fault_filter_mode_comparison",
        "topology_candidate_expansion",
    }
    assert report["range_review"]["bounded_single_axis_controls"] is True
    assert report["range_review"]["resource_budget_limit_included"] is True
    assert report["range_review"]["range_rows_are_not_training_data"] is True


def test_stage5_0l_rows_have_stage3_schema_and_valid_ranges() -> None:
    report = build_stage5_0l_stage3_backed_sweep_range_review()

    for row in report["sweep_rows"]:
        assert set(REQUIRED_SWEEP_ROW_FIELDS).issubset(row)
        assert row["stage3_backed"] is True
        assert row["stage3_network_record_count"] == 12
        assert row["finite_blocklength_regime_id"] == "urlcc_finite_blocklength_v1"
        assert row["network_regime_id"] == "stage3_network_communication_v1"
        assert row["matrix_adapter_id"] == "stage4_stage3_network_to_pbft_matrix_v1"
        assert row["protocol_accounting_model_id"] == "stage4_pbft_protocol_accounting_v1"
        assert 0.0 <= row["consensus_success_probability"] <= 1.0
        assert 0.0 <= row["mean_network_delivery_probability"] <= 1.0
        assert 0.0 <= row["min_network_delivery_probability"] <= 1.0
        assert row["latency"] >= 0.0
        assert row["energy"] >= 0.0
        assert row["total_network_energy_j"] >= 0.0
        assert row["tau_requirement_min"] == 0.9
        assert row["is_deployment_actor_input"] is False
        assert not (row["is_full_graph_baseline"] and row["is_oracle_candidate"])


def test_stage5_0l_every_range_sweep_has_recovery_signal() -> None:
    report = build_stage5_0l_stage3_backed_sweep_range_review()
    summary = {row["sweep_id"]: row for row in report["sweep_summary"]}

    for sweep_id in STAGE5_0L_SWEEP_IDS:
        assert summary[sweep_id]["stage3_backed"] is True
        assert summary[sweep_id]["baseline_requirement_met"] is False
        assert summary[sweep_id]["intervention_requirement_met"] is True
        assert summary[sweep_id]["max_delta_consensus_success_probability"] > 0.0
        assert summary[sweep_id]["monotonicity_check_passed"] is True

    checks = report["checks"]
    assert checks["every_sweep_has_infeasible_to_feasible_transition"] is True
    assert checks["tx_power_sweep_improves_reliability"] is True
    assert checks["payload_reduction_improves_reliability"] is True
    assert checks["rsu_height_sweep_improves_reliability"] is True
    assert checks["resource_budget_limit_sweep_improves_reliability"] is True


def test_stage5_0l_resource_budget_limit_is_visible() -> None:
    report = build_stage5_0l_stage3_backed_sweep_range_review()
    rows = {
        row["control_value_label"]: row
        for row in report["sweep_rows"]
        if row["sweep_id"] == "resource_budget_limit_sweep"
    }

    assert rows["resource_budget_2_stage3"]["interference_group_ids"]
    assert rows["resource_budget_6_stage3"]["interference_group_ids"] == []
    assert rows["resource_budget_6_stage3"]["consensus_success_probability"] > (
        rows["resource_budget_2_stage3"]["consensus_success_probability"]
    )
    assert report["checks"]["limited_resource_budget_has_interference_groups"] is True
    assert report["checks"]["sufficient_resource_budget_has_no_interference_groups"] is True


def test_stage5_0l_realism_review_keeps_ranges_diagnostic() -> None:
    report = build_stage5_0l_stage3_backed_sweep_range_review()
    review = {row["parameter"]: row for row in report["realism_review"]}

    for parameter in (
        "tx_power_dbm",
        "payload_bits",
        "rsu_height_m",
        "resource_budget_count",
    ):
        assert review[parameter]["realism_status"] == "unknown_needs_reference"

    assert review["fault_filter_mode"]["realism_status"] == "conservative_diagnostic"
    assert report["metric_governance"]["new_metric_names_introduced"] == []
