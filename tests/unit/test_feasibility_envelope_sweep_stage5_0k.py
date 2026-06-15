from marl_topology.evaluation import (
    REQUIRED_SWEEP_ROW_FIELDS,
    STAGE5_0K_STAGE3_BACKED_SWEEP_IDS,
    STAGE5_0K_STAGE3_BACKED_SWEEP_STAGE_ID,
    TAU_REQUIREMENT_MIN,
    build_stage5_0k_stage3_backed_feasibility_envelope_sweep,
)


def test_stage5_0k_report_keeps_tau_fixed_and_no_selection() -> None:
    report = build_stage5_0k_stage3_backed_feasibility_envelope_sweep()

    assert report["stage"] == STAGE5_0K_STAGE3_BACKED_SWEEP_STAGE_ID
    assert report["tau_requirement_min"] == TAU_REQUIREMENT_MIN == 0.9
    checks = report["checks"]
    assert checks["tau_requirement_min_fixed"] is True
    assert checks["final_tau_selected"] is False
    assert checks["final_tau_below_requirement_selected"] is False
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False


def test_stage5_0k_executes_minimal_stage3_backed_scope() -> None:
    report = build_stage5_0k_stage3_backed_feasibility_envelope_sweep()

    assert tuple(report["executed_sweep_ids"]) == STAGE5_0K_STAGE3_BACKED_SWEEP_IDS
    assert set(report["deferred_sweep_ids"]) == {
        "tx_power_sweep",
        "payload_sweep",
        "rsu_height_placement_sweep",
        "fault_filter_mode_comparison",
        "topology_candidate_expansion",
    }
    assert report["stage3_backing"]["uses_stage3_network_records"] is True
    assert report["stage3_backing"]["finite_blocklength_regime_id"] == (
        "urlcc_finite_blocklength_v1"
    )
    assert report["stage3_backing"]["matrix_adapter_id"] == (
        "stage4_stage3_network_to_pbft_matrix_v1"
    )


def test_stage5_0k_rows_have_stage3_schema_and_valid_ranges() -> None:
    report = build_stage5_0k_stage3_backed_feasibility_envelope_sweep()

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


def test_stage5_0k_stage3_backed_sweeps_have_expected_signals() -> None:
    report = build_stage5_0k_stage3_backed_feasibility_envelope_sweep()
    summary = {row["sweep_id"]: row for row in report["sweep_summary"]}

    for sweep_id in STAGE5_0K_STAGE3_BACKED_SWEEP_IDS:
        assert summary[sweep_id]["stage3_backed"] is True
        assert summary[sweep_id]["baseline_requirement_met"] is False
        assert summary[sweep_id]["intervention_requirement_met"] is True
        assert summary[sweep_id]["max_delta_consensus_success_probability"] > 0.0
        assert summary[sweep_id]["monotonicity_check_passed"] is True

    checks = report["checks"]
    assert checks["at_least_one_stage3_backed_infeasible_to_feasible_transition"] is True
    assert checks["bandwidth_sweep_improves_reliability"] is True
    assert checks["deadline_sweep_improves_reliability"] is True
    assert checks["deadline_sweep_records_latency_cost"] is True
    assert checks["resource_orthogonalization_improves_reliability"] is True


def test_stage5_0k_resource_orthogonalization_changes_interference_groups() -> None:
    report = build_stage5_0k_stage3_backed_feasibility_envelope_sweep()
    rows = {
        row["control_value_label"]: row
        for row in report["sweep_rows"]
        if row["sweep_id"] == "resource_orthogonalization_sweep"
    }

    assert rows["shared_resource_stage3"]["interference_group_ids"]
    assert rows["orthogonal_resources_stage3"]["interference_group_ids"] == []
    assert rows["orthogonal_resources_stage3"]["consensus_success_probability"] > (
        rows["shared_resource_stage3"]["consensus_success_probability"]
    )
    assert report["checks"]["shared_resource_has_interference_groups"] is True
    assert report["checks"]["orthogonal_resource_has_no_interference_groups"] is True


def test_stage5_0k_metric_governance_introduces_no_new_metric_names() -> None:
    report = build_stage5_0k_stage3_backed_feasibility_envelope_sweep()

    metric_governance = report["metric_governance"]
    assert metric_governance["metric_valued_fields"] == [
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    ]
    assert metric_governance["new_metric_names_introduced"] == []
