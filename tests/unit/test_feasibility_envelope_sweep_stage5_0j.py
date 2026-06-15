from marl_topology.evaluation import (
    REQUIRED_SWEEP_ROW_FIELDS,
    STAGE5_0J_MINIMAL_SWEEP_STAGE_ID,
    STAGE5_0J_SWEEP_IDS,
    TAU_REQUIREMENT_MIN,
    build_stage5_0j_minimal_feasibility_envelope_sweep,
)


def test_stage5_0j_report_keeps_tau_fixed_and_no_selection() -> None:
    report = build_stage5_0j_minimal_feasibility_envelope_sweep()

    assert report["stage"] == STAGE5_0J_MINIMAL_SWEEP_STAGE_ID
    assert report["tau_requirement_min"] == TAU_REQUIREMENT_MIN == 0.9
    checks = report["checks"]
    assert checks["tau_requirement_min_fixed"] is True
    assert checks["final_tau_selected"] is False
    assert checks["final_tau_below_requirement_selected"] is False
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False


def test_stage5_0j_executed_and_deferred_sweeps_match_minimal_scope() -> None:
    report = build_stage5_0j_minimal_feasibility_envelope_sweep()

    assert tuple(report["executed_sweep_ids"]) == STAGE5_0J_SWEEP_IDS
    assert set(report["deferred_sweep_ids"]) == {
        "tx_power_sweep",
        "payload_sweep",
        "rsu_height_placement_sweep",
    }

    row_sweep_ids = {row["sweep_id"] for row in report["sweep_rows"]}
    assert set(STAGE5_0J_SWEEP_IDS).issubset(row_sweep_ids)


def test_stage5_0j_rows_have_schema_and_valid_ranges() -> None:
    report = build_stage5_0j_minimal_feasibility_envelope_sweep()

    for row in report["sweep_rows"]:
        assert set(REQUIRED_SWEEP_ROW_FIELDS).issubset(row)
        assert 0.0 <= row["consensus_success_probability"] <= 1.0
        assert row["latency"] >= 0.0
        assert row["energy"] >= 0.0
        assert row["tau_requirement_min"] == 0.9
        assert row["is_deployment_actor_input"] is False
        assert not (row["is_full_graph_baseline"] and row["is_oracle_candidate"])
        assert row["parameter_source"]
        assert row["diagnostic_flags"]


def test_stage5_0j_detects_minimal_feasibility_envelope_signals() -> None:
    report = build_stage5_0j_minimal_feasibility_envelope_sweep()
    summary = {row["sweep_id"]: row for row in report["sweep_summary"]}

    for sweep_id in (
        "bandwidth_sweep",
        "deadline_sweep",
        "resource_orthogonalization_sweep",
        "topology_candidate_expansion",
    ):
        assert summary[sweep_id]["baseline_requirement_met"] is False
        assert summary[sweep_id]["intervention_requirement_met"] is True
        assert summary[sweep_id]["max_delta_consensus_success_probability"] > 0.0
        assert summary[sweep_id]["monotonicity_check_passed"] is True

    assert summary["fault_filter_mode_comparison"]["intervention_requirement_met"] is False
    assert summary["fault_filter_mode_comparison"]["monotonicity_check_passed"] is True
    assert report["checks"]["at_least_one_infeasible_to_feasible_transition"] is True
    assert report["checks"]["conservative_fault_filter_does_not_increase_reliability"] is True


def test_stage5_0j_metric_governance_introduces_no_new_metric_names() -> None:
    report = build_stage5_0j_minimal_feasibility_envelope_sweep()

    metric_governance = report["metric_governance"]
    assert metric_governance["metric_valued_fields"] == [
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    ]
    assert metric_governance["new_metric_names_introduced"] == []
