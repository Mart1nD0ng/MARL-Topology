import pytest

from marl_topology.evaluation import (
    STAGE5_0H_REQUIREMENT_DIAGNOSIS_STAGE_ID,
    SUPPORTED_FAILURE_REASONS,
    TAU_DIAGNOSTIC_VALUES,
    TAU_REQUIREMENT_MIN,
    TAU_STRESS_CANDIDATES,
    build_stage5_0h_requirement_feasibility_diagnosis,
)


def test_stage5_0h_report_records_requirement_baseline_without_selection() -> None:
    report = build_stage5_0h_requirement_feasibility_diagnosis()

    assert report["stage"] == STAGE5_0H_REQUIREMENT_DIAGNOSIS_STAGE_ID
    assert report["source_kind"] == "stage5_0f_alpha_fixture_suite"
    assert report["tau_requirement_min"] == TAU_REQUIREMENT_MIN == 0.9
    assert report["tau_stress_candidates"] == list(TAU_STRESS_CANDIDATES)
    assert report["tau_diagnostic_values"] == list(TAU_DIAGNOSTIC_VALUES)
    assert report["objective_feasibility_condition"] == (
        "consensus_success_probability >= tau_requirement_min"
    )

    checks = report["checks"]
    assert checks["tau_requirement_min_recorded"] is True
    assert checks["objective_direction_is_greater_equal"] is True
    assert checks["final_tau_below_requirement_selected"] is False
    assert checks["final_tau_selected"] is False
    assert checks["simulation_parameters_changed_to_force_feasibility"] is False
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False


def test_stage5_0h_classifies_all_infeasible_rows() -> None:
    report = build_stage5_0h_requirement_feasibility_diagnosis()
    row_diagnosis = report["row_diagnosis"]
    infeasible = [row for row in row_diagnosis if not row["requirement_met"]]
    feasible = [row for row in row_diagnosis if row["requirement_met"]]

    assert len(row_diagnosis) == 24
    assert len(feasible) == 6
    assert len(infeasible) == 18
    assert all(row["failure_reason"] in SUPPORTED_FAILURE_REASONS for row in infeasible)
    assert all(row["failure_reason"] is None for row in feasible)

    reasons = {row["failure_reason"] for row in infeasible}
    for expected in {
        "link_budget_failure",
        "deadline_failure",
        "retransmission_insufficient",
        "topology_candidate_failure",
        "interference_failure",
        "pbft_quorum_failure",
        "primary_distribution_failure",
        "resource_budget_failure",
    }:
        assert expected in reasons


def test_stage5_0h_family_dominant_failure_reasons_match_requirement_view() -> None:
    report = build_stage5_0h_requirement_feasibility_diagnosis()
    summary = {row["scenario_family"]: row for row in report["family_summary"]}

    assert len(summary) == 8
    assert summary["blocked_or_nlos_urban"]["dominant_failure_reason"] == "link_budget_failure"
    assert summary["near_threshold_link_budget"]["dominant_failure_reason"] == (
        "link_budget_failure"
    )
    assert summary["deadline_tight_retransmission"]["dominant_failure_reason"] == (
        "deadline_failure"
    )
    assert summary["same_resource_interference"]["dominant_failure_reason"] == (
        "interference_failure"
    )
    assert summary["weak_primary_distribution"]["dominant_failure_reason"] == (
        "primary_distribution_failure"
    )
    assert summary["clear_free_space_reference"]["feasible_topology_count"] == 2
    assert summary["sparse_vs_dense_tradeoff"]["feasible_topology_count"] == 2


def test_stage5_0h_parameter_sanity_table_and_envelope_plan_cover_required_controls() -> None:
    report = build_stage5_0h_requirement_feasibility_diagnosis()

    params = {row["parameter"]: row["sanity_status"] for row in report["parameter_sanity_table"]}
    for required in {
        "tx_power",
        "bandwidth",
        "noise",
        "carrier_frequency",
        "path_loss_los_nlos_penalty",
        "interference_model",
        "payload_bits",
        "deadline",
        "attempt_duration",
        "max_retransmissions",
        "rsu_vehicle_distances",
        "candidate_edge_radius",
        "pbft_n_f_quorum",
        "fault_filter_mode",
    }:
        assert required in params
    assert params["pbft_n_f_quorum"] == "reasonable"
    assert params["fault_filter_mode"] == "too_loose"
    assert params["deadline"] == "too_strict"

    sweeps = {row["sweep_id"]: row for row in report["feasibility_envelope_plan"]}
    for required in {
        "bandwidth_sweep",
        "tx_power_sweep",
        "deadline_sweep",
        "payload_sweep",
        "rsu_height_placement_sweep",
        "resource_orthogonalization_sweep",
        "fault_filter_mode_comparison",
        "topology_candidate_expansion",
    }:
        assert required in sweeps
        assert "do not lower tau_requirement_min during sweep" in sweeps[required]["blocked_changes"]


def test_stage5_0h_rejects_lower_requirement_baseline() -> None:
    with pytest.raises(ValueError, match="must not be below 0.9"):
        build_stage5_0h_requirement_feasibility_diagnosis(tau_requirement_min=0.8)

    with pytest.raises(ValueError, match="diagnostic_values"):
        build_stage5_0h_requirement_feasibility_diagnosis(tau_diagnostic_values=(0.95,))

    with pytest.raises(ValueError, match="stress_candidates"):
        build_stage5_0h_requirement_feasibility_diagnosis(tau_stress_candidates=(0.85,))
