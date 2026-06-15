from marl_topology.evaluation import (
    STAGE5_0M_OBJECTIVE_READINESS_STAGE_ID,
    STAGE5_0M_RECOMMENDED_NEXT_TASK,
    TAU_REQUIREMENT_MIN,
    build_stage5_0m_objective_readiness_review,
)


def test_stage5_0m_report_allows_stage5_1_plan_only() -> None:
    report = build_stage5_0m_objective_readiness_review()

    assert report["stage"] == STAGE5_0M_OBJECTIVE_READINESS_STAGE_ID
    assert report["tau_requirement_min"] == TAU_REQUIREMENT_MIN == 0.9
    decision = report["decision"]
    assert decision["stage5_1_plan_only_allowed"] is True
    assert decision["stage5_1_scope"] == "reward implementation plan without reward code"
    assert decision["recommended_next_task"] == STAGE5_0M_RECOMMENDED_NEXT_TASK
    assert decision["owner_approval_required_before_stage5_1"] is True


def test_stage5_0m_keeps_reward_training_model_and_tau_selection_blocked() -> None:
    report = build_stage5_0m_objective_readiness_review()
    checks = report["checks"]
    decision = report["decision"]

    assert checks["final_tau_selected"] is False
    assert checks["final_tau_below_requirement_selected"] is False
    assert checks["reward_code_allowed"] is False
    assert checks["reward_weight_calibration_allowed"] is False
    assert checks["training_allowed"] is False
    assert checks["actor_critic_model_work_allowed"] is False
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False
    assert decision["reward_code_allowed"] is False
    assert decision["training_allowed"] is False


def test_stage5_0m_readiness_gates_all_pass() -> None:
    report = build_stage5_0m_objective_readiness_review()

    gates = {gate["gate_id"]: gate for gate in report["readiness_gates"]}
    required = {
        "tau_requirement_baseline_gate",
        "objective_contract_semantics_gate",
        "metric_governance_gate",
        "failure_diagnosis_gate",
        "stage3_backed_evidence_gate",
        "feasibility_lever_evidence_gate",
        "negative_control_gate",
        "stage5_1_scope_gate",
        "conservative_fault_filter_awareness_gate",
    }
    assert required.issubset(gates)
    assert all(gate["passed"] is True for gate in gates.values())
    assert report["checks"]["all_readiness_gates_passed"] is True
    assert report["checks"]["additional_stage5_0_tasks_required_before_stage5_1_plan"] is False


def test_stage5_0m_summarizes_stage3_backed_evidence() -> None:
    report = build_stage5_0m_objective_readiness_review()
    summary = report["evidence_summary"]

    assert summary["stage5_0h_diagnosis"]["feasible_family_count"] == 4
    assert summary["stage5_0h_diagnosis"]["infeasible_family_count"] == 4
    assert summary["stage5_0k_stage3_backed_sweep"]["all_rows_stage3_backed"] is True
    assert summary["stage5_0l_range_review"]["all_rows_stage3_backed"] is True
    assert summary["stage5_0k_stage3_backed_sweep"]["all_rows_use_finite_blocklength"] is True
    assert summary["stage5_0l_range_review"]["all_rows_use_finite_blocklength"] is True
    assert set(summary["stage3_backed_sweep_coverage"]) == {
        "bandwidth_sweep",
        "deadline_sweep",
        "payload_sweep",
        "resource_budget_limit_sweep",
        "resource_orthogonalization_sweep",
        "rsu_height_placement_sweep",
        "tx_power_sweep",
    }


def test_stage5_0m_metric_governance_adds_no_metric_names() -> None:
    report = build_stage5_0m_objective_readiness_review()

    assert report["metric_governance"]["metric_valued_fields"] == [
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    ]
    assert report["metric_governance"]["new_metric_names_introduced"] == []
    assert report["checks"]["registered_metric_fields_only"] is True
    assert report["checks"]["new_metric_names_introduced"] is False
