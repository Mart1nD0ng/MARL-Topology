from marl_topology.evaluation import (
    STAGE5_5_RECOMMENDED_NEXT_TASK,
    STAGE5_5_TRAINING_PREFLIGHT_STAGE_ID,
    STAGE5_5_VERDICT,
    build_stage5_5_training_preflight_review,
)


def test_stage5_5_preflight_blocks_training_execution() -> None:
    report = build_stage5_5_training_preflight_review()
    checks = report["checks"]

    assert report["stage"] == STAGE5_5_TRAINING_PREFLIGHT_STAGE_ID
    assert report["verdict"] == STAGE5_5_VERDICT
    assert report["training_execution_allowed"] is False
    assert report["training_design_contract_allowed"] is True
    assert report["recommended_next_task"] == STAGE5_5_RECOMMENDED_NEXT_TASK
    assert report["owner_decision_required"] is True
    assert checks["training_execution_allowed"] is False
    assert checks["training_run"] is False
    assert checks["model_code_added"] is False
    assert checks["v5_code_migrated"] is False
    assert checks["reward_weight_calibration_performed"] is False
    assert checks["final_tau_selected"] is False


def test_stage5_5_preflight_exposes_pass_blocked_and_deferred_gates() -> None:
    report = build_stage5_5_training_preflight_review()
    summary = report["gate_summary"]
    gates = {gate["gate_id"]: gate for gate in report["gates"]}

    assert summary["pass_count"] >= 6
    assert summary["blocked_count"] >= 5
    assert summary["deferred_count"] >= 3
    assert "metric_governance_ready" in gates
    assert "surrogate_weight_policy_missing" in gates
    assert "return_advantage_target_contract_missing" in gates
    assert "actor_critic_architecture_contract_missing" in gates
    assert "training_artifact_policy_missing" in gates
    assert "multi_seed_protocol_missing" in gates
    assert "scenario_distribution_calibration_deferred" in gates
    assert gates["metric_governance_ready"]["status"] == "pass"
    assert gates["surrogate_weight_policy_missing"]["status"] == "blocked"
    assert gates["scenario_distribution_calibration_deferred"]["status"] == "deferred"
    assert gates["surrogate_weight_policy_missing"]["blocks_training_execution"] is True
    assert report["checks"]["all_blocking_gates_visible"] is True


def test_stage5_5_preflight_consumes_stage5_4_component_report() -> None:
    report = build_stage5_5_training_preflight_review()
    checks = report["checks"]

    assert report["source_stage"] == "stage_5_4_reward_report_integration_without_training"
    assert checks["source_stage5_4_report_available"] is True
    assert checks["component_only_report_policy"] is True
    assert checks["scalar_surrogate_not_reported"] is True
    assert checks["surrogate_outputs_not_metrics"] is True
    assert checks["normalization_reference_applied"] is True


def test_stage5_5_required_items_before_training_execution_are_explicit() -> None:
    report = build_stage5_5_training_preflight_review()
    required = set(report["required_before_training_execution"])

    assert "stage_5_6_training_design_contract_without_execution" in required
    assert "surrogate_scalarization_and_weight_policy" in required
    assert "learning_target_and_replay_column_contract" in required
    assert "actor_critic_architecture_contract" in required
    assert "artifact_seed_and_run_manifest_policy" in required
    assert "multi_seed_evaluation_protocol" in required
    assert "owner_approval_for_training_execution" in required
    assert "training_execution" in report["blocked_tasks"]
