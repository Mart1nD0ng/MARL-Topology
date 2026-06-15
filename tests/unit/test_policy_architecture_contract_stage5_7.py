from marl_topology.policies import (
    STAGE5_7_POLICY_ARCHITECTURE_STAGE_ID,
    STAGE5_7_RECOMMENDED_NEXT_TASK,
    STAGE5_7_VERDICT,
    build_stage5_7_policy_architecture_contract,
)


def test_stage5_7_contract_blocks_implementation_and_training() -> None:
    contract = build_stage5_7_policy_architecture_contract()
    checks = contract["checks"]

    assert contract["stage"] == STAGE5_7_POLICY_ARCHITECTURE_STAGE_ID
    assert contract["verdict"] == STAGE5_7_VERDICT
    assert contract["implementation_allowed"] is False
    assert contract["training_execution_allowed"] is False
    assert contract["model_code_added"] is False
    assert contract["checkpoint_code_added"] is False
    assert contract["weight_calibration_performed"] is False
    assert contract["final_tau_selected"] is False
    assert contract["v5_code_migrated"] is False
    assert checks["implementation_allowed"] is False
    assert checks["training_execution_allowed"] is False
    assert checks["centralized_training_view_deployment_leakage"] is False


def test_stage5_7_deployment_actor_boundary_is_local_only() -> None:
    contract = build_stage5_7_policy_architecture_contract()
    actor = contract["deployment_actor_contract"]

    assert actor["schema_id"] == "actor_local_observation_architecture_boundary_v1"
    assert "local_neighbor_observations" in actor["allowed_inputs"]
    assert "local_history" in actor["allowed_inputs"]
    assert "global_topology" in actor["forbidden_inputs"]
    assert "registered_evaluation_metrics" in actor["forbidden_inputs"]
    assert "surrogate_diagnostics" in actor["forbidden_inputs"]
    assert "per_primary_reliability" in actor["forbidden_inputs"]
    assert actor["deployment_can_run_without_training_only_tensors"] is True


def test_stage5_7_architecture_options_are_design_only() -> None:
    contract = build_stage5_7_policy_architecture_contract()
    options = contract["architecture_options"]

    assert len(options) == 3
    assert {option["option_id"] for option in options} == {
        "local_reactive_edge_scorer",
        "local_history_edge_scorer",
        "local_message_aggregation_edge_scorer",
    }
    assert all(
        option["status"] == "allowed_for_future_design_not_implemented"
        for option in options
    )
    assert all(option["implementation_status"] == "not_implemented" for option in options)


def test_stage5_7_centralized_training_and_credit_are_not_deployed() -> None:
    contract = build_stage5_7_policy_architecture_contract()
    central = contract["centralized_training_contract"]
    credit = contract["credit_assignment_contract"]

    assert central["training_only_view_allowed"] is True
    assert central["deployment_actor_receives_training_only_view"] is False
    assert central["critic_status"] == "future_training_only_design_not_implemented"
    assert central["actor_export_rule"] == "export_actor_without_training_only_tensors"
    assert credit["status"] == "not_selected"
    assert credit["local_reward_override_allowed"] is False
    assert "ranking_fidelity" in credit["required_calibration"]


def test_stage5_7_serialization_and_next_task_are_explicit() -> None:
    contract = build_stage5_7_policy_architecture_contract()
    serialization = contract["serialization_contract"]

    assert contract["recommended_next_task"] == STAGE5_7_RECOMMENDED_NEXT_TASK
    assert STAGE5_7_RECOMMENDED_NEXT_TASK == (
        "stage_5_8_learning_target_replay_contract_without_implementation"
    )
    assert serialization["status"] == "planned_not_active"
    assert serialization["checkpoint_creation_allowed"] is False
    assert "critic_state_not_required_for_actor_load" in serialization["future_required_checks"]
    assert "policy_network_implementation" in contract["blocked_until_future_approval"]
