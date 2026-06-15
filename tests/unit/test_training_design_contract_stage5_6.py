from marl_topology.training import (
    STAGE5_6_RECOMMENDED_NEXT_TASK,
    STAGE5_6_TRAINING_DESIGN_STAGE_ID,
    STAGE5_6_VERDICT,
    build_stage5_6_training_design_contract,
)


def test_stage5_6_contract_blocks_training_execution() -> None:
    contract = build_stage5_6_training_design_contract()
    checks = contract["checks"]

    assert contract["stage"] == STAGE5_6_TRAINING_DESIGN_STAGE_ID
    assert contract["verdict"] == STAGE5_6_VERDICT
    assert contract["training_execution_allowed"] is False
    assert contract["training_code_added"] is False
    assert contract["model_code_added"] is False
    assert contract["weight_calibration_performed"] is False
    assert contract["final_tau_selected"] is False
    assert contract["v5_code_migrated"] is False
    assert contract["owner_decision_required"] is True
    assert checks["training_run"] is False
    assert checks["checkpoint_written"] is False
    assert checks["deployment_actor_boundary_unchanged"] is True


def test_stage5_6_scalarization_policy_is_design_only() -> None:
    contract = build_stage5_6_training_design_contract()
    policy = contract["scalarization_policy"]

    assert policy["policy_id"] == "constraint_first_component_policy_v1"
    assert policy["status"] == "design_only_no_weights_selected"
    assert policy["tau_policy"]["tau_requirement_min"] == 0.9
    assert policy["tau_policy"]["final_tau_selected"] is False
    assert policy["plateau_rule"] == "no_reliability_bonus_above_tau"
    assert policy["weight_policy"] == "future_owner_approved_config_required"
    assert "legacy_formula_copy" in policy["forbidden_terms"]


def test_stage5_6_learning_targets_are_planned_not_active() -> None:
    contract = build_stage5_6_training_design_contract()
    target = contract["learning_target_contract"]

    assert target["status"] == "planned_not_active"
    assert "reward_surrogate" in target["currently_admitted_training_columns"]
    assert "return" in target["future_columns_require_contract"]
    assert "advantage" in target["future_columns_require_contract"]
    assert "value_target" in target["future_columns_require_contract"]
    assert target["deployment_actor_input"] == "unchanged_actor_observation_only"
    assert target["no_dataset_writer_added"] is True


def test_stage5_6_information_boundary_and_next_task_are_explicit() -> None:
    contract = build_stage5_6_training_design_contract()
    boundary = contract["information_boundary"]

    assert contract["recommended_next_task"] == STAGE5_6_RECOMMENDED_NEXT_TASK
    assert STAGE5_6_RECOMMENDED_NEXT_TASK == (
        "stage_5_7_policy_architecture_contract_without_implementation"
    )
    assert "local_neighbor_observations" in boundary["deployment_actor_allowed"]
    assert "global_topology" in boundary["deployment_actor_forbidden"]
    assert "surrogate_diagnostics" in boundary["deployment_actor_forbidden"]
    assert boundary["architecture_status"] == "not_selected_not_implemented"


def test_stage5_6_scenario_baseline_artifact_and_stop_rules_are_required() -> None:
    contract = build_stage5_6_training_design_contract()

    assert contract["scenario_protocol"]["minimum_seed_plan"] == (
        "at_least_five_seeds_in_future_run_contract"
    )
    assert contract["baseline_protocol"]["full_graph_status"] == "baseline_not_oracle"
    assert contract["artifact_policy"]["status"] == "design_only_no_artifacts_written"
    assert contract["artifact_policy"]["checkpoint_creation_allowed"] is False
    assert "validation_plateau_rule" in contract["stop_conditions"]["future_required"]
    assert "training_execution" in contract["blocked_until_future_approval"]
