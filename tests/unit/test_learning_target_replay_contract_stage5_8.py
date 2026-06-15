import pytest

from marl_topology.data import (
    PLANNED_LEARNING_TARGET_COLUMNS,
    STAGE5_8_LEARNING_TARGET_REPLAY_STAGE_ID,
    STAGE5_8_RECOMMENDED_NEXT_TASK,
    STAGE5_8_VERDICT,
    ReplayColumnViolation,
    build_stage5_8_learning_target_replay_contract,
    validate_deployment_actor_input_columns,
    validate_registered_replay_columns,
)


def test_stage5_8_contract_blocks_replay_and_training_implementation() -> None:
    contract = build_stage5_8_learning_target_replay_contract()
    checks = contract["checks"]

    assert contract["stage"] == STAGE5_8_LEARNING_TARGET_REPLAY_STAGE_ID
    assert contract["verdict"] == STAGE5_8_VERDICT
    assert contract["dataset_writer_allowed"] is False
    assert contract["replay_buffer_allowed"] is False
    assert contract["learner_batch_allowed"] is False
    assert contract["training_execution_allowed"] is False
    assert contract["model_code_added"] is False
    assert contract["checkpoint_code_added"] is False
    assert contract["weight_calibration_performed"] is False
    assert contract["final_tau_selected"] is False
    assert contract["v5_code_migrated"] is False
    assert checks["planned_learning_targets_active"] is False
    assert checks["planned_columns_are_actor_inputs"] is False


def test_stage5_8_planned_learning_target_columns_are_explicit() -> None:
    contract = build_stage5_8_learning_target_replay_contract()
    columns = {
        column["name"]: column for column in contract["planned_learning_target_columns"]
    }

    assert set(PLANNED_LEARNING_TARGET_COLUMNS) == set(columns)
    for name in ("return", "advantage", "value_target", "discount_factor"):
        assert columns[name]["status"] == "planned_not_active"
        assert columns[name]["actor_input_allowed"] is False
        assert columns[name]["metric_column"] is False
        assert columns[name]["requires_future_schema_update"] is True


def test_stage5_8_active_replay_schema_still_rejects_core_learning_targets() -> None:
    contract = build_stage5_8_learning_target_replay_contract()

    assert contract["checks"]["core_learning_targets_still_unsupported"] is True
    for column in ("return", "advantage", "value_target"):
        with pytest.raises(ReplayColumnViolation):
            validate_registered_replay_columns([column])
        with pytest.raises(ReplayColumnViolation):
            validate_deployment_actor_input_columns([column])


def test_stage5_8_actor_batch_and_centralized_view_boundaries_are_explicit() -> None:
    contract = build_stage5_8_learning_target_replay_contract()
    actor = contract["actor_batch_contract"]
    central = contract["centralized_view_contract"]

    assert "local_neighbor_observations" in actor["actor_batch_columns"]
    assert actor["planned_target_columns_actor_input_allowed"] is False
    assert actor["centralized_view_actor_input_allowed"] is False
    assert actor["future_outcome_actor_input_allowed"] is False
    assert central["status"] == "future_training_only_not_active_dataset"
    assert central["deployment_actor_receives_view"] is False
    assert "global_topology" in central["future_allowed_context"]


def test_stage5_8_next_task_and_derivation_policy_are_design_only() -> None:
    contract = build_stage5_8_learning_target_replay_contract()
    policy = contract["derivation_policy"]

    assert contract["recommended_next_task"] == STAGE5_8_RECOMMENDED_NEXT_TASK
    assert STAGE5_8_RECOMMENDED_NEXT_TASK == (
        "stage_5_9_training_run_manifest_artifact_contract_without_execution"
    )
    assert policy["status"] == "planned_not_computed"
    assert policy["discount_policy"] == "future_explicit_config_required"
    assert policy["bootstrap_policy"] == "future_value_estimator_contract_required"
    assert "dataset_writer_implementation" in contract["blocked_until_future_approval"]
