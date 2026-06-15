from marl_topology.data import ACTOR_BATCH_REQUIRED_FIELDS
from marl_topology.policies import (
    ACTOR_POLICY_ALLOWED_INPUT_FIELDS,
    ACTOR_POLICY_INPUT_SCHEMA_ID,
    ACTOR_POLICY_OUTPUT_SCHEMA_ID,
    ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID,
    LEGACY_ACTOR_EDGE_DECISION_OUTPUT_SCHEMA_ID,
    STAGE8_0_ACTOR_POLICY_INTERFACE_STAGE_ID,
    STAGE8_0_RECOMMENDED_NEXT_TASK,
    STAGE8_0_VERDICT,
    PolicyInterfaceViolation,
    build_stage8_0_actor_policy_interface_contract,
    validate_actor_policy_input_fields,
    validate_actor_policy_input_row,
    validate_actor_policy_output_fields,
    validate_legacy_actor_edge_decision_output_fields,
)


def test_stage8_0_contract_blocks_implementation_training_and_artifacts() -> None:
    contract = build_stage8_0_actor_policy_interface_contract()
    checks = contract["checks"]

    assert contract["stage"] == STAGE8_0_ACTOR_POLICY_INTERFACE_STAGE_ID
    assert contract["verdict"] == STAGE8_0_VERDICT
    assert contract["implementation_allowed"] is False
    assert contract["training_execution_allowed"] is False
    assert contract["model_code_added"] is False
    assert contract["checkpoint_code_added"] is False
    assert contract["artifact_write_allowed"] is False
    assert contract["weight_calibration_performed"] is False
    assert contract["final_tau_selected"] is False
    assert contract["v5_code_migrated"] is False
    assert checks["deployment_actor_boundary_local_only"] is True
    assert checks["objective_metrics_actor_input"] is False
    assert checks["learning_target_actor_input"] is False
    assert checks["surrogate_diagnostic_actor_input"] is False


def test_stage8_0_actor_input_schema_matches_actor_safe_batch_fields() -> None:
    contract = build_stage8_0_actor_policy_interface_contract()
    actor_input = contract["actor_input_contract"]

    assert ACTOR_POLICY_INPUT_SCHEMA_ID == "actor_policy_local_input_v1"
    assert ACTOR_POLICY_ALLOWED_INPUT_FIELDS == ACTOR_BATCH_REQUIRED_FIELDS
    assert actor_input["schema_id"] == ACTOR_POLICY_INPUT_SCHEMA_ID
    assert actor_input["allowed_fields"] == list(ACTOR_BATCH_REQUIRED_FIELDS)
    assert actor_input["objective_metrics_allowed"] is False
    assert actor_input["oracle_labels_allowed"] is False
    assert actor_input["surrogate_diagnostics_allowed"] is False
    assert actor_input["global_topology_allowed"] is False
    validate_actor_policy_input_fields(ACTOR_BATCH_REQUIRED_FIELDS)


def test_stage8_0_actor_input_validation_rejects_leakage_fields() -> None:
    forbidden_fields = [
        "global_topology",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
        "oracle_label",
        "learning_targets",
        "reward_surrogate",
        "critic_view",
        "future_consensus_outcome",
    ]
    for forbidden in forbidden_fields:
        fields = set(ACTOR_BATCH_REQUIRED_FIELDS)
        fields.add(forbidden)
        try:
            validate_actor_policy_input_fields(fields)
        except PolicyInterfaceViolation as exc:
            assert "forbidden policy input fields" in str(exc)
        else:
            raise AssertionError(f"forbidden actor policy field accepted: {forbidden}")


def test_stage8_0_actor_input_validation_rejects_missing_or_unknown_fields() -> None:
    missing_fields = set(ACTOR_BATCH_REQUIRED_FIELDS)
    missing_fields.remove("local_history")
    try:
        validate_actor_policy_input_fields(missing_fields)
    except PolicyInterfaceViolation as exc:
        assert "missing policy input fields" in str(exc)
    else:
        raise AssertionError("missing local_history was accepted")

    unknown_fields = set(ACTOR_BATCH_REQUIRED_FIELDS)
    unknown_fields.add("local_unregistered_feature")
    try:
        validate_actor_policy_input_fields(unknown_fields)
    except PolicyInterfaceViolation as exc:
        assert "unknown policy input fields" in str(exc)
    else:
        raise AssertionError("unknown local field was accepted")


def test_stage8_0_actor_input_row_validation_accepts_actor_safe_row() -> None:
    row = {
        "agent_id": "veh_0",
        "agent_kind": "vehicle",
        "time_step": 0,
        "local_position_m": (0.0, 0.0, 1.5),
        "local_neighbor_observations": (),
        "local_messages": (),
        "local_history": {},
    }

    validate_actor_policy_input_row(row)


def test_stage8_0_actor_output_schema_is_edge_score_only_after_stage9_0_unification() -> None:
    contract = build_stage8_0_actor_policy_interface_contract()
    actor_output = contract["actor_output_contract"]

    assert actor_output["schema_id"] == ACTOR_POLICY_OUTPUT_SCHEMA_ID
    assert ACTOR_POLICY_OUTPUT_SCHEMA_ID == ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID
    assert actor_output["output_role"] == "local_directed_edge_score_batch"
    assert actor_output["joint_topology_assembly"] == "environment_side_only"
    assert actor_output["may_output_global_topology"] is False
    assert actor_output["may_output_activate"] is False
    assert (
        actor_output["legacy_edge_decision_schema_id"]
        == LEGACY_ACTOR_EDGE_DECISION_OUTPUT_SCHEMA_ID
    )
    validate_actor_policy_output_fields(
        {
            "schema_id",
            "agent_id",
            "edge_scores",
            "policy_state_id",
            "contains_final_topology",
        }
    )


def test_stage8_0_actor_output_validation_rejects_activate_global_and_target_fields() -> None:
    forbidden_fields = [
        "activate",
        "global_topology",
        "selected_edge_ids",
        "selected_directed_edges",
        "joint_action",
        "oracle_label",
        "consensus_success_probability",
        "learning_targets",
    ]
    for forbidden in forbidden_fields:
        fields = {"agent_id", "edge_scores", forbidden}
        try:
            validate_actor_policy_output_fields(fields)
        except PolicyInterfaceViolation as exc:
            assert "forbidden policy output fields" in str(exc)
        else:
            raise AssertionError(f"forbidden policy output field accepted: {forbidden}")


def test_stage8_0_legacy_actor_edge_decision_schema_still_owns_activate() -> None:
    validate_legacy_actor_edge_decision_output_fields(
        {
            "agent_id",
            "neighbor_id",
            "edge_id",
            "activate",
            "action_probability",
        }
    )


def test_stage8_0_training_only_views_remain_reference_only() -> None:
    contract = build_stage8_0_actor_policy_interface_contract()
    critic = contract["critic_training_interface_reference"]
    target = contract["learning_target_boundary"]

    assert critic["status"] == "reference_only_not_implemented"
    assert critic["centralized_view_allowed_for_future_training"] is True
    assert critic["deployment_actor_receives_centralized_view"] is False
    assert critic["actor_export_includes_critic_state"] is False
    assert target["actor_input_receives_learning_targets"] is False
    assert target["actor_input_receives_objective_metrics"] is False
    assert target["policy_interface_selects_credit_method"] is False
    assert contract["recommended_next_task"] == STAGE8_0_RECOMMENDED_NEXT_TASK
