"""Stage 5.7 design-only policy architecture contract.

This module is a manifest for architecture boundaries. It does not implement
policy networks, value estimators, updates, checkpoints, or deployment code.
"""

from __future__ import annotations


STAGE5_7_POLICY_ARCHITECTURE_STAGE_ID = (
    "stage_5_7_policy_architecture_contract_without_implementation"
)
STAGE5_7_VERDICT = "architecture_contract_frozen_implementation_blocked"
STAGE5_7_RECOMMENDED_NEXT_TASK = (
    "stage_5_8_learning_target_replay_contract_without_implementation"
)


def build_stage5_7_policy_architecture_contract() -> dict[str, object]:
    """Return the Stage 5.7 policy architecture contract."""

    return {
        "stage": STAGE5_7_POLICY_ARCHITECTURE_STAGE_ID,
        "verdict": STAGE5_7_VERDICT,
        "implementation_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "checkpoint_code_added": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "owner_decision_required": True,
        "recommended_next_task": STAGE5_7_RECOMMENDED_NEXT_TASK,
        "controlled_object": (
            "future policy and centralized-training architecture boundary"
        ),
        "deployment_actor_contract": _deployment_actor_contract(),
        "architecture_options": _architecture_options(),
        "centralized_training_contract": _centralized_training_contract(),
        "credit_assignment_contract": _credit_assignment_contract(),
        "serialization_contract": _serialization_contract(),
        "leakage_tests_required": _leakage_tests_required(),
        "blocked_until_future_approval": _blocked_until_future_approval(),
        "checks": _checks(),
    }


def _deployment_actor_contract() -> dict[str, object]:
    return {
        "schema_id": "actor_local_observation_architecture_boundary_v1",
        "allowed_inputs": [
            "agent_id",
            "agent_kind",
            "time_step",
            "local_position_m",
            "local_neighbor_observations",
            "local_messages",
            "local_history",
        ],
        "forbidden_inputs": [
            "global_topology",
            "candidate_graph_full_state",
            "all_node_positions",
            "oracle_labels",
            "registered_evaluation_metrics",
            "surrogate_diagnostics",
            "future_outcomes",
            "centralized_training_view",
            "per_primary_reliability",
        ],
        "output_contract": "local_edge_action_decisions_only",
        "history_rule": "local_or_permitted_message_history_only",
        "deployment_can_run_without_training_only_tensors": True,
    }


def _architecture_options() -> tuple[dict[str, object], ...]:
    return (
        {
            "option_id": "local_reactive_edge_scorer",
            "status": "allowed_for_future_design_not_implemented",
            "actor_inputs": "deployment_actor_allowed_inputs_only",
            "memory": "none",
            "message_use": "permitted_local_messages_only",
            "implementation_status": "not_implemented",
        },
        {
            "option_id": "local_history_edge_scorer",
            "status": "allowed_for_future_design_not_implemented",
            "actor_inputs": "deployment_actor_allowed_inputs_only",
            "memory": "local_history_only",
            "message_use": "permitted_local_messages_only",
            "implementation_status": "not_implemented",
        },
        {
            "option_id": "local_message_aggregation_edge_scorer",
            "status": "allowed_for_future_design_not_implemented",
            "actor_inputs": "deployment_actor_allowed_inputs_only",
            "memory": "local_or_permitted_message_history_only",
            "message_use": "declared_neighbor_messages_only",
            "implementation_status": "not_implemented",
        },
    )


def _centralized_training_contract() -> dict[str, object]:
    return {
        "training_only_view_allowed": True,
        "deployment_actor_receives_training_only_view": False,
        "allowed_training_only_inputs": [
            "scenario_id",
            "node_ids",
            "candidate_edge_ids",
            "selected_edge_ids",
            "joint_action",
            "global_topology",
            "registered_evaluation_metrics",
            "surrogate_diagnostics",
            "oracle_status",
        ],
        "critic_status": "future_training_only_design_not_implemented",
        "actor_export_rule": "export_actor_without_training_only_tensors",
    }


def _credit_assignment_contract() -> dict[str, object]:
    return {
        "status": "not_selected",
        "allowed_future_review_paths": [
            "centralized_value_estimator",
            "counterfactual_edge_credit_review",
            "difference_against_baseline_review",
        ],
        "required_calibration": [
            "ranking_fidelity",
            "sign_accuracy",
            "rare_failure_recall",
            "magnitude_error",
            "oracle_edge_hit_rate",
        ],
        "local_reward_override_allowed": False,
        "credit_signal_primary_evidence": "registered_objective_metrics",
    }


def _serialization_contract() -> dict[str, object]:
    return {
        "status": "planned_not_active",
        "checkpoint_creation_allowed": False,
        "future_required_checks": [
            "actor_export_contains_only_deployment_inputs",
            "critic_state_not_required_for_actor_load",
            "feature_schema_hash_recorded",
            "contract_ids_recorded",
            "metric_registry_version_recorded",
        ],
    }


def _leakage_tests_required() -> tuple[str, ...]:
    return (
        "actor_schema_rejects_global_topology",
        "actor_schema_rejects_metrics_and_surrogate_diagnostics",
        "actor_schema_rejects_oracle_labels",
        "actor_schema_rejects_future_outcomes",
        "deployment_export_excludes_training_only_view",
        "replay_actor_projection_excludes_training_only_columns",
    )


def _blocked_until_future_approval() -> tuple[str, ...]:
    return (
        "policy_network_implementation",
        "critic_network_implementation",
        "training_execution",
        "checkpoint_creation",
        "credit_assignment_selection",
        "final_tau_selection",
        "legacy_code_migration",
    )


def _checks() -> dict[str, object]:
    return {
        "implementation_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "checkpoint_code_added": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "deployment_actor_boundary_local_only": True,
        "centralized_training_view_deployment_leakage": False,
        "credit_assignment_selected": False,
        "full_graph_is_oracle": False,
        "oracle_candidate_actor_input": False,
    }
