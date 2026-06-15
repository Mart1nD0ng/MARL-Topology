"""Stage 8.0 actor policy interface contract.

This module freezes the deployment policy boundary. It does not implement a
network, value estimator, learner, checkpoint, or training execution path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from marl_topology.data.actor_batch import ACTOR_BATCH_REQUIRED_FIELDS
from marl_topology.data.learning_evidence import ACTOR_VIEW_FORBIDDEN_FIELDS
from marl_topology.env import ACTOR_FORBIDDEN_FIELDS


STAGE8_0_ACTOR_POLICY_INTERFACE_STAGE_ID = (
    "stage_8_0_actor_policy_interface_contract_with_owner_approval"
)
STAGE8_0_VERDICT = "actor_policy_interface_contract_frozen_implementation_blocked"
STAGE8_0_RECOMMENDED_NEXT_TASK = (
    "stage_8_1_actor_policy_interface_skeleton_without_model_or_training"
)
ACTOR_POLICY_INPUT_SCHEMA_ID = "actor_policy_local_input_v1"
ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID = "actor_policy_local_edge_score_output_v1"
LEGACY_ACTOR_EDGE_DECISION_OUTPUT_SCHEMA_ID = "actor_policy_local_edge_decision_v1"
ACTOR_POLICY_OUTPUT_SCHEMA_ID = ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID
CRITIC_TRAINING_INTERFACE_SCHEMA_ID = "critic_training_view_reference_v1"

ACTOR_POLICY_ALLOWED_INPUT_FIELDS = tuple(ACTOR_BATCH_REQUIRED_FIELDS)
ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS = tuple(
    sorted(
        ACTOR_FORBIDDEN_FIELDS
        | ACTOR_VIEW_FORBIDDEN_FIELDS
        | {
            "all_node_distances",
            "complete_candidate_edges",
            "edge_delta_targets",
            "feasibility_label",
            "learning_target_view",
            "objective_value",
            "oracle_candidate_id",
            "per_primary_reliability",
            "quality_report_flags",
            "surrogate_components",
        }
    )
)
ACTOR_POLICY_OUTPUT_FIELDS = (
    "schema_id",
    "agent_id",
    "edge_scores",
    "policy_state_id",
    "contains_final_topology",
    "contains_oracle_label",
    "contains_consensus_metric",
)
LEGACY_ACTOR_EDGE_DECISION_OUTPUT_FIELDS = (
    "agent_id",
    "neighbor_id",
    "edge_id",
    "action_kind",
    "activate",
    "action_score",
    "action_probability",
    "policy_state_id",
    "decision_role",
)
ACTOR_POLICY_FORBIDDEN_OUTPUT_FIELDS = (
    "activate",
    "global_topology",
    "selected_edge_ids",
    "selected_directed_edges",
    "joint_action",
    "oracle_label",
    "consensus_success_probability",
    "latency",
    "energy",
    "learning_targets",
    "edge_delta_targets",
    "reward_surrogate",
)


class PolicyInterfaceViolation(ValueError):
    """Raised when the Stage 8.0 policy interface boundary is violated."""


def build_stage8_0_actor_policy_interface_contract() -> dict[str, object]:
    """Return the Stage 8.0 actor policy interface contract."""

    return {
        "stage": STAGE8_0_ACTOR_POLICY_INTERFACE_STAGE_ID,
        "verdict": STAGE8_0_VERDICT,
        "implementation_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "checkpoint_code_added": False,
        "artifact_write_allowed": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "owner_decision_required": True,
        "recommended_next_task": STAGE8_0_RECOMMENDED_NEXT_TASK,
        "controlled_object": "deployment actor policy interface boundary",
        "actor_input_contract": _actor_input_contract(),
        "actor_output_contract": _actor_output_contract(),
        "critic_training_interface_reference": _critic_training_interface_reference(),
        "learning_target_boundary": _learning_target_boundary(),
        "state_and_history_contract": _state_and_history_contract(),
        "deployment_assembly_contract": _deployment_assembly_contract(),
        "blocked_until_future_approval": _blocked_until_future_approval(),
        "required_gates": _required_gates(),
        "checks": _checks(),
    }


def validate_actor_policy_input_fields(fields: Iterable[str]) -> None:
    """Require policy input rows to match the Stage 6 actor-safe field set."""

    field_set = {str(field) for field in fields}
    forbidden = sorted(field_set & set(ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS))
    if forbidden:
        raise PolicyInterfaceViolation(f"forbidden policy input fields: {forbidden}")
    required = set(ACTOR_POLICY_ALLOWED_INPUT_FIELDS)
    missing = sorted(required - field_set)
    extra = sorted(field_set - required)
    if missing:
        raise PolicyInterfaceViolation(f"missing policy input fields: {missing}")
    if extra:
        raise PolicyInterfaceViolation(f"unknown policy input fields: {extra}")


def validate_actor_policy_output_fields(fields: Iterable[str]) -> None:
    """Require active actor outputs to be edge-score batches, not hard actions."""

    field_set = {str(field) for field in fields}
    forbidden = sorted(field_set & set(ACTOR_POLICY_FORBIDDEN_OUTPUT_FIELDS))
    if forbidden:
        raise PolicyInterfaceViolation(f"forbidden policy output fields: {forbidden}")
    allowed = set(ACTOR_POLICY_OUTPUT_FIELDS)
    missing = sorted({"agent_id", "edge_scores"} - field_set)
    extra = sorted(field_set - allowed)
    if missing:
        raise PolicyInterfaceViolation(f"missing policy output fields: {missing}")
    if extra:
        raise PolicyInterfaceViolation(f"unknown policy output fields: {extra}")


def validate_legacy_actor_edge_decision_output_fields(fields: Iterable[str]) -> None:
    """Validate the legacy local edge-decision schema that still owns activate."""

    field_set = {str(field) for field in fields}
    forbidden = sorted(
        field_set
        & {
            "global_topology",
            "selected_edge_ids",
            "selected_directed_edges",
            "joint_action",
            "oracle_label",
            "consensus_success_probability",
            "latency",
            "energy",
            "learning_targets",
            "edge_delta_targets",
            "reward_surrogate",
        }
    )
    if forbidden:
        raise PolicyInterfaceViolation(
            f"forbidden legacy edge-decision output fields: {forbidden}"
        )
    allowed = set(LEGACY_ACTOR_EDGE_DECISION_OUTPUT_FIELDS)
    missing = sorted({"agent_id", "neighbor_id", "edge_id", "activate"} - field_set)
    extra = sorted(field_set - allowed)
    if missing:
        raise PolicyInterfaceViolation(
            f"missing legacy edge-decision output fields: {missing}"
        )
    if extra:
        raise PolicyInterfaceViolation(
            f"unknown legacy edge-decision output fields: {extra}"
        )


def validate_actor_policy_input_row(row: Mapping[str, object]) -> None:
    """Validate one policy input row by field names only."""

    validate_actor_policy_input_fields(row.keys())


def _actor_input_contract() -> dict[str, object]:
    return {
        "schema_id": ACTOR_POLICY_INPUT_SCHEMA_ID,
        "source": "stage6_actor_safe_batch_rows",
        "allowed_fields": list(ACTOR_POLICY_ALLOWED_INPUT_FIELDS),
        "forbidden_fields": list(ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS),
        "required_locality": "incident_neighbors_local_messages_and_local_history_only",
        "objective_metrics_allowed": False,
        "oracle_labels_allowed": False,
        "future_outcomes_allowed": False,
        "surrogate_diagnostics_allowed": False,
        "global_topology_allowed": False,
    }


def _actor_output_contract() -> dict[str, object]:
    return {
        "schema_id": ACTOR_POLICY_OUTPUT_SCHEMA_ID,
        "output_role": "local_directed_edge_score_batch",
        "allowed_fields": list(ACTOR_POLICY_OUTPUT_FIELDS),
        "forbidden_fields": list(ACTOR_POLICY_FORBIDDEN_OUTPUT_FIELDS),
        "required_fields": ["agent_id", "edge_scores"],
        "legacy_edge_decision_schema_id": LEGACY_ACTOR_EDGE_DECISION_OUTPUT_SCHEMA_ID,
        "legacy_activate_owner": "legacy_edge_decision_or_assembler_selected_topology",
        "joint_topology_assembly": "environment_side_only",
        "must_output_local_scores": True,
        "may_output_activate": False,
        "may_output_global_topology": False,
    }


def _critic_training_interface_reference() -> dict[str, object]:
    return {
        "schema_id": CRITIC_TRAINING_INTERFACE_SCHEMA_ID,
        "status": "reference_only_not_implemented",
        "centralized_view_allowed_for_future_training": True,
        "deployment_actor_receives_centralized_view": False,
        "allowed_future_sources": [
            "stage7_critic_centralized_view",
            "stage7_learning_target_view",
            "stage7_diagnostics_view",
        ],
        "actor_export_includes_critic_state": False,
    }


def _learning_target_boundary() -> dict[str, object]:
    return {
        "stage7_edge_delta_targets": "training_only",
        "oracle_candidates": "diagnostic_or_training_target_only",
        "surrogate_components": "training_or_diagnostic_only",
        "actor_input_receives_learning_targets": False,
        "actor_input_receives_objective_metrics": False,
        "policy_interface_selects_credit_method": False,
    }


def _state_and_history_contract() -> dict[str, object]:
    return {
        "stateless_actor_allowed": True,
        "local_history_actor_allowed": True,
        "permitted_message_actor_allowed": True,
        "global_recurrent_state_allowed": False,
        "history_must_be_derived_from_actor_allowed_fields": True,
    }


def _deployment_assembly_contract() -> dict[str, object]:
    return {
        "actor_outputs": "local_directed_edge_scores_only",
        "joint_topology_action": "assembled_by_environment_or_controller",
        "candidate_edge_validation": "environment_side",
        "hard_activation_owner": "environment_side_topology_assembler",
        "full_graph_baseline_is_oracle": False,
        "oracle_candidate_actor_input": False,
    }


def _blocked_until_future_approval() -> tuple[str, ...]:
    return (
        "policy_model_implementation",
        "critic_model_implementation",
        "learner_update_implementation",
        "checkpoint_creation",
        "training_execution",
        "artifact_export_beyond_existing_evidence_dataset",
        "credit_method_selection",
        "reward_weight_calibration",
        "final_tau_selection",
        "legacy_code_migration",
    )


def _required_gates() -> tuple[str, ...]:
    return (
        "actor_policy_input_schema_gate",
        "actor_policy_output_schema_gate",
        "dec_pomdp_leakage_gate",
        "stage7_target_separation_gate",
        "critic_training_view_separation_gate",
        "model_implementation_block_gate",
        "training_execution_block_gate",
    )


def _checks() -> dict[str, object]:
    return {
        "actor_policy_input_schema_declared": True,
        "actor_policy_output_schema_declared": True,
        "deployment_actor_boundary_local_only": True,
        "objective_metrics_actor_input": False,
        "oracle_label_actor_input": False,
        "learning_target_actor_input": False,
        "surrogate_diagnostic_actor_input": False,
        "centralized_training_view_deployment_leakage": False,
        "joint_topology_assembled_by_actor": False,
        "implementation_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "checkpoint_code_added": False,
        "artifact_write_allowed": False,
        "v5_code_migrated": False,
    }
