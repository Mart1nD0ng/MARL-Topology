"""Stage 5.8 design-only learning target and replay contract.

This module is a structured manifest. It does not write datasets, create replay
buffers, assemble learner batches, run updates, or change active actor inputs.
"""

from __future__ import annotations

from .replay_schema import (
    DEPLOYMENT_ACTOR_INPUT_COLUMNS,
    REWARD_TRAINING_ONLY_COLUMNS,
    UNSUPPORTED_REPLAY_COLUMNS,
)


STAGE5_8_LEARNING_TARGET_REPLAY_STAGE_ID = (
    "stage_5_8_learning_target_replay_contract_without_implementation"
)
STAGE5_8_VERDICT = "learning_target_replay_contract_frozen_implementation_blocked"
STAGE5_8_RECOMMENDED_NEXT_TASK = (
    "stage_5_9_training_run_manifest_artifact_contract_without_execution"
)

PLANNED_LEARNING_TARGET_COLUMNS = (
    "return",
    "advantage",
    "value_target",
    "episode_id",
    "trajectory_id",
    "transition_index",
    "discount_factor",
    "bootstrap_value",
)

PLANNED_DIAGNOSTIC_COLUMNS = (
    "policy_log_probability",
    "policy_entropy",
    "action_sparsity",
    "constraint_violation",
)


def build_stage5_8_learning_target_replay_contract() -> dict[str, object]:
    """Return the Stage 5.8 learning-target replay contract."""

    return {
        "stage": STAGE5_8_LEARNING_TARGET_REPLAY_STAGE_ID,
        "verdict": STAGE5_8_VERDICT,
        "dataset_writer_allowed": False,
        "replay_buffer_allowed": False,
        "learner_batch_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "checkpoint_code_added": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "owner_decision_required": True,
        "recommended_next_task": STAGE5_8_RECOMMENDED_NEXT_TASK,
        "controlled_object": "future learning-target and replay-column boundary",
        "active_columns_unchanged": _active_columns_unchanged(),
        "planned_learning_target_columns": _planned_learning_target_columns(),
        "planned_diagnostic_columns": _planned_diagnostic_columns(),
        "actor_batch_contract": _actor_batch_contract(),
        "centralized_view_contract": _centralized_view_contract(),
        "derivation_policy": _derivation_policy(),
        "validation_gates": _validation_gates(),
        "blocked_until_future_approval": _blocked_until_future_approval(),
        "checks": _checks(),
    }


def _active_columns_unchanged() -> dict[str, object]:
    return {
        "deployment_actor_input_columns": sorted(DEPLOYMENT_ACTOR_INPUT_COLUMNS),
        "currently_admitted_training_only_columns": sorted(
            REWARD_TRAINING_ONLY_COLUMNS
        ),
        "planned_target_columns_remain_inactive": True,
        "active_replay_schema_changed": False,
    }


def _planned_learning_target_columns() -> tuple[dict[str, object], ...]:
    return tuple(
        _planned_column(
            name=column_name,
            column_class="future_training_target",
            used_for="future learner target construction",
        )
        for column_name in PLANNED_LEARNING_TARGET_COLUMNS
    )


def _planned_diagnostic_columns() -> tuple[dict[str, object], ...]:
    return tuple(
        _planned_column(
            name=column_name,
            column_class="future_training_diagnostic",
            used_for="future learning diagnostics only",
        )
        for column_name in PLANNED_DIAGNOSTIC_COLUMNS
    )


def _planned_column(name: str, column_class: str, used_for: str) -> dict[str, object]:
    return {
        "name": name,
        "status": "planned_not_active",
        "column_class": column_class,
        "used_for": used_for,
        "actor_input_allowed": False,
        "metric_column": False,
        "requires_future_schema_update": True,
        "requires_owner_approval": True,
    }


def _actor_batch_contract() -> dict[str, object]:
    return {
        "actor_batch_columns": sorted(DEPLOYMENT_ACTOR_INPUT_COLUMNS),
        "planned_target_columns_actor_input_allowed": False,
        "centralized_view_actor_input_allowed": False,
        "evaluation_metric_actor_input_allowed": False,
        "oracle_label_actor_input_allowed": False,
        "future_outcome_actor_input_allowed": False,
    }


def _centralized_view_contract() -> dict[str, object]:
    return {
        "status": "future_training_only_not_active_dataset",
        "may_reference_global_state": True,
        "deployment_actor_receives_view": False,
        "future_allowed_context": [
            "scenario_id",
            "node_ids",
            "candidate_edge_ids",
            "selected_edge_ids",
            "joint_action",
            "global_topology",
            "registered_evaluation_metrics",
            "surrogate_diagnostics",
        ],
        "future_required_negative_check": (
            "centralized view columns must be dropped before actor projection"
        ),
    }


def _derivation_policy() -> dict[str, object]:
    return {
        "status": "planned_not_computed",
        "discount_policy": "future_explicit_config_required",
        "bootstrap_policy": "future_value_estimator_contract_required",
        "episode_boundary_policy": "future_done_and_truncation_contract_required",
        "normalization_policy": "future_training_config_required",
        "single_step_target_policy": "future_contract_required",
        "uses_registered_evaluation_as_success_evidence": True,
    }


def _validation_gates() -> tuple[str, ...]:
    return (
        "planned_target_columns_rejected_by_active_replay_schema",
        "planned_target_columns_rejected_by_actor_input_projection",
        "active_actor_columns_unchanged",
        "dataset_writer_absent",
        "learner_batch_absent",
        "no_training_execution",
    )


def _blocked_until_future_approval() -> tuple[str, ...]:
    return (
        "dataset_writer_implementation",
        "replay_buffer_implementation",
        "learner_batch_implementation",
        "target_derivation_implementation",
        "training_execution",
        "checkpoint_creation",
        "weight_calibration",
        "final_tau_selection",
        "legacy_code_migration",
    )


def _checks() -> dict[str, object]:
    planned_columns = set(PLANNED_LEARNING_TARGET_COLUMNS) | set(
        PLANNED_DIAGNOSTIC_COLUMNS
    )
    return {
        "dataset_writer_allowed": False,
        "replay_buffer_allowed": False,
        "learner_batch_allowed": False,
        "training_execution_allowed": False,
        "model_code_added": False,
        "checkpoint_code_added": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "planned_learning_targets_active": False,
        "planned_columns_are_actor_inputs": bool(
            planned_columns & DEPLOYMENT_ACTOR_INPUT_COLUMNS
        ),
        "core_learning_targets_still_unsupported": {
            "return",
            "advantage",
            "value_target",
        }.issubset(UNSUPPORTED_REPLAY_COLUMNS),
        "active_actor_columns_unchanged": True,
    }
