"""Stage 5.6 design-only training contract.

This module is a structured contract manifest. It does not create learners,
run updates, write artifacts, or instantiate policy networks.
"""

from __future__ import annotations


STAGE5_6_TRAINING_DESIGN_STAGE_ID = (
    "stage_5_6_training_design_contract_without_execution"
)
STAGE5_6_VERDICT = "design_contract_frozen_training_execution_blocked"
STAGE5_6_RECOMMENDED_NEXT_TASK = (
    "stage_5_7_policy_architecture_contract_without_implementation"
)


def build_stage5_6_training_design_contract() -> dict[str, object]:
    """Return the Stage 5.6 design contract as structured data."""

    return {
        "stage": STAGE5_6_TRAINING_DESIGN_STAGE_ID,
        "verdict": STAGE5_6_VERDICT,
        "training_execution_allowed": False,
        "training_code_added": False,
        "model_code_added": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "owner_decision_required": True,
        "recommended_next_task": STAGE5_6_RECOMMENDED_NEXT_TASK,
        "controlled_object": (
            "future decentralized MARL training workflow for topology decisions"
        ),
        "training_purpose": _training_purpose(),
        "preconditions": _preconditions(),
        "scalarization_policy": _scalarization_policy(),
        "learning_target_contract": _learning_target_contract(),
        "information_boundary": _information_boundary(),
        "scenario_protocol": _scenario_protocol(),
        "baseline_protocol": _baseline_protocol(),
        "diagnostics": _diagnostics(),
        "stop_conditions": _stop_conditions(),
        "artifact_policy": _artifact_policy(),
        "blocked_until_future_approval": _blocked_until_future_approval(),
        "checks": _checks(),
    }


def _training_purpose() -> dict[str, object]:
    return {
        "purpose": (
            "learn decentralized edge-activation behavior that satisfies the "
            "consensus reliability requirement while reducing latency and energy"
        ),
        "deployment_policy_boundary": "actor_local_observation_only",
        "training_paradigm": "ctde_allowed_not_algorithm_commitment",
        "evaluation_objective": {
            "constraint": "consensus_success_probability >= tau_requirement_min",
            "primary_objectives": ["minimize_latency", "minimize_energy"],
            "diagnostics_only": [
                "topology_diagnostics",
                "per_primary_reliability",
                "surrogate_components",
            ],
        },
    }


def _preconditions() -> tuple[dict[str, object], ...]:
    return (
        _gate("metric_governance", "pass", "Stage 5.5 preflight gate passed."),
        _gate("objective_contract", "pass", "Stage 5.0 objective contract exists."),
        _gate("surrogate_interface", "pass", "Stage 5.2 pure interface exists."),
        _gate("normalization_reference", "pass", "Stage 5.3 references exist."),
        _gate("component_report", "pass", "Stage 5.4 component report exists."),
        _gate("dec_pomdp_boundary", "pass", "ActorObservation boundary exists."),
        _gate("replay_boundary", "pass", "Stage 2.6 replay boundary exists."),
        _gate(
            "execution_authorization",
            "blocked",
            "This contract does not authorize training execution.",
        ),
    )


def _scalarization_policy() -> dict[str, object]:
    return {
        "policy_id": "constraint_first_component_policy_v1",
        "status": "design_only_no_weights_selected",
        "tau_policy": {
            "tau_requirement_min": 0.9,
            "final_tau_selected": False,
            "lower_tau_values": "diagnostic_only",
        },
        "components": [
            "reliability_violation_penalty",
            "normalized_latency_penalty",
            "normalized_energy_penalty",
        ],
        "plateau_rule": "no_reliability_bonus_above_tau",
        "weight_policy": "future_owner_approved_config_required",
        "forbidden_terms": [
            "standalone_timeout_term",
            "standalone_quorum_term",
            "standalone_density_term",
            "full_graph_bonus",
            "oracle_label_target",
            "legacy_formula_copy",
        ],
    }


def _learning_target_contract() -> dict[str, object]:
    return {
        "status": "planned_not_active",
        "currently_admitted_training_columns": [
            "reward_surrogate",
            "reward_reliability_penalty",
            "reward_latency_penalty",
            "reward_energy_penalty",
            "reward_config_id",
        ],
        "future_columns_require_contract": [
            "return",
            "advantage",
            "value_target",
            "episode_id",
            "trajectory_id",
            "discount_factor",
        ],
        "deployment_actor_input": "unchanged_actor_observation_only",
        "no_dataset_writer_added": True,
    }


def _information_boundary() -> dict[str, object]:
    return {
        "deployment_actor_allowed": [
            "agent_id",
            "agent_kind",
            "time_step",
            "local_position_m",
            "local_neighbor_observations",
            "local_messages",
            "local_history",
        ],
        "deployment_actor_forbidden": [
            "global_topology",
            "oracle_labels",
            "registered_evaluation_metrics",
            "surrogate_diagnostics",
            "future_outcomes",
            "centralized_training_view",
        ],
        "centralized_training_view": "future_training_only_interface",
        "architecture_status": "not_selected_not_implemented",
    }


def _scenario_protocol() -> dict[str, object]:
    return {
        "source_evidence": [
            "stage_5_0f_alpha_fixture_suite",
            "stage_5_0l_stage3_backed_range_review",
            "stage_5_4_component_report",
        ],
        "split_policy": "scenario_id_and_seed_split_required_before_execution",
        "minimum_seed_plan": "at_least_five_seeds_in_future_run_contract",
        "deployment_distribution_status": "not_calibrated",
    }


def _baseline_protocol() -> dict[str, object]:
    return {
        "required_baselines": [
            "empty_topology",
            "sparse_non_learning_baseline",
            "dense_full_graph_baseline",
            "random_local_baseline",
            "oracle_candidate_diagnostic",
        ],
        "full_graph_status": "baseline_not_oracle",
        "oracle_candidate_status": "review_only_not_actor_input",
    }


def _diagnostics() -> dict[str, object]:
    return {
        "registered_evaluation": [
            "consensus_success_probability",
            "latency",
            "energy",
            "topology_diagnostics",
        ],
        "training_only_diagnostics": [
            "surrogate_components",
            "constraint_violation_rate",
            "policy_entropy",
            "action_sparsity",
            "seed_variance",
        ],
        "success_evidence_rule": (
            "registered evaluation metrics and baselines remain primary evidence"
        ),
    }


def _stop_conditions() -> dict[str, object]:
    return {
        "status": "design_only",
        "future_required": [
            "max_update_budget",
            "validation_plateau_rule",
            "constraint_failure_stop_rule",
            "metric_regression_stop_rule",
            "artifact_budget_rule",
        ],
        "single_seed_success_is_insufficient": True,
    }


def _artifact_policy() -> dict[str, object]:
    return {
        "status": "design_only_no_artifacts_written",
        "future_manifest_required": True,
        "future_artifact_root": "result_save",
        "future_required_fields": [
            "stage_id",
            "config_id",
            "scenario_set_id",
            "seed",
            "code_version_marker",
            "contract_ids",
            "metric_registry_version",
        ],
        "checkpoint_creation_allowed": False,
    }


def _blocked_until_future_approval() -> tuple[str, ...]:
    return (
        "training_execution",
        "policy_network_implementation",
        "critic_network_implementation",
        "checkpoint_creation",
        "weight_calibration",
        "final_tau_selection",
        "legacy_code_migration",
    )


def _checks() -> dict[str, object]:
    return {
        "training_execution_allowed": False,
        "training_run": False,
        "model_code_added": False,
        "checkpoint_written": False,
        "weight_calibration_performed": False,
        "final_tau_selected": False,
        "v5_code_migrated": False,
        "full_graph_is_oracle": False,
        "oracle_candidate_actor_input": False,
        "deployment_actor_boundary_unchanged": True,
    }


def _gate(gate_id: str, status: str, evidence: str) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "status": status,
        "evidence": evidence,
    }
