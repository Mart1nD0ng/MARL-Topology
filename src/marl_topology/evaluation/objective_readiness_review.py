"""Stage 5.0m objective readiness review before reward implementation."""

from __future__ import annotations

from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics

from .feasibility_envelope_sweep import build_stage5_0j_minimal_feasibility_envelope_sweep
from .feasibility_envelope_sweep_design import REGISTERED_SWEEP_METRIC_FIELDS
from .feasibility_envelope_sweep_range_review import (
    build_stage5_0l_stage3_backed_sweep_range_review,
)
from .feasibility_envelope_sweep_stage3_backed import (
    build_stage5_0k_stage3_backed_feasibility_envelope_sweep,
)
from .requirement_feasibility_diagnosis import (
    TAU_REQUIREMENT_MIN,
    build_stage5_0h_requirement_feasibility_diagnosis,
)


STAGE5_0M_OBJECTIVE_READINESS_STAGE_ID = (
    "stage_5_0m_objective_readiness_review_before_reward_implementation"
)
STAGE5_0M_RECOMMENDED_NEXT_TASK = "stage_5_1_reward_implementation_plan_without_code"


def build_stage5_0m_objective_readiness_review() -> dict[str, object]:
    """Build a go/no-go review for entering Stage 5.1 plan-only work.

    The review composes existing Stage 5.0 evidence. It does not select a final
    tau, implement reward, train models, add model code, or migrate v5 code.
    """

    require_registered_metrics(REGISTERED_SWEEP_METRIC_FIELDS)
    diagnosis = build_stage5_0h_requirement_feasibility_diagnosis()
    alpha_sweep = build_stage5_0j_minimal_feasibility_envelope_sweep()
    stage3_sweep = build_stage5_0k_stage3_backed_feasibility_envelope_sweep()
    range_review = build_stage5_0l_stage3_backed_sweep_range_review()

    evidence_summary = _evidence_summary(diagnosis, alpha_sweep, stage3_sweep, range_review)
    readiness_gates = _readiness_gates(diagnosis, alpha_sweep, stage3_sweep, range_review)
    checks = _checks(readiness_gates)
    return {
        "stage": STAGE5_0M_OBJECTIVE_READINESS_STAGE_ID,
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "review_scope": "objective_readiness_for_stage5_1_plan_only",
        "source_stages": [
            diagnosis["stage"],
            alpha_sweep["stage"],
            stage3_sweep["stage"],
            range_review["stage"],
        ],
        "evidence_summary": evidence_summary,
        "readiness_gates": readiness_gates,
        "decision": {
            "stage5_1_plan_only_allowed": checks["stage5_1_plan_only_allowed"],
            "stage5_1_scope": "reward implementation plan without reward code",
            "owner_approval_required_before_stage5_1": True,
            "reward_code_allowed": False,
            "reward_weight_calibration_allowed": False,
            "training_allowed": False,
            "actor_critic_model_work_allowed": False,
            "final_tau_selected": False,
            "recommended_next_task": STAGE5_0M_RECOMMENDED_NEXT_TASK,
        },
        "stage5_1_entry_conditions": _stage5_1_entry_conditions(),
        "blocked_after_stage5_0m": _blocked_after_stage5_0m(),
        "metric_governance": {
            "metric_valued_fields": list(REGISTERED_SWEEP_METRIC_FIELDS),
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "new_metric_names_introduced": [],
        },
        "checks": checks,
    }


def _evidence_summary(
    diagnosis: dict[str, object],
    alpha_sweep: dict[str, object],
    stage3_sweep: dict[str, object],
    range_review: dict[str, object],
) -> dict[str, object]:
    diagnosis_checks = diagnosis["checks"]
    alpha_checks = alpha_sweep["checks"]
    stage3_checks = stage3_sweep["checks"]
    range_checks = range_review["checks"]
    stage3_sweeps = tuple(stage3_sweep["executed_sweep_ids"]) + tuple(
        range_review["executed_sweep_ids"]
    )
    return {
        "requirement_baseline": {
            "tau_requirement_min": TAU_REQUIREMENT_MIN,
            "tau_is_requirement_baseline": diagnosis_checks["tau_requirement_min_recorded"],
            "final_tau_selected": False,
            "lower_tau_diagnostic_only": True,
        },
        "stage5_0h_diagnosis": {
            "feasible_family_count": diagnosis_checks["feasible_family_count"],
            "infeasible_family_count": diagnosis_checks["infeasible_family_count"],
            "all_infeasible_rows_have_failure_reason": diagnosis_checks[
                "all_infeasible_rows_have_failure_reason"
            ],
            "family_dominant_failure_reasons_present": diagnosis_checks[
                "family_dominant_failure_reasons_present"
            ],
        },
        "stage5_0j_alpha_sweep": {
            "executed_sweep_ids": list(alpha_sweep["executed_sweep_ids"]),
            "at_least_one_infeasible_to_feasible_transition": alpha_checks[
                "at_least_one_infeasible_to_feasible_transition"
            ],
            "conservative_fault_filter_does_not_increase_reliability": alpha_checks[
                "conservative_fault_filter_does_not_increase_reliability"
            ],
        },
        "stage5_0k_stage3_backed_sweep": {
            "executed_sweep_ids": list(stage3_sweep["executed_sweep_ids"]),
            "all_rows_stage3_backed": stage3_checks["all_rows_stage3_backed"],
            "all_rows_use_finite_blocklength": stage3_checks[
                "all_rows_use_finite_blocklength"
            ],
            "at_least_one_stage3_backed_infeasible_to_feasible_transition": stage3_checks[
                "at_least_one_stage3_backed_infeasible_to_feasible_transition"
            ],
        },
        "stage5_0l_range_review": {
            "executed_sweep_ids": list(range_review["executed_sweep_ids"]),
            "all_rows_stage3_backed": range_checks["all_rows_stage3_backed"],
            "all_rows_use_finite_blocklength": range_checks["all_rows_use_finite_blocklength"],
            "every_sweep_has_infeasible_to_feasible_transition": range_checks[
                "every_sweep_has_infeasible_to_feasible_transition"
            ],
            "realism_statuses": sorted(
                {item["realism_status"] for item in range_review["realism_review"]}
            ),
        },
        "stage3_backed_sweep_coverage": sorted(set(stage3_sweeps)),
    }


def _readiness_gates(
    diagnosis: dict[str, object],
    alpha_sweep: dict[str, object],
    stage3_sweep: dict[str, object],
    range_review: dict[str, object],
) -> list[dict[str, object]]:
    diagnosis_checks = diagnosis["checks"]
    alpha_checks = alpha_sweep["checks"]
    stage3_checks = stage3_sweep["checks"]
    range_checks = range_review["checks"]
    all_checks = (diagnosis_checks, alpha_checks, stage3_checks, range_checks)
    return [
        _gate(
            "tau_requirement_baseline_gate",
            True,
            "tau_requirement_min = 0.9 is recorded as a requirement baseline, not a fitted low threshold",
        ),
        _gate(
            "objective_contract_semantics_gate",
            True,
            "reliability remains a constraint and latency/energy remain primary objectives",
        ),
        _gate(
            "metric_governance_gate",
            all(
                check.get("registered_metric_fields_only", True)
                and check.get("final_tau_selected", False) is False
                for check in all_checks
            ),
            "reports use registered metric-valued fields only and do not select final tau",
        ),
        _gate(
            "failure_diagnosis_gate",
            diagnosis_checks["all_infeasible_rows_have_failure_reason"]
            and diagnosis_checks["family_dominant_failure_reasons_present"],
            "infeasible rows and families have explicit failure-reason diagnostics",
        ),
        _gate(
            "stage3_backed_evidence_gate",
            stage3_checks["all_rows_stage3_backed"]
            and range_checks["all_rows_stage3_backed"]
            and stage3_checks["all_rows_use_finite_blocklength"]
            and range_checks["all_rows_use_finite_blocklength"],
            "Stage 3-backed rows use finite-blocklength communication records",
        ),
        _gate(
            "feasibility_lever_evidence_gate",
            stage3_checks["at_least_one_stage3_backed_infeasible_to_feasible_transition"]
            and range_checks["every_sweep_has_infeasible_to_feasible_transition"],
            "single-axis Stage 3-backed controls can move selected rows to requirement feasibility",
        ),
        _gate(
            "negative_control_gate",
            all(
                check["reward_implemented"] is False
                and check["training_run"] is False
                and check["v5_code_migrated"] is False
                and check["full_graph_not_oracle"] is True
                and check["oracle_labels_not_actor_inputs"] is True
                for check in all_checks
            ),
            "reward, training, v5 migration, full-graph oracle, and actor-leakage routes remain closed",
        ),
        _gate(
            "stage5_1_scope_gate",
            True,
            "Stage 5.1 may be a plan-only reward implementation design task; reward code remains blocked",
        ),
        _gate(
            "conservative_fault_filter_awareness_gate",
            alpha_checks["conservative_fault_filter_does_not_increase_reliability"],
            "remove_largest remains a conservative diagnostic, not a strict Byzantine model",
        ),
    ]


def _checks(readiness_gates: list[dict[str, object]]) -> dict[str, object]:
    gates_passed = all(gate["passed"] is True for gate in readiness_gates)
    return {
        "tau_requirement_min_fixed": True,
        "final_tau_selected": False,
        "final_tau_below_requirement_selected": False,
        "stage5_1_plan_only_allowed": gates_passed,
        "additional_stage5_0_tasks_required_before_stage5_1_plan": not gates_passed,
        "reward_code_allowed": False,
        "reward_weight_calibration_allowed": False,
        "training_allowed": False,
        "actor_critic_model_work_allowed": False,
        "full_graph_not_oracle": True,
        "oracle_labels_not_actor_inputs": True,
        "registered_metric_fields_only": True,
        "new_metric_names_introduced": False,
        "reward_implemented": False,
        "training_run": False,
        "v5_code_migrated": False,
        "all_readiness_gates_passed": gates_passed,
    }


def _stage5_1_entry_conditions() -> list[dict[str, object]]:
    return [
        _condition(
            "allowed_scope",
            "Stage 5.1 may design a reward implementation plan without writing reward code",
            True,
        ),
        _condition(
            "tau_policy",
            "Use tau_requirement_min = 0.9 as the requirement baseline; do not select a lower final tau",
            True,
        ),
        _condition(
            "surrogate_shape",
            "Plan must preserve reliability plateau above tau and latency/energy penalties below objective governance",
            True,
        ),
        _condition(
            "normalization",
            "Plan must specify latency and energy normalization references before any code",
            True,
        ),
        _condition(
            "negative_tests",
            "Plan must include reward-hacking, metric-governance, and Dec-POMDP leakage tests",
            True,
        ),
        _condition(
            "blocked_code",
            "Actual reward module, reward weights, training, and model work remain blocked",
            True,
        ),
    ]


def _blocked_after_stage5_0m() -> list[str]:
    return [
        "reward implementation code",
        "reward weight calibration",
        "training runs",
        "actor critic and advanced model implementation",
        "final tau_consensus selection below tau_requirement_min",
        "v5 code migration",
        "oracle labels as actor inputs",
    ]


def _gate(gate_id: str, passed: bool, evidence: str) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "passed": passed,
        "evidence": evidence,
    }


def _condition(condition_id: str, requirement: str, satisfied_for_plan: bool) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "requirement": requirement,
        "satisfied_for_stage5_1_plan": satisfied_for_plan,
    }
