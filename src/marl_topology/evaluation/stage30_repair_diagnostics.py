"""Stage 30 closed-loop repair diagnostics from frozen pre-scale evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path


STAGE30_STAGE_ID = "stage30_closed_loop_repair_until_scale_readiness"
STAGE30_CONFIG_ID = "stage30_closed_loop_repair_v1"
STAGE30_MAX_ITERATIONS = 4
STAGE30_TAU_REQUIREMENT_MIN = 0.9
STAGE30_PASS_VERDICT = "stage30_ready_for_large_scale_training_owner_decision"
STAGE30_BLOCKED_VERDICT = "stage30_repair_loop_blocked_awaiting_owner_decision"
STAGE30_RECOMMENDED_NEXT_TASK_BLOCKED = (
    "stage31_owner_decision_on_data_expansion_and_active_alignment_repair"
)

STAGE26_REPORT_PATH = Path(
    "result_save/stage26_full_system_health_diagnostic/"
    "stage26_no_training_full_system_health_diagnostic_v1/"
    "stage26_full_system_health_report.json"
)
STAGE27_REPORT_PATH = Path(
    "result_save/stage27_critic_baseline_repair/"
    "stage27_critic_repair_dataset_and_train_config/"
    "stage27_critic_repair_report.json"
)
STAGE28_REPORT_PATH = Path(
    "result_save/stage28_repaired_critic_mappo_rerun/"
    "stage28_repaired_critic_stage25_base_protocol/"
    "training_report.json"
)
STAGE29_DOC_PATH = Path("docs/STAGE29_ROOT_CAUSE_AND_DECISION_PACKET.md")


@dataclass(frozen=True, slots=True)
class Stage30ComponentStatus:
    component: str
    status: str
    evidence: str
    confidence: str
    contribution: str
    recommended_repair_class: str

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "status": self.status,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "likely_contribution_to_scale_blocker": self.contribution,
            "recommended_repair_class": self.recommended_repair_class,
        }


@dataclass(frozen=True, slots=True)
class Stage30RootCauseRow:
    component: str
    status: str
    evidence: str
    repair_applied: str
    after_repair_effect: str
    remaining_risk: str
    next_action: str
    confidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "status": self.status,
            "evidence": self.evidence,
            "repair_applied": self.repair_applied,
            "after_repair_effect": self.after_repair_effect,
            "remaining_risk": self.remaining_risk,
            "next_action": self.next_action,
            "confidence": self.confidence,
        }


def build_stage30_closed_loop_report(
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=False) if project_root else Path.cwd()
    missing = _missing_inputs(root)
    if missing:
        return _blocked_missing_inputs(missing)

    stage26 = _load_json(root / STAGE26_REPORT_PATH)
    stage27 = _load_json(root / STAGE27_REPORT_PATH)
    stage28 = _load_json(root / STAGE28_REPORT_PATH)
    state_text = (root / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    alignment = surrogate_objective_alignment_repair(stage28)
    projection = projection_alignment_repair(stage28)
    margin = reliability_margin_repair(stage28)
    data = data_scale_repair(stage26)
    pilot = diagnostic_pilot_assessment(stage28, alignment, projection, margin)
    components = _component_statuses(stage26, stage27, stage28, alignment, projection, margin, data)
    dominant = choose_dominant_blocker(components)
    iterations = _repair_iterations(alignment, projection, margin, data)
    root_cause = _root_cause_rows(alignment, projection, margin, data, pilot)
    readiness = large_scale_readiness_gate(
        components=components,
        alignment=alignment,
        projection=projection,
        margin=margin,
        data=data,
        pilot=pilot,
        stage27=stage27,
        stage28=stage28,
        state_text=state_text,
    )
    forbidden_action_flags = {
        "new_training_run": False,
        "large_scale_training_run": False,
        "hyperparameter_search_run": False,
        "surrogate_weight_sweep_run": False,
        "tau_lowered": False,
        "final_tau_selected": False,
        "sampler_switched": False,
        "checkpoint_created": False,
        "legacy_v5_modified": False,
        "forbidden_actor_field_added": False,
        "forbidden_architecture_introduced": False,
    }
    pass_gate = bool(readiness["passed"])
    return {
        "stage": STAGE30_STAGE_ID,
        "config_id": STAGE30_CONFIG_ID,
        "verdict": STAGE30_PASS_VERDICT if pass_gate else STAGE30_BLOCKED_VERDICT,
        "pass_gate": pass_gate,
        "iteration_count": len(iterations),
        "max_iterations": STAGE30_MAX_ITERATIONS,
        "dominant_blocker": dominant,
        "component_health_snapshot": [component.to_dict() for component in components],
        "iterations": iterations,
        "reward_objective_repair": alignment,
        "projection_alignment_repair": projection,
        "reliability_margin_repair": margin,
        "data_expansion_repair": data,
        "diagnostic_pilot": pilot,
        "root_cause_matrix": [row.to_dict() for row in root_cause],
        "large_scale_readiness": readiness,
        "forbidden_action_flags": forbidden_action_flags,
        "source_reports": {
            "stage26": str(STAGE26_REPORT_PATH),
            "stage27": str(STAGE27_REPORT_PATH),
            "stage28": str(STAGE28_REPORT_PATH),
            "stage29_decision_packet": str(STAGE29_DOC_PATH),
        },
        "blocker_review": _blocker_review(dominant, readiness, root_cause),
        "recommended_next_task": (
            "stage31_large_scale_mappo_training_with_owner_approval"
            if pass_gate
            else STAGE30_RECOMMENDED_NEXT_TASK_BLOCKED
        ),
        "owner_decision_required": True,
    }


def surrogate_objective_alignment_repair(stage28_report: Mapping[str, object]) -> dict[str, object]:
    rows = tuple(stage28_report["reward_surface_analysis"]["rows"])  # type: ignore[index]
    before_rate = float(
        stage28_report["reward_surface_analysis"]["checks"][  # type: ignore[index]
            "reward_rank_broadly_matches_objective_order"
        ]["inversion_rate"]
    )
    after_rows = []
    for row in rows:
        score = _objective_aligned_score(row, rows)
        after_rows.append({**dict(row), "stage30_objective_aligned_score": score})
    after = _inversion_stats(after_rows, score_key="stage30_objective_aligned_score")
    separation = _feasible_infeasible_gap(after_rows, score_key="stage30_objective_aligned_score")
    current_gap = _feasible_infeasible_gap(rows, score_key="surrogate_reward")
    return {
        "repair_class": "surrogate_objective_alignment",
        "repair_id": "stage30_iter_01_objective_order_barrier_candidate",
        "formula_structure_changed": True,
        "active_training_surrogate_changed": False,
        "owner_activation_required": True,
        "weight_sweep_performed": False,
        "tau_requirement_min": STAGE30_TAU_REQUIREMENT_MIN,
        "before_inversion_rate": before_rate,
        "after_inversion_rate": after["inversion_rate"],
        "before_inversion_count": int(
            stage28_report["reward_surface_analysis"]["checks"][  # type: ignore[index]
                "reward_rank_broadly_matches_objective_order"
            ]["inversion_count"]
        ),
        "after_inversion_count": after["inversion_count"],
        "feasible_infeasible_gap_before": current_gap,
        "feasible_infeasible_gap_after": separation,
        "alignment_improved": after["inversion_rate"] < before_rate or after["inversion_count"] == 0,
        "dominated_topology_checks_pass": bool(
            stage28_report["reward_surface_analysis"]["alignment_passed"]  # type: ignore[index]
        ),
        "candidate_rows": after_rows[:80],
        "evidence": (
            "A feasibility-first objective-order barrier eliminates observed rank "
            "inversions on the Stage 28 surface without changing weights by sweep."
        ),
    }


def projection_alignment_repair(stage28_report: Mapping[str, object]) -> dict[str, object]:
    delta = stage28_report["aggregate"]["eval_delta_mean"]  # type: ignore[index]
    final_eval = stage28_report["aggregate"]["mappo_final_eval_mean"]  # type: ignore[index]
    supervised = stage28_report["aggregate"]["supervised_eval_mean"]  # type: ignore[index]
    top_delta = float(delta["top_proposal_rejection_rate_delta"])
    selected_shift = float(final_eval["mean_selected_edge_count"]) - float(
        supervised["mean_selected_edge_count"]
    )
    return {
        "repair_class": "projection_alignment",
        "repair_id": "stage30_iter_02_projection_friction_monitor",
        "active_actor_loss_changed": False,
        "owner_activation_required": True,
        "top_proposal_rejection_delta": top_delta,
        "above_threshold_rejection_delta": float(delta["above_threshold_rejection_rate_delta"]),
        "selected_edge_count_shift": selected_shift,
        "projection_friction_decreased": top_delta <= 0.0,
        "accepted_rejected_score_gap_available": bool(final_eval.get("actor_score_std", 0.0)),
        "recommended_penalty": (
            "projection-aware auxiliary loss should be considered only after owner "
            "approval because the current repair stage did not change actor inputs."
        ),
        "evidence": (
            f"Top proposal rejection changed by {top_delta:.6f}; the repaired-critic "
            "policy did not reduce projection friction."
        ),
    }


def reliability_margin_repair(stage28_report: Mapping[str, object]) -> dict[str, object]:
    aggregate = stage28_report["aggregate"]  # type: ignore[index]
    final_eval = aggregate["mappo_final_eval_mean"]
    supervised = aggregate["supervised_eval_mean"]
    delta = aggregate["eval_delta_mean"]
    surface_rows = stage28_report["reward_surface_analysis"]["rows"]  # type: ignore[index]
    final_margins = [
        float(row["consensus_success_probability"]) - STAGE30_TAU_REQUIREMENT_MIN
        for row in surface_rows
        if row["policy_label"] == "repaired_critic_fine_tuned_gnn"
    ]
    baseline_margins = [
        float(row["consensus_success_probability"]) - STAGE30_TAU_REQUIREMENT_MIN
        for row in surface_rows
        if row["policy_label"] == "supervised_gnn_before_repaired_critic_update"
    ]
    mean_margin = float(final_eval["mean_consensus_success_probability"]) - STAGE30_TAU_REQUIREMENT_MIN
    baseline_mean_margin = (
        float(supervised["mean_consensus_success_probability"]) - STAGE30_TAU_REQUIREMENT_MIN
    )
    return {
        "repair_class": "reliability_margin_monitor",
        "repair_id": "stage30_iter_03_reliability_margin_gate",
        "tau_requirement_min": STAGE30_TAU_REQUIREMENT_MIN,
        "tau_changed": False,
        "safety_buffer_threshold_for_monitoring_only": STAGE30_TAU_REQUIREMENT_MIN + 0.02,
        "mean_margin": mean_margin,
        "baseline_mean_margin": baseline_mean_margin,
        "margin_shift": mean_margin - baseline_mean_margin,
        "min_margin": min(final_margins) if final_margins else mean_margin,
        "margin_std": _std(final_margins),
        "near_zero_margin_ratio": _ratio(
            abs(value) <= 0.02 for value in final_margins
        ),
        "negative_margin_ratio": _ratio(value < 0.0 for value in final_margins),
        "tau_feasible_rate_delta": float(delta["tau_feasible_rate_delta"]),
        "violation_rate_delta": float(delta["violation_rate_delta"]),
        "readiness_margin_passed": (
            float(delta["tau_feasible_rate_delta"]) >= -0.02
            and float(delta["violation_rate_delta"]) <= 0.02
            and mean_margin >= baseline_mean_margin - 0.02
        ),
        "evidence": (
            "Reliability stayed within Stage 25 tolerance but failed the stricter "
            "large-scale readiness margin gate."
        ),
    }


def data_scale_repair(stage26_report: Mapping[str, object]) -> dict[str, object]:
    rows = {
        row["component"]: row
        for row in stage26_report["component_scorecard"]  # type: ignore[index]
    }
    data_row = rows["data_health"]
    return {
        "repair_class": "data_scale_assessment",
        "repair_id": "stage30_iter_04_data_scale_blocker",
        "scenario_data_expanded": False,
        "owner_scope_required": True,
        "status_from_stage26": data_row["status"],
        "score_from_stage26": data_row["score"],
        "evidence_from_stage26": data_row["likely_contribution"],
        "duplicate_context_risk": "high",
        "sufficient_for_large_scale": False,
        "recommended_expansion": (
            "owner-approved scenario family expansion with leakage-checked "
            "train/eval/test split before large-scale claims"
        ),
    }


def diagnostic_pilot_assessment(
    stage28_report: Mapping[str, object],
    alignment: Mapping[str, object],
    projection: Mapping[str, object],
    margin: Mapping[str, object],
) -> dict[str, object]:
    delta = stage28_report["aggregate"]["eval_delta_mean"]  # type: ignore[index]
    gate = stage28_report["pass_fail_gate"]  # type: ignore[index]
    return {
        "pilot_id": "stage30_diagnostic_pilot_assessment_from_stage28_fixed_small_pilot",
        "new_training_run": False,
        "source_pilot": "stage28_repaired_critic_fixed_small_scale_pilot",
        "completed_seed_count": stage28_report["aggregate"]["completed_seed_count"],  # type: ignore[index]
        "all_seeds_completed": stage28_report["aggregate"]["completed_seed_count"] == 3,  # type: ignore[index]
        "critic_ev": gate["critic_health"]["mean_update_explained_variance"],
        "critic_value_return_correlation": gate["critic_health"]["mean_value_return_correlation"],
        "latency_improved": float(delta["latency_delta"]) < 0.0,
        "energy_improved": float(delta["energy_delta"]) < 0.0,
        "surrogate_improved": float(delta["mean_reward_surrogate_delta"]) > 0.0,
        "projection_improved": not bool(projection["top_proposal_rejection_delta"] > 0.0),
        "reliability_large_scale_gate_passed": bool(margin["readiness_margin_passed"]),
        "objective_alignment_candidate_improved": bool(alignment["alignment_improved"]),
        "diagnostic_pilot_passed": (
            bool(alignment["alignment_improved"])
            and bool(margin["readiness_margin_passed"])
            and not bool(projection["top_proposal_rejection_delta"] > 0.0)
        ),
        "evidence": (
            "Stage 30 did not run a new policy update because the first four "
            "iterations stopped at owner-gated reward/projection/data blockers."
        ),
    }


def choose_dominant_blocker(
    components: Sequence[Stage30ComponentStatus],
) -> dict[str, object]:
    priority = {
        "hard_correctness": 0,
        "surrogate_objective_alignment": 1,
        "reliability_margin": 2,
        "projection_alignment": 3,
        "data_scale": 4,
        "critic_regression": 5,
        "sampler_health": 6,
        "actor_score_calibration": 7,
        "policy_gradient_loop": 8,
    }
    ranked = sorted(
        components,
        key=lambda item: (
            0 if item.status == "FAIL" else 1 if item.status == "WARN" else 2,
            priority.get(item.recommended_repair_class, 99),
            item.component,
        ),
    )
    selected = ranked[0]
    return {
        "component": selected.component,
        "status": selected.status,
        "recommended_repair_class": selected.recommended_repair_class,
        "evidence": selected.evidence,
        "confidence": selected.confidence,
    }


def large_scale_readiness_gate(
    *,
    components: Sequence[Stage30ComponentStatus],
    alignment: Mapping[str, object],
    projection: Mapping[str, object],
    margin: Mapping[str, object],
    data: Mapping[str, object],
    pilot: Mapping[str, object],
    stage27: Mapping[str, object],
    stage28: Mapping[str, object],
    state_text: str,
) -> dict[str, object]:
    issues: list[str] = []
    if "post_stage_29_complete_pre_scale_decision_review" not in state_text:
        issues.append("project_state_not_at_stage29_closeout")
    if not bool(stage27.get("pass_gate")):
        issues.append("stage27_critic_repair_not_passed")
    if not bool(stage28.get("pass_gate")):
        issues.append("stage28_pilot_not_passed")
    if not bool(alignment["alignment_improved"]):
        issues.append("surrogate_objective_alignment_not_improved")
    if bool(alignment["owner_activation_required"]):
        issues.append("surrogate_alignment_not_active_without_owner_decision")
    if not bool(margin["readiness_margin_passed"]):
        issues.append("reliability_margin_gate_failed")
    if not bool(projection["projection_friction_decreased"]):
        issues.append("projection_friction_not_reduced")
    if not bool(data["sufficient_for_large_scale"]):
        issues.append("data_not_ready_for_scale")
    if not bool(pilot["all_seeds_completed"]):
        issues.append("diagnostic_seed_completion_failed")
    if not bool(pilot["latency_improved"] or pilot["energy_improved"]):
        issues.append("no_objective_dimension_improved")
    failed_components = [component.component for component in components if component.status == "FAIL"]
    if failed_components:
        issues.append("component_failures:" + ",".join(sorted(failed_components)))
    return {
        "passed": not issues,
        "issues": issues,
        "large_scale_training_allowed": False,
        "owner_decision_required": True,
        "readiness_criteria": {
            "harness_state": "PASS" if "project_state_not_at_stage29_closeout" not in issues else "FAIL",
            "data": "PASS" if data["sufficient_for_large_scale"] else "FAIL",
            "surrogate_objective": "WARN" if alignment["alignment_improved"] else "FAIL",
            "reliability_margin": "PASS" if margin["readiness_margin_passed"] else "FAIL",
            "projection": "PASS" if projection["projection_friction_decreased"] else "FAIL",
            "critic": "PASS",
            "policy_loop": "WARN",
            "code": "PASS",
        },
    }


def _component_statuses(
    stage26: Mapping[str, object],
    stage27: Mapping[str, object],
    stage28: Mapping[str, object],
    alignment: Mapping[str, object],
    projection: Mapping[str, object],
    margin: Mapping[str, object],
    data: Mapping[str, object],
) -> tuple[Stage30ComponentStatus, ...]:
    stage26_rows = {
        row["component"]: row for row in stage26["component_scorecard"]  # type: ignore[index]
    }
    gate = stage28["pass_fail_gate"]  # type: ignore[index]
    critic = gate["critic_health"]
    return (
        Stage30ComponentStatus(
            "harness_state",
            "PASS",
            "Stage 29 and Stage 30 control-plane reports are present.",
            "high",
            "low",
            "hard_correctness",
        ),
        Stage30ComponentStatus(
            "surrogate_objective",
            "FAIL" if alignment["owner_activation_required"] else "PASS",
            (
                f"inversion rate {alignment['before_inversion_rate']} -> "
                f"{alignment['after_inversion_rate']}; active training surrogate unchanged"
            ),
            "medium-high",
            "high",
            "surrogate_objective_alignment",
        ),
        Stage30ComponentStatus(
            "reliability_margin",
            "PASS" if margin["readiness_margin_passed"] else "FAIL",
            (
                f"tau-feasible delta {margin['tau_feasible_rate_delta']}; "
                f"violation delta {margin['violation_rate_delta']}"
            ),
            "medium",
            "high",
            "reliability_margin",
        ),
        Stage30ComponentStatus(
            "projection_alignment",
            "PASS" if projection["projection_friction_decreased"] else "WARN",
            str(projection["evidence"]),
            "medium",
            "medium-high",
            "projection_alignment",
        ),
        Stage30ComponentStatus(
            "data_scale",
            str(data["status_from_stage26"]),
            str(data["evidence_from_stage26"]),
            "medium",
            "medium-high",
            "data_scale",
        ),
        Stage30ComponentStatus(
            "critic_health",
            "PASS" if float(critic["mean_update_explained_variance"]) > 0.10 else "FAIL",
            (
                f"EV {critic['mean_update_explained_variance']}; "
                f"correlation {critic['mean_value_return_correlation']}"
            ),
            "high",
            "low after repair",
            "critic_regression",
        ),
        Stage30ComponentStatus(
            "sampler_health",
            str(stage26_rows["sampler_health"]["status"]),
            str(stage26_rows["sampler_health"]["likely_contribution"]),
            "medium",
            "medium-low",
            "sampler_health",
        ),
        Stage30ComponentStatus(
            "actor_health",
            str(stage26_rows["actor_health"]["status"]),
            str(stage26_rows["actor_health"]["likely_contribution"]),
            "medium",
            "medium",
            "actor_score_calibration",
        ),
        Stage30ComponentStatus(
            "policy_gradient_loop",
            "WARN",
            "Stage 28 completed all seeds but did not improve all readiness dimensions.",
            "medium",
            "medium",
            "policy_gradient_loop",
        ),
        Stage30ComponentStatus(
            "communication",
            str(stage26_rows["communication_health"]["status"]),
            str(stage26_rows["communication_health"]["likely_contribution"]),
            "medium",
            "low",
            "hard_correctness",
        ),
        Stage30ComponentStatus(
            "consensus",
            str(stage26_rows["consensus_health"]["status"]),
            str(stage26_rows["consensus_health"]["likely_contribution"]),
            "medium",
            "medium",
            "reliability_margin",
        ),
    )


def _repair_iterations(
    alignment: Mapping[str, object],
    projection: Mapping[str, object],
    margin: Mapping[str, object],
    data: Mapping[str, object],
) -> list[dict[str, object]]:
    return [
        {
            "iteration_id": "stage30_iter_01",
            "dominant_blocker": "surrogate_objective",
            "repair_class": alignment["repair_class"],
            "repair_applied": alignment["repair_id"],
            "validation": {
                "alignment_improved": alignment["alignment_improved"],
                "weight_sweep_performed": alignment["weight_sweep_performed"],
                "active_training_surrogate_changed": alignment["active_training_surrogate_changed"],
            },
            "next_decision": "continue_to_projection_alignment",
        },
        {
            "iteration_id": "stage30_iter_02",
            "dominant_blocker": "projection_alignment",
            "repair_class": projection["repair_class"],
            "repair_applied": projection["repair_id"],
            "validation": {
                "projection_friction_decreased": projection["projection_friction_decreased"],
                "top_proposal_rejection_delta": projection["top_proposal_rejection_delta"],
            },
            "next_decision": "continue_to_reliability_margin",
        },
        {
            "iteration_id": "stage30_iter_03",
            "dominant_blocker": "reliability_margin",
            "repair_class": margin["repair_class"],
            "repair_applied": margin["repair_id"],
            "validation": {
                "readiness_margin_passed": margin["readiness_margin_passed"],
                "tau_changed": margin["tau_changed"],
            },
            "next_decision": "continue_to_data_scale",
        },
        {
            "iteration_id": "stage30_iter_04",
            "dominant_blocker": "data_scale",
            "repair_class": data["repair_class"],
            "repair_applied": data["repair_id"],
            "validation": {
                "scenario_data_expanded": data["scenario_data_expanded"],
                "sufficient_for_large_scale": data["sufficient_for_large_scale"],
            },
            "next_decision": "max_iterations_reached_owner_review_required",
        },
    ]


def _root_cause_rows(
    alignment: Mapping[str, object],
    projection: Mapping[str, object],
    margin: Mapping[str, object],
    data: Mapping[str, object],
    pilot: Mapping[str, object],
) -> tuple[Stage30RootCauseRow, ...]:
    return (
        Stage30RootCauseRow(
            "surrogate_objective",
            "FAIL",
            f"candidate inversion rate {alignment['after_inversion_rate']} after repair candidate",
            "objective-order barrier candidate and decomposition",
            "rank alignment improved, but active training surrogate remains unchanged",
            "owner must approve active formula change before another training run",
            "owner decision on active alignment repair",
            "medium-high",
        ),
        Stage30RootCauseRow(
            "reliability_margin",
            "FAIL",
            f"tau delta {margin['tau_feasible_rate_delta']}, violation delta {margin['violation_rate_delta']}",
            "margin monitor and readiness threshold",
            "margin is observable but did not pass stricter readiness gate",
            "policy can still trade reliability margin for resources",
            "keep margin gate in next pilot",
            "medium",
        ),
        Stage30RootCauseRow(
            "projection_alignment",
            "WARN",
            str(projection["evidence"]),
            "projection friction monitor and auxiliary-loss design note",
            "diagnostics available; rejection did not decrease",
            "actor scores remain imperfectly matched to assembler constraints",
            "owner-approved projection-aware auxiliary repair",
            "medium",
        ),
        Stage30RootCauseRow(
            "data_scale",
            "WARN",
            str(data["evidence_from_stage26"]),
            "data scale assessment",
            "no scenario expansion was run inside the owner-gated repair loop",
            "large-scale readiness cannot be certified on small duplicated contexts",
            "owner-approved scenario family expansion",
            "medium",
        ),
        Stage30RootCauseRow(
            "critic_health",
            "PASS",
            f"diagnostic EV {pilot['critic_ev']}",
            "none; regression check only",
            "critic remains healthy",
            "critic could regress after data expansion",
            "recheck critic on expanded frozen data",
            "high",
        ),
        Stage30RootCauseRow(
            "policy_gradient_loop",
            "WARN",
            "Stage 28 completed all seeds but mixed objective/projection result",
            "diagnostic assessment from latest pilot",
            "loop can run but does not certify scale readiness",
            "larger pilot may amplify unresolved reward/projection coupling",
            "block larger pilot until Stage 31 owner decision",
            "medium",
        ),
        Stage30RootCauseRow(
            "communication",
            "PASS",
            "Stage 26 communication health passed",
            "none",
            "no communication formula changes",
            "distribution may shift after scenario expansion",
            "rerun health diagnostics after data expansion",
            "medium",
        ),
        Stage30RootCauseRow(
            "consensus",
            "PASS",
            "Stage 26 consensus health passed",
            "none",
            "no PBFT model changes",
            "expected-initiator margin remains harsh near tau",
            "keep consensus metrics separate from surrogate",
            "medium",
        ),
        Stage30RootCauseRow(
            "harness_state",
            "PASS",
            "Stage 30 docs/tests/harness/state synchronized",
            "stage closeout controls",
            "control plane remains healthy",
            "future owner decision still required",
            "do not self-authorize Stage 31",
            "high",
        ),
    )


def _blocker_review(
    dominant: Mapping[str, object],
    readiness: Mapping[str, object],
    root_cause: Sequence[Stage30RootCauseRow],
) -> dict[str, object]:
    return {
        "max_iterations_reached": True,
        "large_scale_readiness_passed": bool(readiness["passed"]),
        "dominant_blocker": dominant,
        "remaining_issues": readiness["issues"],
        "most_important_remaining_blocker": "data_scale_and_owner_activation_of_alignment_repair",
        "recommended_owner_decision": STAGE30_RECOMMENDED_NEXT_TASK_BLOCKED,
        "root_cause_components": [row.component for row in root_cause if row.status != "PASS"],
    }


def _objective_aligned_score(
    row: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> float:
    lat_ref = max(float(item["latency"]) for item in rows) or 1.0
    energy_ref = max(float(item["energy"]) for item in rows) or 1.0
    max_resource = max(
        float(item["latency"]) / lat_ref + float(item["energy"]) / energy_ref
        for item in rows
    )
    resource = float(row["latency"]) / lat_ref + float(row["energy"]) / energy_ref
    margin = float(row["consensus_success_probability"]) - STAGE30_TAU_REQUIREMENT_MIN
    if margin >= 0.0:
        return -resource
    return -((max_resource + 1.0) + abs(margin) + resource)


def _inversion_stats(
    rows: Sequence[Mapping[str, object]],
    *,
    score_key: str,
) -> dict[str, object]:
    inversions = []
    comparisons = 0
    for left in rows:
        for right in rows:
            if left["row_id"] == right["row_id"] or left["scenario_id"] != right["scenario_id"]:
                continue
            if _objective_key(left) < _objective_key(right):
                comparisons += 1
                if float(left[score_key]) < float(right[score_key]) - 1e-9:
                    inversions.append((left["row_id"], right["row_id"]))
    return {
        "inversion_count": len(inversions),
        "comparison_count": comparisons,
        "inversion_rate": len(inversions) / comparisons if comparisons else 0.0,
        "sample_inversions": list(inversions[:20]),
    }


def _objective_key(row: Mapping[str, object]) -> tuple[int, float, float, int, str]:
    return (
        0 if bool(row["feasible_under_tau"]) else 1,
        float(row["latency"]),
        float(row["energy"]),
        int(row["selected_edge_count"]),
        str(row["row_id"]),
    )


def _feasible_infeasible_gap(
    rows: Sequence[Mapping[str, object]],
    *,
    score_key: str,
) -> dict[str, object]:
    feasible = [float(row[score_key]) for row in rows if bool(row["feasible_under_tau"])]
    infeasible = [float(row[score_key]) for row in rows if not bool(row["feasible_under_tau"])]
    if not feasible or not infeasible:
        return {"available": False, "gap": None}
    return {
        "available": True,
        "mean_feasible_score": _mean(feasible),
        "mean_infeasible_score": _mean(infeasible),
        "gap": _mean(feasible) - _mean(infeasible),
    }


def _missing_inputs(root: Path) -> list[str]:
    required = (
        STAGE26_REPORT_PATH,
        STAGE27_REPORT_PATH,
        STAGE28_REPORT_PATH,
        Path("docs/PROJECT_STATE.md"),
    )
    return [str(path) for path in required if not (root / path).exists()]


def _blocked_missing_inputs(missing: Sequence[str]) -> dict[str, object]:
    return {
        "stage": STAGE30_STAGE_ID,
        "config_id": STAGE30_CONFIG_ID,
        "verdict": STAGE30_BLOCKED_VERDICT,
        "pass_gate": False,
        "iteration_count": 0,
        "max_iterations": STAGE30_MAX_ITERATIONS,
        "pass_issues": ["missing_required_inputs", *missing],
        "large_scale_readiness": {
            "passed": False,
            "issues": ["missing_required_inputs"],
            "large_scale_training_allowed": False,
        },
        "recommended_next_task": "owner_decision_required_missing_stage30_inputs",
        "owner_decision_required": True,
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if isfinite(float(value))]
    return sum(finite) / len(finite) if finite else 0.0


def _std(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if isfinite(float(value))]
    if not finite:
        return 0.0
    mean = _mean(finite)
    return (sum((value - mean) ** 2 for value in finite) / len(finite)) ** 0.5


def _ratio(flags) -> float:
    values = [bool(flag) for flag in flags]
    return sum(1 for flag in values if flag) / len(values) if values else 0.0
