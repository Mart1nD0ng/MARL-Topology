"""Stage 29 pre-scale decision review from frozen Stage 26-28 evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path


STAGE29_STAGE_ID = "stage_29_pre_scale_decision_review"
STAGE29_CONFIG_ID = "stage29_pre_scale_decision_from_stage26_to_stage28_evidence_v1"
STAGE29_PASS_VERDICT = "stage29_pass_pre_scale_decision_review_complete"
STAGE29_FAIL_VERDICT = "stage29_fail_pre_scale_decision_review_blocked"
STAGE29_RECOMMENDED_NEXT_TASK = "stage_30_reward_objective_projection_alignment_repair"
STAGE29_RECOMMENDED_OPTION = "option_b_repair_reward_objective_and_projection_before_larger_pilot"

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


@dataclass(frozen=True, slots=True)
class Stage29DecisionRow:
    component: str
    status: str
    evidence: str
    likely_contribution: str
    confidence: str
    recommended_repair: str
    blocks_larger_pilot: bool
    blocks_scale_up: bool
    blocks_recurrent_policy: bool
    blocks_reward_tuning: bool
    blocks_sampler_change: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "status": self.status,
            "evidence": self.evidence,
            "likely_contribution": self.likely_contribution,
            "confidence": self.confidence,
            "recommended_repair": self.recommended_repair,
            "blocks_larger_pilot": self.blocks_larger_pilot,
            "blocks_scale_up": self.blocks_scale_up,
            "blocks_recurrent_policy": self.blocks_recurrent_policy,
            "blocks_reward_tuning": self.blocks_reward_tuning,
            "blocks_sampler_change": self.blocks_sampler_change,
        }


def build_stage29_pre_scale_decision_report(
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=False) if project_root else Path.cwd()
    missing_inputs = _missing_inputs(root)
    if missing_inputs:
        return _blocked_report(missing_inputs)

    stage26_report = _load_json(root / STAGE26_REPORT_PATH)
    stage27_report = _load_json(root / STAGE27_REPORT_PATH)
    stage28_report = _load_json(root / STAGE28_REPORT_PATH)
    state_text = (root / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    stage28_delta = stage28_report["aggregate"]["eval_delta_mean"]
    stage28_gate = stage28_report["pass_fail_gate"]
    critic_health = stage28_gate["critic_health"]
    stage25_comparison = stage28_report["stage25_historical_comparison"]
    stage26_components = {
        row["component"]: row
        for row in stage26_report["component_scorecard"]
    }

    decision_rows = _decision_rows(
        stage26_components=stage26_components,
        stage27_report=stage27_report,
        stage28_delta=stage28_delta,
        stage28_gate=stage28_gate,
        stage25_comparison=stage25_comparison,
        critic_health=critic_health,
    )
    scorecard = _scorecard(decision_rows)
    scale_blockers = [
        row.component
        for row in decision_rows
        if row.blocks_scale_up or row.blocks_larger_pilot
    ]
    forbidden_action_flags = {
        "new_training_run": False,
        "scale_up_training_run": False,
        "reward_weights_changed": False,
        "hyperparameter_tuning_performed": False,
        "sampler_switched": False,
        "checkpoint_created": False,
        "legacy_v5_modified": False,
        "recurrent_policy_introduced": False,
        "actor_stack_changed": False,
    }
    pass_issues: list[str] = []
    if not stage28_report.get("pass_gate"):
        pass_issues.append("stage28_report_not_passed")
    if not stage27_report.get("pass_gate"):
        pass_issues.append("stage27_report_not_passed")
    if not stage26_report.get("pass_gate"):
        pass_issues.append("stage26_report_not_passed")
    if not _project_state_has_stage28_evidence(state_text):
        pass_issues.append("project_state_missing_stage28_evidence")
    if any(forbidden_action_flags.values()):
        pass_issues.append("forbidden_action_detected")

    owner_options = {
        "option_a_larger_pilot_now": (
            "blocked because reward/objective and projection evidence remain mixed"
        ),
        "option_b_reward_objective_projection_repair": (
            "recommended because the critic is healthy but surrogate reward and "
            "projection rejection worsened in the repaired-critic rerun"
        ),
        "option_c_data_expansion_before_training": (
            "reasonable secondary option because Stage 26 data health remains "
            "small-pilot-only with duplicate contexts"
        ),
        "option_d_hold_training": (
            "reasonable if owner wants no further training work until the active "
            "objective contract is reviewed"
        ),
    }
    scale_readiness = {
        "larger_pilot_ready": False,
        "scale_up_ready": False,
        "critic_ready": True,
        "reward_objective_ready": False,
        "projection_ready": False,
        "data_ready_for_scale": False,
        "reason": (
            "Stage 28 fixed the value-baseline failure, but the repaired-critic "
            "policy update still worsened surrogate reward and projection "
            "rejection while data health remains small-pilot-only."
        ),
    }
    return {
        "stage": STAGE29_STAGE_ID,
        "config_id": STAGE29_CONFIG_ID,
        "verdict": STAGE29_PASS_VERDICT if not pass_issues else STAGE29_FAIL_VERDICT,
        "pass_gate": not pass_issues,
        "pass_issues": pass_issues,
        "source_reports": {
            "stage26": str(STAGE26_REPORT_PATH),
            "stage27": str(STAGE27_REPORT_PATH),
            "stage28": str(STAGE28_REPORT_PATH),
        },
        "stage28_evidence": {
            "completed_seed_count": stage28_report["aggregate"]["completed_seed_count"],
            "seed_count": stage28_report["aggregate"]["seed_count"],
            "stop_reasons": stage28_report["aggregate"]["stop_reasons"],
            "eval_delta_mean": stage28_delta,
            "critic_health": critic_health,
            "stage25_historical_comparison": stage25_comparison,
            "objective_improvement_flags": stage28_gate["objective_improvement_flags"],
        },
        "stage27_evidence": {
            "selected_critic_id": stage27_report["selected_critic_id"],
            "mappo_readiness": stage27_report["mappo_readiness"],
            "selection_reason": stage27_report["selection_reason"],
        },
        "stage26_prior_blockers": {
            key: stage26_components[key]
            for key in (
                "data_health",
                "reward_objective_health",
                "assembler_health",
                "sampler_health",
                "actor_health",
                "critic_health",
                "scale_readiness",
            )
        },
        "decision_scorecard": scorecard,
        "root_cause_matrix": [row.to_dict() for row in decision_rows],
        "scale_readiness": scale_readiness,
        "decision_packet": {
            "recommended_option": STAGE29_RECOMMENDED_OPTION,
            "recommended_next_task": STAGE29_RECOMMENDED_NEXT_TASK,
            "owner_options": owner_options,
            "larger_pilot_approved": False,
            "scale_up_approved": False,
            "owner_decision_required": True,
            "scale_blockers": scale_blockers,
            "rationale": scale_readiness["reason"],
        },
        "forbidden_action_flags": forbidden_action_flags,
        "recommended_next_task": STAGE29_RECOMMENDED_NEXT_TASK,
        "owner_decision_required": True,
    }


def _missing_inputs(root: Path) -> list[str]:
    required = (
        STAGE26_REPORT_PATH,
        STAGE27_REPORT_PATH,
        STAGE28_REPORT_PATH,
        Path("docs/PROJECT_STATE.md"),
    )
    return [str(path) for path in required if not (root / path).exists()]


def _blocked_report(missing_inputs: Sequence[str]) -> dict[str, object]:
    return {
        "stage": STAGE29_STAGE_ID,
        "config_id": STAGE29_CONFIG_ID,
        "verdict": STAGE29_FAIL_VERDICT,
        "pass_gate": False,
        "pass_issues": ["missing_required_inputs", *missing_inputs],
        "decision_packet": {
            "recommended_option": "option_d_hold_training",
            "recommended_next_task": "owner_decision_required_before_any_more_training",
            "larger_pilot_approved": False,
            "scale_up_approved": False,
            "owner_decision_required": True,
            "scale_blockers": ["missing_required_inputs"],
            "rationale": "Stage 29 cannot make a data-backed decision without Stage 26-28 evidence.",
        },
        "forbidden_action_flags": {
            "new_training_run": False,
            "scale_up_training_run": False,
            "reward_weights_changed": False,
            "hyperparameter_tuning_performed": False,
            "sampler_switched": False,
            "checkpoint_created": False,
            "legacy_v5_modified": False,
            "recurrent_policy_introduced": False,
            "actor_stack_changed": False,
        },
        "recommended_next_task": "owner_decision_required_before_any_more_training",
        "owner_decision_required": True,
    }


def _decision_rows(
    *,
    stage26_components: Mapping[str, Mapping[str, object]],
    stage27_report: Mapping[str, object],
    stage28_delta: Mapping[str, float],
    stage28_gate: Mapping[str, object],
    stage25_comparison: Mapping[str, object],
    critic_health: Mapping[str, float],
) -> list[Stage29DecisionRow]:
    stage25_metric_deltas = stage25_comparison["metric_deltas"]
    critic_ready = (
        critic_health["mean_update_explained_variance"] > 0.10
        and critic_health["mean_value_return_correlation"] > 0.30
        and stage27_report["selected_critic_id"]
        == "centralized_message_passing_graph_value_critic_v1"
    )
    reward_mismatch = (
        stage28_delta["mean_reward_surrogate_delta"] < 0.0
        and (
            stage28_delta["latency_delta"] < 0.0
            or stage28_delta["energy_delta"] < 0.0
        )
    )
    projection_worse = stage28_delta["top_proposal_rejection_rate_delta"] > 0.0
    reliability_within_bound = (
        stage28_delta["tau_feasible_rate_delta"] >= -0.05
        and stage28_delta["violation_rate_delta"] <= 0.05
    )
    rows = [
        Stage29DecisionRow(
            component="critic_baseline",
            status="PASS" if critic_ready else "FAIL",
            evidence=(
                "Stage 28 critic EV "
                f"{critic_health['mean_update_explained_variance']:.6f}, "
                "value-return correlation "
                f"{critic_health['mean_value_return_correlation']:.6f}, "
                "selected graph value critic reused."
            ),
            likely_contribution="low after Stage 27/28 repair",
            confidence="high",
            recommended_repair="keep selected graph value critic active for future owner-approved runs",
            blocks_larger_pilot=not critic_ready,
            blocks_scale_up=not critic_ready,
            blocks_recurrent_policy=True,
            blocks_reward_tuning=False,
            blocks_sampler_change=False,
        ),
        Stage29DecisionRow(
            component="reward_objective_alignment",
            status="FAIL" if reward_mismatch else "PASS",
            evidence=(
                "Stage 28 latency delta "
                f"{stage28_delta['latency_delta']:.8f}, energy delta "
                f"{stage28_delta['energy_delta']:.8f}, surrogate reward delta "
                f"{stage28_delta['mean_reward_surrogate_delta']:.6f}."
            ),
            likely_contribution="high because the policy improved latency/energy while the surrogate worsened",
            confidence="medium-high",
            recommended_repair=(
                "run a reward/objective and projection-alignment repair stage without "
                "selecting final tau or tuning weights by default"
            ),
            blocks_larger_pilot=reward_mismatch,
            blocks_scale_up=reward_mismatch,
            blocks_recurrent_policy=True,
            blocks_reward_tuning=True,
            blocks_sampler_change=True,
        ),
        Stage29DecisionRow(
            component="assembler_projection_alignment",
            status="WARN" if projection_worse else "PASS",
            evidence=(
                "Top proposal rejection delta versus supervised "
                f"{stage28_delta['top_proposal_rejection_rate_delta']:.6f}; "
                "Stage 28 minus Stage 25 top rejection "
                f"{stage25_metric_deltas['top_proposal_rejection_rate_stage28_minus_stage25']:.6f}."
            ),
            likely_contribution="medium because repaired-critic updates did not reduce projection friction",
            confidence="medium",
            recommended_repair="diagnose accepted/rejected score separation and projection constraints before larger pilot",
            blocks_larger_pilot=projection_worse,
            blocks_scale_up=projection_worse,
            blocks_recurrent_policy=True,
            blocks_reward_tuning=False,
            blocks_sampler_change=True,
        ),
        Stage29DecisionRow(
            component="reliability_constraint",
            status="WARN" if reliability_within_bound else "FAIL",
            evidence=(
                "Tau-feasible delta "
                f"{stage28_delta['tau_feasible_rate_delta']:.6f}; violation delta "
                f"{stage28_delta['violation_rate_delta']:.6f}; allowed absolute bound 0.05."
            ),
            likely_contribution="medium because reliability stayed within bound but moved in the wrong direction",
            confidence="medium",
            recommended_repair="keep reliability as a hard gate in any future pilot",
            blocks_larger_pilot=False,
            blocks_scale_up=True,
            blocks_recurrent_policy=True,
            blocks_reward_tuning=True,
            blocks_sampler_change=False,
        ),
        Stage29DecisionRow(
            component="data_scale_readiness",
            status=str(stage26_components["data_health"]["status"]),
            evidence=str(stage26_components["data_health"]["likely_contribution"]),
            likely_contribution="medium because Stage 26 still labels data as small-pilot-only",
            confidence="medium",
            recommended_repair="expand or diversify scenario evidence before any scale-up claim",
            blocks_larger_pilot=False,
            blocks_scale_up=True,
            blocks_recurrent_policy=True,
            blocks_reward_tuning=False,
            blocks_sampler_change=False,
        ),
        Stage29DecisionRow(
            component="small_scale_policy_update",
            status="WARN",
            evidence=(
                "All three seeds completed, latency/energy improved versus supervised, "
                "but reward and projection did not improve together."
            ),
            likely_contribution="medium because the update signal is usable but not clean enough for larger training",
            confidence="medium",
            recommended_repair="do not run a larger pilot until reward/projection blockers are resolved or owner accepts the risk",
            blocks_larger_pilot=True,
            blocks_scale_up=True,
            blocks_recurrent_policy=True,
            blocks_reward_tuning=True,
            blocks_sampler_change=True,
        ),
        Stage29DecisionRow(
            component="boundary_and_harness",
            status="PASS" if bool(stage28_gate["forbidden_work_avoided"]) else "FAIL",
            evidence="Stage 28 reported reward weights unchanged, sampler fixed, no checkpoint, and no scale-up.",
            likely_contribution="low; control-plane evidence is healthy",
            confidence="high",
            recommended_repair="keep owner gate and manifest/report discipline",
            blocks_larger_pilot=not bool(stage28_gate["forbidden_work_avoided"]),
            blocks_scale_up=not bool(stage28_gate["forbidden_work_avoided"]),
            blocks_recurrent_policy=True,
            blocks_reward_tuning=False,
            blocks_sampler_change=False,
        ),
    ]
    return rows


def _scorecard(rows: Sequence[Stage29DecisionRow]) -> list[dict[str, object]]:
    score_by_status = {"PASS": 90, "WARN": 65, "FAIL": 35}
    return [
        {
            "component": row.component,
            "status": row.status,
            "score": score_by_status.get(row.status, 50),
            "blocks_larger_pilot": row.blocks_larger_pilot,
            "blocks_scale_up": row.blocks_scale_up,
            "confidence": row.confidence,
        }
        for row in rows
    ]


def _project_state_has_stage28_evidence(state_text: str) -> bool:
    required = (
        "stage_28_rerun_small_scale_mappo_with_repaired_critic",
        "stage28_repaired_critic_rerun_gate",
        "stage_29_pre_scale_decision_review",
    )
    return all(term in state_text for term in required)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
