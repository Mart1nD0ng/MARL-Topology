from pathlib import Path

from marl_topology.evaluation.stage30_repair_diagnostics import (
    STAGE30_BLOCKED_VERDICT,
    STAGE30_RECOMMENDED_NEXT_TASK_BLOCKED,
    build_stage30_closed_loop_report,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage30_large_scale_readiness_requires_all_components_to_pass() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)
    readiness = report["large_scale_readiness"]

    assert readiness["passed"] is False
    assert readiness["large_scale_training_allowed"] is False
    assert "component_failures:reliability_margin,surrogate_objective" in readiness["issues"]
    assert readiness["readiness_criteria"]["data"] == "FAIL"
    assert readiness["readiness_criteria"]["projection"] == "FAIL"


def test_stage30_large_scale_readiness_blocks_owner_gated_alignment_activation() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)
    issues = report["large_scale_readiness"]["issues"]

    assert "surrogate_alignment_not_active_without_owner_decision" in issues
    assert report["reward_objective_repair"]["active_training_surrogate_changed"] is False
    assert report["reward_objective_repair"]["owner_activation_required"] is True


def test_stage30_project_state_is_synchronized_to_blocked_closeout() -> None:
    state_text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    required_terms = [
        "current_stage: post_stage30_repair_loop_blocked_awaiting_owner_decision",
        "stage30_closed_loop_repair_until_scale_readiness",
        "stage30_large_scale_readiness_gate_failed",
        "stage31_owner_decision_on_data_expansion_and_active_alignment_repair",
        "Large-scale training remains blocked",
        "LSTM/recurrent PPO remains blocked",
    ]
    missing = [term for term in required_terms if term not in state_text]

    assert missing == []


def test_stage30_report_closeout_recommends_owner_decision_not_scale_training() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)

    assert report["verdict"] == STAGE30_BLOCKED_VERDICT
    assert report["recommended_next_task"] == STAGE30_RECOMMENDED_NEXT_TASK_BLOCKED
    assert report["owner_decision_required"] is True
    assert report["large_scale_readiness"]["owner_decision_required"] is True
