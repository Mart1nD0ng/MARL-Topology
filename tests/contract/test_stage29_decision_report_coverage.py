from pathlib import Path

from marl_topology.evaluation.stage29_pre_scale_decision import (
    STAGE29_RECOMMENDED_NEXT_TASK,
    build_stage29_pre_scale_decision_report,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage29_docs_cover_decision_scorecard_and_root_cause_matrix() -> None:
    expected_docs = {
        "STAGE29_PRE_SCALE_DECISION_REVIEW.md",
        "STAGE29_SCALE_READINESS_SCORECARD.md",
        "STAGE29_ROOT_CAUSE_AND_DECISION_PACKET.md",
    }
    missing = [name for name in expected_docs if not (ROOT / "docs" / name).exists()]
    texts = {
        name: (ROOT / "docs" / name).read_text(encoding="utf-8")
        for name in expected_docs
        if (ROOT / "docs" / name).exists()
    }

    assert missing == []
    assert "recommended next task" in texts["STAGE29_PRE_SCALE_DECISION_REVIEW.md"]
    assert "reward_objective_alignment" in texts["STAGE29_ROOT_CAUSE_AND_DECISION_PACKET.md"]
    assert "assembler_projection_alignment" in texts["STAGE29_ROOT_CAUSE_AND_DECISION_PACKET.md"]
    assert "scale-up approved: `False`" in texts["STAGE29_PRE_SCALE_DECISION_REVIEW.md"]
    assert "critic ready: `True`" in texts["STAGE29_SCALE_READINESS_SCORECARD.md"]


def test_stage29_project_state_is_synchronized_after_closeout() -> None:
    state_text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    required_terms = [
        "post_stage_29_complete_pre_scale_decision_review_awaiting_owner_decision",
        "stage_29_pre_scale_decision_review",
        "stage29_pre_scale_decision_gate",
        "stage29_scale_readiness_blocked_gate",
        f"recommended_next_task: {STAGE29_RECOMMENDED_NEXT_TASK}",
        "Scale-up training remains blocked",
        "LSTM/recurrent PPO remains blocked",
    ]
    missing = [term for term in required_terms if term not in state_text]

    assert missing == []


def test_stage29_report_contains_all_decision_components() -> None:
    report = build_stage29_pre_scale_decision_report(project_root=ROOT)
    components = {row["component"] for row in report["root_cause_matrix"]}

    assert {
        "critic_baseline",
        "reward_objective_alignment",
        "assembler_projection_alignment",
        "reliability_constraint",
        "data_scale_readiness",
        "small_scale_policy_update",
        "boundary_and_harness",
    } <= components
