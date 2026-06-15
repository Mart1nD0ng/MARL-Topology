from pathlib import Path

import yaml

from marl_topology.evaluation.stage23_policy_gradient_readiness import (
    STAGE23_READINESS_STAGE_ID,
    run_stage23_controlled_policy_gradient_pilot_readiness_review,
)
from marl_topology.policies import UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage23_readiness_review_runs_without_policy_gradient() -> None:
    report = run_stage23_controlled_policy_gradient_pilot_readiness_review(
        project_root=ROOT
    )

    assert report["stage"] == STAGE23_READINESS_STAGE_ID
    assert report["cleanup_passed"] is True
    assert report["pilot_execution_allowed"] is False
    assert report["policy_gradient_performed"] is False
    assert report["checkpoint_written"] is False
    assert report["artifact_written"] is False
    assert report["v5_modified"] is False
    assert report["gates"]["selected_physical_pilot_harness_ready"]["passed"] is False
    assert report["gates"]["single_selected_action_semantics"]["passed"] is True
    assert UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID in report["gates"][
        "single_selected_action_semantics"
    ]["active_registry"]


def test_stage23_harness_task_declares_cleanup_and_readiness_boundaries() -> None:
    task = yaml.safe_load(
        _read("harness/tasks/stage23_controlled_policy_gradient_pilot_readiness_review.yaml")
    )

    assert task["id"] == "stage23_controlled_policy_gradient_pilot_readiness_review"
    for field in [
        "controlled_object",
        "desired_state",
        "required_evidence",
        "negative_checks",
        "low_entropy_cleanup_requirement",
        "policy_gradient_readiness_rule",
        "post_task_self_review_requirement",
    ]:
        assert task[field]


def test_stage23_docs_and_project_state_are_synchronized() -> None:
    for path in [
        "docs/STAGE23_LOW_ENTROPY_PREFLIGHT_CLEANUP.md",
        "docs/STAGE23_CONTROLLED_POLICY_GRADIENT_PILOT_READINESS_REVIEW.md",
        "harness/reports/post_task_self_review_stage23.md",
    ]:
        text = _read(path)
        assert "undirected_physical_link_v1" in text
        assert "policy-gradient" in text

    state = _read("docs/PROJECT_STATE.md")
    required = [
        "current_stage: post_stage_23_complete_pg_pilot_landed_awaiting_owner_decision",
        "post_stage_23_readiness_review_complete_policy_gradient_blocked_awaiting_owner_decision",
        "stage_23_controlled_policy_gradient_pilot_readiness_review",
        "stage_23_selected_physical_policy_gradient_landing",
        "stage23_low_entropy_preflight_cleanup_gate",
        "stage23_selected_physical_policy_gradient_landing_gate",
        "recommended_next_task: stage_24_policy_gradient_pilot_analysis_and_scale_readiness_review",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in state]
    assert not missing
