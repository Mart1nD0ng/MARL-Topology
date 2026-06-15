from pathlib import Path

import yaml

from marl_topology.models import LOCAL_GNN_EDGE_SCORER_MODEL_ID, build_model_registry
from marl_topology.policies import (
    ACTIVE_ACTION_SEMANTICS_ID,
    UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
    build_active_action_semantics_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage22_harness_task_declares_selection_cleanup_and_full_gnn() -> None:
    task = yaml.safe_load(
        _read("harness/tasks/stage22_action_semantics_ab_full_gnn_repair.yaml")
    )

    assert task["id"] == "stage22_action_semantics_ab_full_gnn_repair"
    for field in [
        "controlled_object",
        "desired_state",
        "relevant_lessons",
        "required_evidence",
        "hard_negative_checks",
        "low_entropy_cleanup_requirement",
        "action_semantics_selection_rule",
        "full_gnn_requirement",
        "post_task_self_review_requirement",
    ]:
        assert task[field]
    negative = " ".join(task["hard_negative_checks"])
    assert "PPO/MAPPO" in negative
    assert "COMA" in negative
    assert "Transformer" in negative
    assert "v5" in negative


def test_stage22_active_registry_contains_only_selected_physical_semantics() -> None:
    active = build_active_action_semantics_registry()

    assert ACTIVE_ACTION_SEMANTICS_ID == UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
    assert set(active) == {UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID}


def test_stage22_model_registry_promotes_full_gnn_not_toy_v1() -> None:
    registry = build_model_registry()
    entry = registry[LOCAL_GNN_EDGE_SCORER_MODEL_ID]

    assert LOCAL_GNN_EDGE_SCORER_MODEL_ID == "local_message_passing_gnn_edge_scorer_v2"
    assert entry.family == "local_message_passing_gnn"
    assert entry.allowed_stage == "stage_22_full_message_passing_gnn_repair"


def test_stage22_report_documents_selection_and_cleanup() -> None:
    for path in [
        "docs/STAGE22_ACTION_SEMANTICS_AB_FULL_GNN_REPAIR.md",
        "docs/STAGE22_ACTION_SEMANTICS_SELECTION.md",
        "docs/STAGE22_OBJECTIVE_AWARE_TEACHER.md",
        "docs/STAGE22_FULL_GNN_ACTOR.md",
    ]:
        text = _read(path)
        assert "undirected_physical_link_v1" in text
        assert "policy-gradient" in text


def test_stage22_project_state_is_synchronized_after_closeout() -> None:
    text = _read("docs/PROJECT_STATE.md")

    required = [
        "current_stage: post_stage_22_complete_awaiting_owner_decision_for_stage_23",
        "stage_22_action_semantics_ab_full_gnn_repair",
        "stage22_action_semantics_selection_gate",
        "stage22_full_message_passing_gnn_gate",
        "stage22_policy_gradient_still_blocked_gate",
        "recommended_next_task: stage_23_controlled_policy_gradient_pilot_readiness_review",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in text]
    assert not missing
