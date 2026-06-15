from marl_topology.evaluation.graph_necessity_metrics import (
    compute_graph_necessity_from_scores,
    graph_structure_rank_gap,
    local_quality_ambiguity_count,
    measured_sensitivity,
)


def test_stage34_graph_necessary_false_when_local_heuristic_matches_teacher() -> None:
    metrics = compute_graph_necessity_from_scores(
        teacher_score=0.95,
        local_heuristic_score=0.95,
    )

    assert metrics.graph_necessary is False
    assert metrics.graph_necessity_confidence == "none"


def test_stage34_bridge_weak_primary_and_role_sensitivity_can_trigger_label() -> None:
    bridge = compute_graph_necessity_from_scores(
        teacher_score=0.95,
        local_heuristic_score=0.94,
        bridge_sensitivity=0.20,
    )
    weak_primary = compute_graph_necessity_from_scores(
        teacher_score=0.95,
        local_heuristic_score=0.94,
        weak_primary_sensitivity=0.20,
    )
    role = compute_graph_necessity_from_scores(
        teacher_score=0.95,
        local_heuristic_score=0.94,
        role_sensitivity=0.20,
    )

    assert bridge.graph_necessary is True
    assert weak_primary.graph_necessary is True
    assert role.graph_necessary is True


def test_stage34_local_ambiguity_and_rank_gap_are_metric_derived() -> None:
    rows = [
        {"edge_id": "a--b", "local_quality": 0.91},
        {"edge_id": "b--c", "local_quality": 0.90},
        {"edge_id": "c--d", "local_quality": 0.72},
    ]
    teacher = ("b--c",)

    assert local_quality_ambiguity_count(rows, teacher, tolerance=0.02) >= 1
    assert graph_structure_rank_gap(rows, teacher) >= 2
    assert measured_sensitivity(0.95, 0.70) == 0.25


def test_stage34_metric_payload_documents_thresholds() -> None:
    payload = compute_graph_necessity_from_scores(
        teacher_score=0.95,
        local_heuristic_score=0.80,
    ).to_payload()

    assert payload["graph_necessary"] is True
    assert "thresholds" in payload
    assert payload["label_source"] == "metric_thresholds_not_family_name"
