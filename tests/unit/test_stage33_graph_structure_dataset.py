from marl_topology.data.stage33_graph_structure_dataset import (
    STAGE33_GRAPH_STRUCTURE_FAMILIES,
    Stage33GraphStructureConfig,
    Stage33GraphStructureDatasetViolation,
    build_stage33_graph_structure_dataset,
)


def test_stage33_graph_structure_dataset_contains_all_required_families() -> None:
    dataset = build_stage33_graph_structure_dataset(
        Stage33GraphStructureConfig(scenario_count=7, node_count_choices=(6,))
    )
    quality = dataset.quality_report

    assert dataset.dataset_id == "stage33_graph_structure_necessity_dataset_v1"
    assert quality["all_required_families_present"] is True
    assert set(quality["families_present"]) == set(STAGE33_GRAPH_STRUCTURE_FAMILIES)
    assert quality["split_leakage_overlap_count"] == 0
    assert quality["duplicate_context_rate"] == 0.0
    assert quality["stage3_stage4_evaluator_used"] is True
    assert quality["simple_link_model_fallback_used"] is False
    assert quality["hardcoded_fake_feasibility"] is False


def test_stage33_graph_necessity_metrics_are_measured_and_finite() -> None:
    dataset = build_stage33_graph_structure_dataset(
        Stage33GraphStructureConfig(scenario_count=7, node_count_choices=(6,))
    )

    for record in dataset.records:
        diagnostics = record.diagnostics
        assert diagnostics["stage3_stage4_evaluator_used"] is True
        assert diagnostics["objective_aware_teacher_used"] is True
        assert diagnostics["all_metrics_finite"] is True
        assert "graph_necessity_score" in diagnostics
        assert "local_edge_heuristic_teacher_gap" in diagnostics
        assert "similar_local_quality_objective_rank_divergence" in diagnostics
        assert diagnostics["hardcoded_fake_feasibility"] is False


def test_stage33_dataset_keeps_node_counts_and_tau_within_owner_boundary() -> None:
    config = Stage33GraphStructureConfig(
        scenario_count=7,
        node_count_choices=(6, 10),
    )
    assert config.tau_requirement_min == 0.9
    assert config.to_payload()["node_count_choices"] == [6, 10]

    # Owner boundary widened from 6..10 to 6..20 to enable the large-scale / variable-density
    # generalization study (the local message-passing GNN is size-invariant; the budget is
    # scale-ready at rsu=64). 6..20 stays inside what the stage31 generator builds in practice.
    Stage33GraphStructureConfig(scenario_count=7, node_count_choices=(6, 16))
    for bad_counts in ((5,), (21,)):
        try:
            Stage33GraphStructureConfig(scenario_count=7, node_count_choices=bad_counts)
        except Stage33GraphStructureDatasetViolation:
            pass
        else:
            raise AssertionError("Stage33 accepted a node count outside 6..20")
