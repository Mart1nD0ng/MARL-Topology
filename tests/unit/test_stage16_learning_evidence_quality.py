from pathlib import Path

from marl_topology.data.learning_evidence_stage16 import (
    STAGE16_DATASET_ID,
    STAGE16_TAU_REQUIREMENT_MIN,
    build_stage16_learning_evidence_report,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage16_dataset_covers_required_evidence_families() -> None:
    build = build_stage16_learning_evidence_report()
    report = build.report
    coverage = report["evidence_coverage"]

    assert build.dataset.dataset_id == STAGE16_DATASET_ID
    assert report["tau_requirement_min"] == STAGE16_TAU_REQUIREMENT_MIN
    assert report["row_count"] >= 50
    assert report["target_count"] > report["row_count"]
    assert coverage["tau_ge_0_9_feasible_row_count"] > 0
    assert coverage["near_threshold_row_count"] > 0
    assert coverage["infeasible_hard_row_count"] > 0
    assert coverage["sparse_feasible_topology_count"] > 0
    assert coverage["full_graph_resource_dominated_topology_count"] > 0
    assert coverage["interference_penalty_example_count"] > 0
    assert coverage["weak_primary_row_count"] > 0
    assert coverage["center_primary_row_count"] > 0
    assert coverage["weak_center_primary_contrast"]["has_contrast"] is True
    assert (
        coverage["weak_center_primary_contrast"]["probability_delta_center_minus_weak"]
        > 0.0
    )


def test_stage16_contains_real_multistep_actor_safe_sequence() -> None:
    build = build_stage16_learning_evidence_report()
    coverage = build.report["evidence_coverage"]

    assert coverage["real_multistep_sequence_count"] >= 1
    assert coverage["real_multistep_time_steps"][
        "stage16_real_mobility_actor_safe_sequence_v1"
    ] == [0, 1, 2]

    sequence_rows = [
        row
        for row in build.dataset.rows
        if row.diagnostics.get("is_real_multistep_actor_safe_sequence")
    ]
    assert sequence_rows
    assert sorted({actor_row["time_step"] for row in sequence_rows for actor_row in row.actor_safe_rows}) == [
        0,
        1,
        2,
    ]
    first_edge_distances = {
        actor_row["time_step"]: tuple(
            round(float(neighbor.distance_3d_m), 3)
            for neighbor in actor_row["local_neighbor_observations"]
        )
        for row in sequence_rows
        for actor_row in row.actor_safe_rows
        if actor_row["agent_id"] == "rsu_0"
    }
    assert len({distances for distances in first_edge_distances.values()}) > 1


def test_stage16_edge_delta_rebuild_has_balance_and_delta_metrics() -> None:
    report = build_stage16_learning_evidence_report().report
    edge = report["edge_delta_target_rebuild"]

    action_counts = edge["action_type_counts"]
    assert action_counts["add_edge"] > 0
    assert action_counts["remove_edge"] > 0
    assert action_counts["keep_edge"] > 0
    assert edge["positive_probability_delta_count"] > 0
    assert edge["negative_probability_delta_count"] > 0
    assert edge["helpful_edge_count"] > 0
    assert edge["harmful_edge_count"] > 0
    assert edge["feasibility_changing_delta_count"] > 0
    assert edge["delta_consensus_success_probability_nonzero_count"] > 0
    assert edge["delta_latency_nonzero_count"] > 0
    assert edge["delta_energy_nonzero_count"] > 0
    assert edge["delta_surrogate_diagnostic_nonzero_count"] > 0
    assert edge["positive_negative_balance"]["has_helpful_and_harmful_edges"] is True
    assert edge["rare_safety_samples"]["rare_sample_count"] > 0
    assert edge["oracle_gap"]["oracle_comparable_row_count"] > 0


def test_stage16_quality_answers_keep_scale_up_training_blocked() -> None:
    report = build_stage16_learning_evidence_report().report
    answers = report["data_quality_answers"]
    contradiction = report["actor_observation_label_contradictions"]

    assert answers["tau_ge_0_9_feasible_positive_examples_present"] is True
    assert answers["sparse_feasible_examples_present"] is True
    assert answers["real_multistep_evidence_present"] is True
    assert answers["current_data_sufficient_to_continue_supervised_training"] is False
    assert answers["current_data_sufficient_for_smoke_or_dry_run_only"] is True
    assert answers["rerun_stage11_to_stage15_now_recommended"] is False
    assert answers["scale_up_training_still_blocked"] is True
    assert contradiction["has_identical_local_observations_with_contradictory_labels"] is True
    assert (
        answers["actor_observable_features_sufficient_to_disambiguate_edge_labels"]
        is False
    )


def test_stage16_base_quality_has_no_actor_leakage_or_blocking_issue() -> None:
    report = build_stage16_learning_evidence_report().report
    base_quality = report["base_quality_report"]

    assert base_quality["blocking_issue_count"] == 0
    assert base_quality["actor_leakage_summary"]["leakage_issue_count"] == 0
    assert base_quality["target_quality_summary"]["missing_action_types"] == []
    assert base_quality["training_execution_ready"] is False
    assert base_quality["checkpoint_creation_allowed"] is False


def test_stage16_builder_does_not_write_result_save_artifacts() -> None:
    result_save = ROOT / "result_save"
    before = sorted(path.relative_to(result_save) for path in result_save.rglob("*") if path.is_file())

    build_stage16_learning_evidence_report()

    after = sorted(path.relative_to(result_save) for path in result_save.rglob("*") if path.is_file())
    assert after == before


def test_stage16_source_keeps_training_and_model_expansion_out() -> None:
    source = (ROOT / "src" / "marl_topology" / "data" / "learning_evidence_stage16.py").read_text(
        encoding="utf-8"
    )
    forbidden_terms = [
        "import torch",
        "import tensorflow",
        "optimizer.step",
        ".backward(",
        "torch.save",
        "checkpoint_path",
        "class COMA",
        "class Transformer",
        "PPOTrainer",
        "MAPPOTrainer",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in forbidden_terms if term in source]
    assert not hits, f"Stage 16 source introduced forbidden implementation terms: {hits}"
