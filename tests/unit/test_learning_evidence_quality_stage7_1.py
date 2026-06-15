from marl_topology.data import (
    LearningEvidenceDataset,
    detect_actor_view_leakage,
    evaluate_learning_evidence_quality,
)
from marl_topology.evaluation.demo import build_demo_stack
from marl_topology.policies import build_local_observations


def _dataset():
    scene, graph, evaluator, oracle = build_demo_stack()
    observations = build_local_observations(
        scene=scene,
        graph=graph,
        link_records=evaluator.link_records,
        time_step=0,
    )
    baselines = oracle.evaluate_baselines(random_seed=7)
    variants = {
        "weak_disconnected_baseline": baselines["empty"].selected_edge_ids,
        "sparse_candidate": baselines["greedy_reliability"].selected_edge_ids,
        "dense_full_graph_baseline": baselines["full"].selected_edge_ids,
    }
    oracle_result = oracle.solve(random_seed=7)
    if oracle_result.evaluation is not None:
        variants["oracle_candidate_diagnostic"] = oracle_result.evaluation.selected_edge_ids

    from marl_topology.data import build_learning_evidence_dataset

    return build_learning_evidence_dataset(
        evaluator,
        observations=observations,
        topology_variants=variants,
        tau_requirement_min=0.9,
    )


def test_stage7_1_quality_report_passes_blocking_gates_and_warns_on_training_coverage() -> None:
    report = evaluate_learning_evidence_quality(_dataset())
    payload = report.to_dict()

    assert report.stage7_exit_ready is True
    assert report.stage8_policy_interface_ready is True
    assert report.training_execution_ready is False
    assert report.blocking_issue_count == 0
    assert report.warning_count == 2
    assert "actor_leakage_gate" in report.gates_passed
    assert "edge_delta_informativeness_gate" in report.gates_passed
    assert payload["recommended_next_task"] == (
        "stage_8_0_actor_policy_interface_contract_with_owner_approval"
    )
    assert payload["feasibility_summary"]["feasible_row_count"] == 0


def test_stage7_1_actor_leakage_detector_flags_forbidden_actor_fields() -> None:
    issues = detect_actor_view_leakage(
        (
            {
                "agent_id": "veh_0",
                "consensus_success_probability": 0.7,
                "global_topology": ("leak",),
            },
        )
    )

    assert len(issues) == 1
    assert issues[0].severity == "blocking"
    assert issues[0].code == "actor_safe_view_leakage"
    assert "consensus_success_probability" in issues[0].evidence["forbidden_fields"]
    assert "global_topology" in issues[0].evidence["forbidden_fields"]


def test_stage7_1_missing_targets_blocks_stage7_exit_ready() -> None:
    dataset = _dataset()
    empty_targets = LearningEvidenceDataset(
        dataset_id="missing_targets",
        rows=dataset.rows,
        edge_delta_targets=(),
    )

    report = evaluate_learning_evidence_quality(empty_targets)

    assert report.stage7_exit_ready is False
    assert "no_learning_targets" in [issue.code for issue in report.issues]


def test_stage7_1_missing_family_coverage_blocks_stage7_exit_ready() -> None:
    dataset = _dataset()
    weak_only = LearningEvidenceDataset(
        dataset_id="weak_only",
        rows=(dataset.rows[0],),
        edge_delta_targets=dataset.edge_delta_targets[:2],
    )

    report = evaluate_learning_evidence_quality(weak_only)

    assert report.stage7_exit_ready is False
    assert "missing_topology_family_coverage" in [issue.code for issue in report.issues]


def test_stage7_1_target_quality_requires_nonzero_deltas() -> None:
    dataset = _dataset()
    first_row_targets = tuple(
        target
        for target in dataset.edge_delta_targets
        if target.topology_id == dataset.rows[0].topology_id and target.action_type == "keep_edge"
    )
    keep_only = LearningEvidenceDataset(
        dataset_id="keep_only",
        rows=dataset.rows,
        edge_delta_targets=first_row_targets,
    )

    report = evaluate_learning_evidence_quality(keep_only)

    assert report.stage7_exit_ready is False
    issue_codes = [issue.code for issue in report.issues]
    assert "missing_edge_delta_action_types" in issue_codes
    assert "no_nonzero_edge_delta_targets" in issue_codes
