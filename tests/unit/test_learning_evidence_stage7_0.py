from pathlib import Path

from marl_topology.data import (
    ACTOR_VIEW_FORBIDDEN_FIELDS,
    STAGE7_0_ARTIFACT_SCOPE,
    TARGET_ROLE_LEARNING_ONLY,
    LearningEvidenceViolation,
    build_learning_evidence_dataset,
    validate_actor_safe_view,
    write_learning_evidence_artifact,
)
from marl_topology.evaluation.demo import build_demo_stack
from marl_topology.policies import build_local_observations
from marl_topology.training import build_valid_stage5_10_dry_run_manifest


def _dataset():
    scene, graph, evaluator, oracle = build_demo_stack()
    observations = build_local_observations(
        scene=scene,
        graph=graph,
        link_records=evaluator.link_records,
        time_step=0,
    )
    baselines = oracle.evaluate_baselines(random_seed=3)
    variants = {
        "weak_disconnected_baseline": baselines["empty"].selected_edge_ids,
        "sparse_candidate": baselines["greedy_reliability"].selected_edge_ids,
        "dense_full_graph_baseline": baselines["full"].selected_edge_ids,
    }
    return build_learning_evidence_dataset(
        evaluator,
        observations=observations,
        topology_variants=variants,
        tau_requirement_min=0.9,
    )


def test_stage7_0_builds_learning_evidence_rows_with_required_views() -> None:
    dataset = _dataset()

    assert dataset.row_count == 3
    assert dataset.target_count > 0
    assert set(dataset.views) == {
        "actor_safe_view",
        "critic_centralized_view",
        "learning_target_view",
        "diagnostics_view",
    }

    row = dataset.rows[0]
    assert row.scenario_id
    assert row.topology_id
    assert 0.0 <= row.consensus_success_probability <= 1.0
    assert row.latency >= 0.0
    assert row.energy >= 0.0
    assert "candidate_edge_ids" in row.critic_view
    assert row.actor_safe_rows
    for actor_row in row.actor_safe_rows:
        assert not (set(actor_row) & ACTOR_VIEW_FORBIDDEN_FIELDS)


def test_stage7_0_edge_delta_targets_are_learning_target_only() -> None:
    dataset = _dataset()
    action_types = {target.action_type for target in dataset.edge_delta_targets}

    assert {"add_edge", "remove_edge", "keep_edge"} <= action_types
    assert all(target.target_role == TARGET_ROLE_LEARNING_ONLY for target in dataset.edge_delta_targets)
    assert any(target.delta_consensus_success_probability != 0.0 for target in dataset.edge_delta_targets)

    row = dataset.rows[0]
    assert row.learning_targets
    assert all(target["target_role"] == TARGET_ROLE_LEARNING_ONLY for target in row.learning_targets)
    assert "learning_targets" not in row.actor_safe_rows[0]


def test_stage7_0_actor_view_rejects_oracle_objective_future_and_global_fields() -> None:
    forbidden_rows = [
        {"agent_id": "veh_0", "oracle_label": "diagnostic_only"},
        {"agent_id": "veh_0", "consensus_success_probability": 0.9},
        {"agent_id": "veh_0", "latency": 0.1},
        {"agent_id": "veh_0", "energy": 1.0},
        {"agent_id": "veh_0", "future_consensus_outcome": True},
        {"agent_id": "veh_0", "global_topology": ("leak",)},
    ]

    for row in forbidden_rows:
        try:
            validate_actor_safe_view((row,))
        except LearningEvidenceViolation as exc:
            assert "actor_safe_view" in str(exc)
        else:
            raise AssertionError(f"forbidden actor row was accepted: {row}")


def test_stage7_0_writer_blocks_missing_approval(tmp_path: Path) -> None:
    manifest = build_valid_stage5_10_dry_run_manifest(
        overrides={"artifact_scope": STAGE7_0_ARTIFACT_SCOPE}
    )

    try:
        write_learning_evidence_artifact(
            _dataset(),
            manifest=manifest,
            project_root=tmp_path,
            owner_approved_evidence_export=False,
        )
    except LearningEvidenceViolation as exc:
        assert "owner approval" in str(exc)
    else:
        raise AssertionError("writer accepted missing owner approval")


def test_stage7_0_writer_blocks_invalid_manifest_and_wrong_scope(tmp_path: Path) -> None:
    dataset = _dataset()
    invalid_manifest = build_valid_stage5_10_dry_run_manifest(
        artifact_root="result_save\\..\\outside",
        overrides={"artifact_scope": STAGE7_0_ARTIFACT_SCOPE},
    )

    try:
        write_learning_evidence_artifact(
            dataset,
            manifest=invalid_manifest,
            project_root=tmp_path,
            owner_approved_evidence_export=True,
        )
    except LearningEvidenceViolation as exc:
        assert "manifest validation failed" in str(exc)
    else:
        raise AssertionError("writer accepted invalid manifest")

    wrong_scope = build_valid_stage5_10_dry_run_manifest(
        overrides={"artifact_scope": "checkpoint_output"}
    )
    try:
        write_learning_evidence_artifact(
            dataset,
            manifest=wrong_scope,
            project_root=tmp_path,
            owner_approved_evidence_export=True,
        )
    except LearningEvidenceViolation as exc:
        assert "artifact_scope" in str(exc)
    else:
        raise AssertionError("writer accepted wrong artifact scope")


def test_stage7_0_writer_accepts_valid_tmp_evidence_export_only(tmp_path: Path) -> None:
    dataset = _dataset()
    manifest = build_valid_stage5_10_dry_run_manifest(
        overrides={
            "run_id": "stage7_0_tmp_evidence_export",
            "stage_id": "stage_7_0_learning_evidence_dataset_generation_with_owner_approval",
            "owner_approval_id": "owner_approved_stage7_0_evidence_only_test",
            "artifact_scope": STAGE7_0_ARTIFACT_SCOPE,
        }
    )

    result = write_learning_evidence_artifact(
        dataset,
        manifest=manifest,
        project_root=tmp_path,
        owner_approved_evidence_export=True,
    )

    assert result.writes_performed is True
    assert result.artifact_scope == STAGE7_0_ARTIFACT_SCOPE
    assert Path(result.manifest_path).exists()
    assert Path(result.evidence_path).exists()
    assert Path(result.artifact_dir).is_relative_to(
        tmp_path / "result_save" / STAGE7_0_ARTIFACT_SCOPE
    )
    written_names = {path.name for path in Path(result.artifact_dir).iterdir()}
    assert written_names == {"manifest.json", "learning_evidence.json"}
    assert not any("checkpoint" in name or "model" in name for name in written_names)
