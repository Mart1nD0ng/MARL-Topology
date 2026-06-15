from pathlib import Path

from marl_topology.data import (
    STAGE7_0_ARTIFACT_SCOPE,
    STAGE7_COMPLETION_DATASET_ID,
    STAGE7_COMPLETION_STAGE_ID,
    TAU_REQUIREMENT_MIN,
    TOPOLOGY_VARIANT_NAMES,
    build_stage7_completion_dataset,
    build_stage7_completion_report,
    iter_stage7_completion_scenario_fixtures,
    write_stage7_completion_artifact,
)


def test_stage7_completion_dataset_covers_scenarios_and_topology_variants() -> None:
    dataset = build_stage7_completion_dataset()
    scenario_ids = {row.scenario_id for row in dataset.rows}
    topology_names = {row.topology_name for row in dataset.rows}

    assert dataset.dataset_id == STAGE7_COMPLETION_DATASET_ID
    assert len(scenario_ids) >= 5
    assert set(TOPOLOGY_VARIANT_NAMES) <= topology_names
    assert dataset.row_count >= 25
    assert dataset.target_count > 0
    assert len(iter_stage7_completion_scenario_fixtures()) >= 5


def test_stage7_completion_has_feasible_infeasible_balance_and_tradeoff() -> None:
    build = build_stage7_completion_report()
    report = build.report

    assert report["class_balance"]["has_both_classes"] is True
    assert report["class_balance"]["feasible_count"] > 0
    assert report["class_balance"]["infeasible_count"] > 0
    assert report["sparse_vs_full_tradeoff"]["has_sparse_better_than_full_case"] is True
    assert report["base_quality_report"]["blocking_issue_count"] == 0
    assert report["tau_requirement_min"] == TAU_REQUIREMENT_MIN


def test_stage7_completion_edge_delta_targets_include_swap_and_surrogate_diagnostic() -> None:
    dataset = build_stage7_completion_dataset()
    action_counts = {target.action_type for target in dataset.edge_delta_targets}

    assert {"add_edge", "remove_edge", "keep_edge", "swap_edge"} <= action_counts
    assert any(
        target.delta_reward_surrogate_diagnostic
        and any(value != 0.0 for value in target.delta_reward_surrogate_diagnostic.values())
        for target in dataset.edge_delta_targets
    )
    assert all(target.target_role == "learning_target_only" for target in dataset.edge_delta_targets)


def test_stage7_completion_keeps_oracle_and_surrogate_out_of_actor_view() -> None:
    dataset = build_stage7_completion_dataset()
    forbidden = {
        "oracle_label",
        "oracle_status",
        "oracle_action",
        "oracle_feasibility",
        "reward_surrogate",
        "reward_reliability_penalty",
        "reward_latency_penalty",
        "reward_energy_penalty",
        "reward_config_id",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    }

    for row in dataset.rows:
        for actor_row in row.actor_safe_rows:
            assert not (set(actor_row) & forbidden)
    assert any(row.diagnostics["is_oracle_candidate"] for row in dataset.rows)


def test_stage7_completion_exit_criteria_all_pass() -> None:
    report = build_stage7_completion_report().report

    assert report["stage"] == STAGE7_COMPLETION_STAGE_ID
    assert report["stage7_exit_ready"] is True
    assert report["stage8_policy_interface_ready"] is True
    assert report["training_execution_ready"] is False
    assert report["model_implementation_allowed"] is False
    assert all(report["exit_criteria"].values())
    assert report["actor_observable_predictability_risk"]["risk"] in {"medium", "high"}
    assert report["actor_observable_predictability_risk"]["actor_observable_edge_coverage"] == 1.0


def test_stage7_completion_artifact_writer_uses_manifest_guard(tmp_path: Path) -> None:
    result = write_stage7_completion_artifact(
        project_root=tmp_path,
        owner_approved_evidence_export=True,
    )

    artifact_dir = Path(result["artifact_dir"])
    assert result["artifact_scope"] == STAGE7_0_ARTIFACT_SCOPE
    assert result["writes_performed"] is True
    assert result["manifest_validated"] is True
    assert artifact_dir.is_relative_to(tmp_path / "result_save" / STAGE7_0_ARTIFACT_SCOPE)
    assert set(result["artifact_file_names"]) == {
        "learning_evidence.json",
        "manifest.json",
        "quality_report.json",
    }
    assert not any("checkpoint" in name or "model" in name for name in result["artifact_file_names"])
