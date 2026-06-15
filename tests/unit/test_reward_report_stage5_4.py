from marl_topology.evaluation import (
    STAGE5_4_REWARD_REPORT_STAGE_ID,
    build_stage5_4_reward_report_integration,
)


def test_stage5_4_report_integrates_surrogate_components_without_scalar_signal() -> None:
    report = build_stage5_4_reward_report_integration()
    summary = report["diagnostic_summary"]
    checks = report["checks"]

    assert report["stage"] == STAGE5_4_REWARD_REPORT_STAGE_ID
    assert report["component_policy"] == "component_only_no_scalar_reward_v1"
    assert summary["row_count"] == 8
    assert summary["constraint_satisfied_count"] == 4
    assert summary["constraint_violation_count"] == 4
    assert summary["scalar_surrogate_reported"] is False
    assert checks["scalar_surrogate_not_reported"] is True
    assert checks["reward_weight_calibration_performed"] is False
    assert checks["training_run"] is False
    assert checks["model_code_added"] is False
    assert checks["v5_code_migrated"] is False


def test_stage5_4_rows_are_training_only_and_not_actor_inputs_or_metrics() -> None:
    report = build_stage5_4_reward_report_integration()

    for row in report["surrogate_diagnostic_rows"]:
        assert row["training_only"] is True
        assert row["is_deployment_actor_input"] is False
        assert row["surrogate_outputs_are_metrics"] is False
        assert row["scalar_surrogate_reported"] is False
        assert row["latency_reference_s"] > 0.0
        assert row["energy_reference_j"] > 0.0

    assert report["metric_governance"]["new_metric_names_introduced"] == []
    assert report["metric_governance"]["surrogate_outputs_are_metrics"] is False


def test_stage5_4_constraint_violation_and_plateau_are_visible() -> None:
    report = build_stage5_4_reward_report_integration()
    rows = report["surrogate_diagnostic_rows"]

    violated = [row for row in rows if not row["constraint_satisfied"]]
    satisfied = [row for row in rows if row["constraint_satisfied"]]

    assert violated
    assert satisfied
    assert all(row["reliability_violation"] > 0.0 for row in violated)
    assert all(row["reliability_penalty"] > 0.0 for row in violated)
    assert all(row["reliability_violation"] == 0.0 for row in satisfied)
    assert all(row["reliability_penalty"] == 0.0 for row in satisfied)
    assert report["checks"]["constraint_violation_visible"] is True
    assert report["checks"]["constraint_plateau_visible"] is True


def test_stage5_4_normalization_references_are_applied() -> None:
    report = build_stage5_4_reward_report_integration()
    reference = report["normalization_reference"]
    rows = report["surrogate_diagnostic_rows"]

    assert reference["latency_reference_s"] == 0.0022698175688954207
    assert reference["energy_reference_j"] == 0.004134917967719052
    assert any(row["normalized_latency"] == 1.0 for row in rows)
    assert any(row["normalized_energy"] == 1.0 for row in rows)
    assert max(row["normalized_latency"] for row in rows) > 1.0
    assert max(row["normalized_energy"] for row in rows) > 1.0
    assert report["checks"]["normalization_reference_applied"] is True
