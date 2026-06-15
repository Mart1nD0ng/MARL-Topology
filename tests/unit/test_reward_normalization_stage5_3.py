import pytest

from marl_topology.evaluation import (
    STAGE5_0L_RANGE_REVIEW_STAGE_ID,
    STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID,
    build_stage5_0l_stage3_backed_sweep_range_review,
    build_stage5_3_normalization_reference_selection,
)
from marl_topology.objectives import (
    NORMALIZATION_SELECTION_POLICY,
    NormalizationReferenceConfig,
    SurrogateSignalInput,
    evaluate_reward_surrogate,
    select_normalization_references,
)


def test_stage5_3_selects_fixed_references_from_stage5_0l_feasible_rows() -> None:
    source_report = build_stage5_0l_stage3_backed_sweep_range_review()
    references = select_normalization_references(
        source_report["sweep_rows"],
        NormalizationReferenceConfig(source_stage_id=STAGE5_0L_RANGE_REVIEW_STAGE_ID),
    )
    feasible_rows = [
        row
        for row in source_report["sweep_rows"]
        if row["consensus_success_probability"] >= 0.9
        and row["latency"] > 0.0
        and row["energy"] > 0.0
    ]

    assert references.selection_policy == NORMALIZATION_SELECTION_POLICY
    assert references.source_stage_id == STAGE5_0L_RANGE_REVIEW_STAGE_ID
    assert references.eligible_row_count == len(feasible_rows) == 4
    assert references.excluded_row_count == 4
    assert references.latency_reference_s == max(row["latency"] for row in feasible_rows)
    assert references.energy_reference_j == max(row["energy"] for row in feasible_rows)
    assert references.reward_weight_calibration_performed is False
    assert references.training_ready is False


def test_stage5_3_references_build_surrogate_config_without_calibrating_weights() -> None:
    references = select_normalization_references(
        build_stage5_0l_stage3_backed_sweep_range_review()["sweep_rows"],
        NormalizationReferenceConfig(source_stage_id=STAGE5_0L_RANGE_REVIEW_STAGE_ID),
    )
    config = references.build_surrogate_config(
        tau=0.9,
        reliability_weight=100.0,
        latency_weight=1.0,
        energy_weight=1.0,
        clip_max=2.0,
    )
    record = evaluate_reward_surrogate(
        SurrogateSignalInput(
            consensus_success_probability=0.95,
            latency=references.latency_reference_s,
            energy=references.energy_reference_j,
            topology_diagnostics={"source_stage": STAGE5_0L_RANGE_REVIEW_STAGE_ID},
        ),
        config,
    )

    assert config.latency_reference_s == references.latency_reference_s
    assert config.energy_reference_j == references.energy_reference_j
    assert record.normalized_latency == 1.0
    assert record.normalized_energy == 1.0
    assert record.reliability_penalty == 0.0


def test_stage5_3_selection_rejects_empty_or_ineligible_rows() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        select_normalization_references(())

    with pytest.raises(ValueError, match="not enough eligible rows"):
        select_normalization_references(
            (
                {
                    "consensus_success_probability": 0.5,
                    "latency": 0.001,
                    "energy": 0.001,
                },
            )
        )


def test_stage5_3_report_keeps_training_and_weight_calibration_blocked() -> None:
    report = build_stage5_3_normalization_reference_selection()
    reference = report["normalization_reference"]
    checks = report["checks"]

    assert report["stage"] == STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID
    assert report["source_stage"] == STAGE5_0L_RANGE_REVIEW_STAGE_ID
    assert report["tau_requirement_min"] == 0.9
    assert reference["latency_reference_s"] > 0.0
    assert reference["energy_reference_j"] > 0.0
    assert checks["source_stage3_backed"] is True
    assert checks["source_uses_finite_blocklength"] is True
    assert checks["reward_weight_calibration_performed"] is False
    assert checks["training_ready"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False
    assert report["metric_governance"]["new_metric_names_introduced"] == []
