"""Stage 5.3 normalization-reference selection sensor."""

from __future__ import annotations

from marl_topology.objectives import (
    NormalizationReferenceConfig,
    select_normalization_references,
)

from .feasibility_envelope_sweep_range_review import (
    STAGE5_0L_RANGE_REVIEW_STAGE_ID,
    build_stage5_0l_stage3_backed_sweep_range_review,
)
from .requirement_feasibility_diagnosis import TAU_REQUIREMENT_MIN


STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID = (
    "stage_5_3_reward_normalization_reference_selection"
)
STAGE5_3_RECOMMENDED_NEXT_TASK = "stage_5_4_reward_report_integration_without_training"


def build_stage5_3_normalization_reference_selection() -> dict[str, object]:
    """Build the Stage 5.3 fixed-reference selection report."""

    source_report = build_stage5_0l_stage3_backed_sweep_range_review()
    references = select_normalization_references(
        source_report["sweep_rows"],
        NormalizationReferenceConfig(
            tau_requirement_min=TAU_REQUIREMENT_MIN,
            source_stage_id=STAGE5_0L_RANGE_REVIEW_STAGE_ID,
        ),
    )
    payload = references.to_payload()
    return {
        "stage": STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID,
        "source_stage": STAGE5_0L_RANGE_REVIEW_STAGE_ID,
        "source_scope": "stage5_0l_stage3_backed_range_review_rows",
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "normalization_reference": payload,
        "surrogate_config_preview": {
            "tau": TAU_REQUIREMENT_MIN,
            "latency_reference_s": references.latency_reference_s,
            "energy_reference_j": references.energy_reference_j,
            "weights_selected": False,
            "training_ready": False,
        },
        "metric_governance": {
            "new_metric_names_introduced": [],
            "metric_inputs": [
                "consensus_success_probability",
                "latency",
                "energy",
                "topology_diagnostics",
            ],
            "surrogate_outputs_are_metrics": False,
        },
        "checks": {
            "source_stage3_backed": source_report["checks"]["all_rows_stage3_backed"],
            "source_uses_finite_blocklength": source_report["checks"][
                "all_rows_use_finite_blocklength"
            ],
            "tau_requirement_min_fixed": payload["tau_requirement_min"] == 0.9,
            "latency_reference_positive": payload["latency_reference_s"] > 0.0,
            "energy_reference_positive": payload["energy_reference_j"] > 0.0,
            "eligible_rows_present": payload["eligible_row_count"] > 0,
            "excluded_rows_visible": payload["excluded_row_count"] > 0,
            "selection_policy_fixed": (
                payload["selection_policy"] == "feasible_positive_max_v1"
            ),
            "reward_weight_calibration_performed": payload[
                "reward_weight_calibration_performed"
            ],
            "training_ready": payload["training_ready"],
            "training_run": False,
            "model_code_added": False,
            "v5_code_migrated": False,
            "recommended_next_task": STAGE5_3_RECOMMENDED_NEXT_TASK,
        },
    }
