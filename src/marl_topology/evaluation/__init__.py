"""Evaluation helpers."""

from .baseline_report import (
    BaselineReportRow,
    build_stage2_baseline_evaluation_report,
    build_stage2_scenario_fixture_report,
)
from .calibration_fixture_suite import (
    STAGE5_0F_REQUIRED_FAMILIES,
    STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID,
    TauCalibrationFixtureRow,
    build_stage5_0f_tau_calibration_fixture_suite_report,
)
from .demo import build_demo_report, build_demo_stack
from .fixture_stack import EvaluationStack, build_fixture_stack
from .stage3_fixture_suite import (
    REQUIRED_STAGE3_MICRO_FIXTURES,
    build_stage3_micro_fixture_suite_review,
)
from .stage4_baseline_oracle_review import (
    STAGE4_5_REFERENCE_SCENARIO_ID,
    STAGE4_5_REVIEW_STAGE_ID,
    Stage45ReviewConfig,
    Stage45TopologyReviewRow,
    build_stage4_5_baseline_oracle_review,
)
from .stage4_application_report import (
    STAGE4_7_REPORT_STAGE_ID,
    Stage47EvaluationReportConfig,
    Stage47TopologySummary,
    build_stage4_7_pbft_application_evaluation_report,
)
from .stage4_boundary_audit import (
    STAGE4_8_BOUNDARY_AUDIT_STAGE_ID,
    STAGE4_8_REQUIRED_CASES,
    Stage48BoundaryAuditConfig,
    Stage48BoundaryAuditRow,
    build_stage4_8_boundary_audit_report,
)
from .tau_consensus_calibration_report import (
    STAGE5_0D_TAU_CALIBRATION_REPORT_STAGE_ID,
    TauCalibrationReportConfig,
    TauCandidate,
    build_tau_consensus_calibration_report,
)
from .requirement_feasibility_diagnosis import (
    STAGE5_0H_REQUIREMENT_DIAGNOSIS_STAGE_ID,
    SUPPORTED_FAILURE_REASONS,
    TAU_DIAGNOSTIC_VALUES,
    TAU_REQUIREMENT_MIN,
    TAU_STRESS_CANDIDATES,
    build_stage5_0h_requirement_feasibility_diagnosis,
)
from .feasibility_envelope_sweep_design import (
    REQUIRED_SWEEP_IDS,
    REQUIRED_SWEEP_ROW_FIELDS,
    STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
    build_stage5_0i_feasibility_envelope_sweep_design,
)
from .feasibility_envelope_sweep import (
    STAGE5_0J_MINIMAL_SWEEP_STAGE_ID,
    STAGE5_0J_SWEEP_IDS,
    Stage50jSweepRow,
    build_stage5_0j_minimal_feasibility_envelope_sweep,
)
from .feasibility_envelope_sweep_stage3_backed import (
    STAGE5_0K_STAGE3_BACKED_SWEEP_IDS,
    STAGE5_0K_STAGE3_BACKED_SWEEP_STAGE_ID,
    Stage50kSweepRow,
    build_stage5_0k_stage3_backed_feasibility_envelope_sweep,
)
from .feasibility_envelope_sweep_range_review import (
    STAGE5_0L_RANGE_REVIEW_STAGE_ID,
    STAGE5_0L_SWEEP_IDS,
    build_stage5_0l_stage3_backed_sweep_range_review,
)
from .objective_readiness_review import (
    STAGE5_0M_OBJECTIVE_READINESS_STAGE_ID,
    STAGE5_0M_RECOMMENDED_NEXT_TASK,
    build_stage5_0m_objective_readiness_review,
)
from .normalization_reference_selection import (
    STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID,
    STAGE5_3_RECOMMENDED_NEXT_TASK,
    build_stage5_3_normalization_reference_selection,
)
from .surrogate_diagnostics_report import (
    STAGE5_4_RECOMMENDED_NEXT_TASK,
    STAGE5_4_REWARD_REPORT_STAGE_ID,
    build_stage5_4_reward_report_integration,
)
from .training_preflight import (
    STAGE5_5_RECOMMENDED_NEXT_TASK,
    STAGE5_5_TRAINING_PREFLIGHT_STAGE_ID,
    STAGE5_5_VERDICT,
    build_stage5_5_training_preflight_review,
)
from .reward_surface_analysis import (
    ObjectiveSurfaceRecord,
    STAGE25_OBJECTIVE_ORDERING,
    STAGE25_REWARD_SURFACE_ANALYSIS_ID,
    build_reward_surface_analysis,
)

__all__ = [
    "BaselineReportRow",
    "EvaluationStack",
    "REQUIRED_STAGE3_MICRO_FIXTURES",
    "STAGE4_5_REFERENCE_SCENARIO_ID",
    "STAGE4_5_REVIEW_STAGE_ID",
    "STAGE4_7_REPORT_STAGE_ID",
    "STAGE4_8_BOUNDARY_AUDIT_STAGE_ID",
    "STAGE4_8_REQUIRED_CASES",
    "STAGE5_0D_TAU_CALIBRATION_REPORT_STAGE_ID",
    "STAGE5_0F_REQUIRED_FAMILIES",
    "STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID",
    "STAGE5_0H_REQUIREMENT_DIAGNOSIS_STAGE_ID",
    "STAGE5_0I_SWEEP_DESIGN_STAGE_ID",
    "STAGE5_0J_MINIMAL_SWEEP_STAGE_ID",
    "STAGE5_0J_SWEEP_IDS",
    "STAGE5_0K_STAGE3_BACKED_SWEEP_IDS",
    "STAGE5_0K_STAGE3_BACKED_SWEEP_STAGE_ID",
    "STAGE5_0L_RANGE_REVIEW_STAGE_ID",
    "STAGE5_0L_SWEEP_IDS",
    "STAGE5_0M_OBJECTIVE_READINESS_STAGE_ID",
    "STAGE5_0M_RECOMMENDED_NEXT_TASK",
    "STAGE5_3_NORMALIZATION_REFERENCE_STAGE_ID",
    "STAGE5_3_RECOMMENDED_NEXT_TASK",
    "STAGE5_4_RECOMMENDED_NEXT_TASK",
    "STAGE5_4_REWARD_REPORT_STAGE_ID",
    "STAGE5_5_RECOMMENDED_NEXT_TASK",
    "STAGE5_5_TRAINING_PREFLIGHT_STAGE_ID",
    "STAGE5_5_VERDICT",
    "STAGE25_OBJECTIVE_ORDERING",
    "STAGE25_REWARD_SURFACE_ANALYSIS_ID",
    "Stage50jSweepRow",
    "Stage50kSweepRow",
    "REQUIRED_SWEEP_IDS",
    "REQUIRED_SWEEP_ROW_FIELDS",
    "SUPPORTED_FAILURE_REASONS",
    "TAU_DIAGNOSTIC_VALUES",
    "TAU_REQUIREMENT_MIN",
    "TAU_STRESS_CANDIDATES",
    "TauCalibrationReportConfig",
    "TauCandidate",
    "TauCalibrationFixtureRow",
    "ObjectiveSurfaceRecord",
    "build_demo_report",
    "build_demo_stack",
    "build_fixture_stack",
    "build_stage2_baseline_evaluation_report",
    "build_stage2_scenario_fixture_report",
    "build_stage3_micro_fixture_suite_review",
    "build_stage4_5_baseline_oracle_review",
    "build_stage4_7_pbft_application_evaluation_report",
    "build_stage4_8_boundary_audit_report",
    "build_stage5_0f_tau_calibration_fixture_suite_report",
    "build_stage5_0h_requirement_feasibility_diagnosis",
    "build_stage5_0i_feasibility_envelope_sweep_design",
    "build_stage5_0j_minimal_feasibility_envelope_sweep",
    "build_stage5_0k_stage3_backed_feasibility_envelope_sweep",
    "build_stage5_0l_stage3_backed_sweep_range_review",
    "build_stage5_0m_objective_readiness_review",
    "build_stage5_3_normalization_reference_selection",
    "build_stage5_4_reward_report_integration",
    "build_stage5_5_training_preflight_review",
    "build_reward_surface_analysis",
    "build_tau_consensus_calibration_report",
    "Stage45ReviewConfig",
    "Stage45TopologyReviewRow",
    "Stage47EvaluationReportConfig",
    "Stage47TopologySummary",
    "Stage48BoundaryAuditConfig",
    "Stage48BoundaryAuditRow",
]
