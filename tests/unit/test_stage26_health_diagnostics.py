from pathlib import Path

import pytest

from marl_topology.evaluation.stage26_health_diagnostics import (
    COMPONENT_ORDER,
    STAGE26_RECOMMENDED_NEXT_TASK,
    STAGE26_RECOMMENDED_OPTION,
    build_stage26_full_system_health_report,
)


ROOT = Path(__file__).resolve().parents[2]
ROOT_CAUSE_COMPONENTS = {
    "data_health",
    "communication_health",
    "consensus_health",
    "reward_objective_health",
    "assembler_health",
    "sampler_health",
    "actor_health",
    "critic_health",
    "mappo_loop_health",
    "harness_state_health",
}


@pytest.fixture(scope="module")
def stage26_report() -> dict[str, object]:
    return build_stage26_full_system_health_report(project_root=ROOT)


def test_stage26_report_passes_diagnostic_gate_without_approving_scale(
    stage26_report: dict[str, object],
) -> None:
    assert stage26_report["pass_gate"] is True
    assert stage26_report["pass_issues"] == []
    assert stage26_report["verdict"] == "stage26_pass_full_system_health_diagnostic_complete"
    assert stage26_report["recommended_option"] == STAGE26_RECOMMENDED_OPTION
    assert stage26_report["recommended_next_task"] == STAGE26_RECOMMENDED_NEXT_TASK
    assert stage26_report["owner_decision_required"] is True

    components = stage26_report["component_health"]
    assert list(components) == list(COMPONENT_ORDER)
    assert components["harness_state_health"]["status"] == "PASS"
    assert components["communication_health"]["status"] == "PASS"
    assert components["consensus_health"]["status"] == "PASS"
    assert components["visualization_health"]["status"] == "PASS"
    assert components["scale_readiness"]["status"] == "FAIL"
    assert components["scale_readiness"]["metrics"]["scale_up_allowed"] is False


def test_stage26_identifies_critic_as_primary_limiter(
    stage26_report: dict[str, object],
) -> None:
    critic = stage26_report["component_health"]["critic_health"]
    loop = stage26_report["component_health"]["mappo_loop_health"]
    packet = stage26_report["decision_packet"]

    assert critic["status"] == "FAIL"
    assert "critic_unused_or_weak" in critic["diagnosis_labels"]
    assert critic["metrics"]["explained_variance"]["mean"] < 0.05
    assert critic["metrics"]["value_return_correlation"] < 0.0
    assert "critic_not_helpful" in loop["diagnosis_labels"]
    assert packet["critical_fail_components"] == ["critic_health"]
    assert packet["recommended_option"] == "option_b_repair_critic_before_more_training"


def test_stage26_secondary_risks_are_evidence_backed(
    stage26_report: dict[str, object],
) -> None:
    components = stage26_report["component_health"]

    assert components["data_health"]["status"] == "WARN"
    assert "data_too_small" in components["data_health"]["diagnosis_labels"]
    assert components["data_health"]["metrics"]["num_unique_source_contexts"] == 10
    assert components["reward_objective_health"]["status"] == "WARN"
    assert "reward_objective_mismatch" in components["reward_objective_health"]["diagnosis_labels"]
    assert components["assembler_health"]["status"] == "WARN"
    assert components["sampler_health"]["status"] == "WARN"
    assert components["actor_health"]["status"] == "WARN"

    matrix = stage26_report["root_cause_matrix"]
    matrix_components = {row["component"] for row in matrix}
    assert ROOT_CAUSE_COMPONENTS == matrix_components
    assert all(row["recommended_repair"] for row in matrix)
