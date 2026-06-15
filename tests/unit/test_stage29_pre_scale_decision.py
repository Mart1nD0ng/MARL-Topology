from pathlib import Path

from marl_topology.evaluation.stage29_pre_scale_decision import (
    STAGE29_RECOMMENDED_NEXT_TASK,
    build_stage29_pre_scale_decision_report,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage29_report_recommends_reward_projection_repair_not_larger_pilot() -> None:
    report = build_stage29_pre_scale_decision_report(project_root=ROOT)
    packet = report["decision_packet"]

    assert report["pass_gate"] is True
    assert report["verdict"] == "stage29_pass_pre_scale_decision_review_complete"
    assert packet["recommended_next_task"] == STAGE29_RECOMMENDED_NEXT_TASK
    assert packet["recommended_option"] == (
        "option_b_repair_reward_objective_and_projection_before_larger_pilot"
    )
    assert packet["larger_pilot_approved"] is False
    assert packet["scale_up_approved"] is False
    assert packet["owner_decision_required"] is True


def test_stage29_critic_is_ready_but_reward_projection_and_scale_are_not() -> None:
    report = build_stage29_pre_scale_decision_report(project_root=ROOT)
    rows = {row["component"]: row for row in report["root_cause_matrix"]}
    readiness = report["scale_readiness"]

    assert rows["critic_baseline"]["status"] == "PASS"
    assert rows["critic_baseline"]["blocks_larger_pilot"] is False
    assert rows["reward_objective_alignment"]["status"] == "FAIL"
    assert rows["reward_objective_alignment"]["blocks_larger_pilot"] is True
    assert rows["assembler_projection_alignment"]["status"] == "WARN"
    assert rows["assembler_projection_alignment"]["blocks_larger_pilot"] is True
    assert readiness["critic_ready"] is True
    assert readiness["reward_objective_ready"] is False
    assert readiness["projection_ready"] is False
    assert readiness["scale_up_ready"] is False


def test_stage29_evidence_keeps_stage28_mixed_result_visible() -> None:
    report = build_stage29_pre_scale_decision_report(project_root=ROOT)
    delta = report["stage28_evidence"]["eval_delta_mean"]
    critic = report["stage28_evidence"]["critic_health"]

    assert report["stage28_evidence"]["completed_seed_count"] == 3
    assert critic["mean_update_explained_variance"] > 0.10
    assert critic["mean_value_return_correlation"] > 0.30
    assert delta["latency_delta"] < 0
    assert delta["energy_delta"] < 0
    assert delta["mean_reward_surrogate_delta"] < 0
    assert delta["top_proposal_rejection_rate_delta"] > 0
    assert delta["tau_feasible_rate_delta"] >= -0.05
    assert delta["violation_rate_delta"] <= 0.05


def test_stage29_forbidden_action_flags_are_false() -> None:
    report = build_stage29_pre_scale_decision_report(project_root=ROOT)

    assert set(report["forbidden_action_flags"].values()) == {False}
