from pathlib import Path

from marl_topology.evaluation.stage26_health_diagnostics import (
    COMPONENT_ORDER,
    build_stage26_full_system_health_report,
    component_rows,
    metrics_summary_rows,
    root_cause_rows,
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


REQUIRED_METRIC_KEYS = {
    "harness_state_health": {
        "project_state_stage25_complete",
        "stage25_manifest_valid",
        "stage26_manifest_valid",
        "no_dynamic_torch_import",
        "no_forbidden_source_patterns",
    },
    "data_health": {
        "num_train_scenarios",
        "num_eval_scenarios",
        "num_unique_source_contexts",
        "feasible_ratio_train",
        "feasible_ratio_eval",
        "near_threshold_ratio",
        "duplicate_context_rate",
        "actor_signature_contradiction_rate",
        "edge_target_high_mid_low_distribution",
        "edge_delta_positive_negative_balance",
        "temporal_sequence_count",
    },
    "communication_health": {
        "link_success_probability",
        "sinr_db",
        "deadline_delivery_probability",
        "required_reliability_met_rate",
        "required_transmission_time_capped_rate",
        "expected_attempts",
        "scheduled_latency",
        "successful_latency",
        "energy",
        "p2p_latency_energy_correlation",
        "reliability_latency_energy_coupling_sanity",
    },
    "consensus_health": {
        "consensus_success_probability",
        "tau_feasible_rate",
        "violation_rate",
        "per_primary_reliability_variance",
        "min_primary_reliability",
        "max_primary_reliability",
        "primary_spread",
        "phase_pre_prepare_delivery",
        "phase_prepare_delivery",
        "phase_commit_delivery",
        "quorum_tail_sensitivity",
    },
    "reward_objective_health": {
        "reward",
        "reward_component_means_std",
        "reward_variance",
        "reward_feasible_vs_infeasible_gap",
        "reward_latency_correlation",
        "reward_energy_correlation",
        "reward_consensus_correlation",
        "objective_rank_reward_rank_spearman",
        "dominated_topology_reward_error_count",
        "empty_graph_reward_rank",
        "full_graph_reward_rank",
        "sparse_feasible_reward_rank",
    },
    "assembler_health": {
        "top_proposal_rejection_rate",
        "above_threshold_rejection_rate",
        "rejection_reason_distribution",
        "high_score_rejected_count",
        "accepted_low_score_count",
        "selected_edge_count",
        "empty_graph_rate",
        "full_graph_rate",
        "tx_budget_exceeded_rate",
        "rx_capacity_exceeded_rate",
        "conflict_rejection_rate",
        "projection_limit_rate",
    },
    "sampler_health": {
        "proposal_count",
        "sampler_logprob",
        "entropy",
        "unique_proposal_rate",
        "proposal_overlap_rate",
        "selected_from_proposed_rate",
        "candidate_mask_violation_count",
        "top_k_capacity_binding_rate",
    },
    "actor_health": {
        "score_mean_std_min_max",
        "logit_mean_std",
        "accepted_score_mean",
        "rejected_score_mean",
        "accepted_rejected_score_gap",
        "score_target_correlation",
        "score_teacher_rank_correlation",
        "actor_parameter_delta",
        "edge_score_entropy_proxy",
        "selected_edge_count_shift",
        "empty_full_graph_tendency",
        "actor_feature_missingness",
    },
    "critic_health": {
        "value_loss",
        "explained_variance",
        "value_prediction",
        "return",
        "value_return_correlation",
        "value_bias",
        "advantage_std",
        "advantage_std_with_critic",
        "critic_grad_norm",
        "critic_parameter_delta",
    },
    "mappo_loop_health": {
        "approx_kl",
        "clip_fraction",
        "entropy_trend",
        "policy_loss",
        "value_loss",
        "actor_grad_norm",
        "critic_grad_norm",
        "ratio_mean",
        "ratio_max",
        "train_eval_gap",
        "seed_success_count",
        "seed_stop_reason_count",
        "update_effect_size",
    },
}


def test_stage26_component_metrics_cover_required_health_surfaces() -> None:
    report = build_stage26_full_system_health_report(project_root=ROOT)
    components = report["component_health"]

    assert set(COMPONENT_ORDER) == set(components)
    for component_id, metric_keys in REQUIRED_METRIC_KEYS.items():
        missing = metric_keys - set(components[component_id]["metrics"])
        assert not missing, f"{component_id} missing metrics: {sorted(missing)}"


def test_stage26_report_tables_cover_scorecard_root_cause_and_metric_summary() -> None:
    report = build_stage26_full_system_health_report(project_root=ROOT)

    scorecard = component_rows(report)
    matrix = root_cause_rows(report)
    metrics = metrics_summary_rows(report)

    assert len(scorecard) == len(COMPONENT_ORDER)
    assert len(matrix) == len(ROOT_CAUSE_COMPONENTS)
    assert {row["component"] for row in scorecard} == set(COMPONENT_ORDER)
    assert {row["component"] for row in matrix} == ROOT_CAUSE_COMPONENTS
    assert len(metrics) >= 100


def test_stage26_project_state_expected_closeout_terms_are_declared_after_report() -> None:
    report = build_stage26_full_system_health_report(project_root=ROOT)
    state_text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    assert report["decision_packet"]["scale_up_approved"] is False
    assert report["decision_packet"]["owner_decision_required"] is True
    assert report["decision_packet"]["recommended_next_task"] == (
        "stage_27_critic_baseline_repair_before_more_training"
    )

    required_terms = [
        "current_stage: post_stage_26_complete_full_system_health_diagnostic_awaiting_owner_decision",
        "stage_26_full_system_health_diagnostic",
        "stage26_critic_health_gate",
        "recommended_next_task: stage_27_critic_baseline_repair_before_more_training",
        "Scale-up training remains blocked",
        "LSTM/recurrent PPO remains blocked",
    ]
    missing = [term for term in required_terms if term not in state_text]
    assert not missing, f"PROJECT_STATE missing Stage 26 closeout terms: {missing}"
