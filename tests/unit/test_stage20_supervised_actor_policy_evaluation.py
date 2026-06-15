from functools import lru_cache

from marl_topology.evaluation.stage20_supervised_actor_policy_evaluation import (
    Stage20PolicyEvaluationConfig,
    run_stage20_supervised_actor_policy_evaluation,
)


@lru_cache(maxsize=1)
def _stage20_smoke_report() -> dict[str, object]:
    return run_stage20_supervised_actor_policy_evaluation(
        config=Stage20PolicyEvaluationConfig(
            stage19_epochs=5,
            stage19_temporal_epochs=5,
        )
    )


def test_stage20_evaluates_actor_scores_through_environment_side_assembler() -> None:
    report = _stage20_smoke_report()

    assert report["environment_side_topology_assembler_used"] is True
    assert report["assembler_type"] == "ConflictAwareGreedyAssembler"
    assert report["actor_outputs_final_topology"] is False
    assert report["actor_outputs_edge_scores_only"] is True
    assert report["assembler_uses_oracle_reward_objective_consensus"] is False


def test_stage20_reports_full_row_and_temporal_subset_metrics() -> None:
    report = _stage20_smoke_report()
    summaries = report["model_summaries"]

    assert summaries["MLP"]["row_count"] == 59
    assert summaries["GNN"]["row_count"] == 59
    assert summaries["GRU"]["row_count"] == 18
    assert summaries["LSTM"]["row_count"] == 18
    assert summaries["MLP"]["nonempty_topology_rate"] > 0.0
    assert summaries["GNN"]["projection_rejection_rate"] > 0.0
    assert summaries["GRU"]["tau_feasible_rate"] >= 0.0
    assert summaries["LSTM"]["tau_feasible_rate"] >= 0.0


def test_stage20_keeps_baselines_and_oracle_diagnostic_separate() -> None:
    report = _stage20_smoke_report()
    baseline = report["baseline_summary"]

    assert baseline["full_graph"]["is_oracle"] is False
    assert baseline["greedy_reliability"]["is_oracle"] is False
    assert baseline["source_topology"]["is_oracle"] is False
    assert baseline["oracle_diagnostic"]["is_oracle"] is True
    assert report["baseline_full_graph_is_oracle"] is False
    assert report["oracle_diagnostic_used_for_actor_input"] is False


def test_stage20_policy_gradient_readiness_stays_blocked_when_actor_lags_greedy() -> None:
    report = _stage20_smoke_report()
    best = report["best_all_row_actor"]
    best_summary = report["model_summaries"][best]
    greedy = report["baseline_summary"]["greedy_reliability"]

    assert best in {"MLP", "GNN"}
    assert best_summary["tau_feasible_rate"] < greedy["tau_feasible_rate"]
    assert report["stage21_policy_gradient_readiness_gate_passed"] is False
    assert report["recommended_next_task"] == "stage_21_assembler_aware_supervised_target_refinement"


def test_stage20_boundaries_remain_closed() -> None:
    report = _stage20_smoke_report()

    assert report["policy_gradient_allowed"] is False
    assert report["ppo_mappo_allowed"] is False
    assert report["coma_allowed"] is False
    assert report["transformer_allowed"] is False
    assert report["scale_up_training_allowed"] is False
    assert report["critic_training_performed"] is False
    assert report["checkpoint_written"] is False
    assert report["artifact_written"] is False
    assert report["reward_weight_tuning_performed"] is False
    assert report["final_tau_selected"] is False
    assert report["v5_modified"] is False
