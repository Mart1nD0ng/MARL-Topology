from functools import lru_cache

from marl_topology.evaluation.stage21_fair_evaluation import (
    Stage21FairEvaluationConfig,
    run_stage21_objective_stack_aligned_assembler_aware_rerun,
)


@lru_cache(maxsize=1)
def _stage21_smoke_report():
    return run_stage21_objective_stack_aligned_assembler_aware_rerun(
        config=Stage21FairEvaluationConfig(
            supervised_epochs=1,
            run_policy_gradient_if_gates_pass=True,
        )
    )


def test_stage21_supervised_rerun_uses_mlp_and_gnn_only() -> None:
    report = _stage21_smoke_report()
    supervised = report["supervised_rerun_report"]

    assert supervised["models_rerun"] == ["MLP", "GNN"]
    assert supervised["gru_lstm_rerun"] is False
    assert supervised["no_actor_leakage"] is True
    assert supervised["checkpoint_written"] is False
    assert supervised["artifact_written"] is False


def test_stage21_fair_evaluation_uses_projected_baselines_for_main_comparison() -> None:
    report = _stage21_smoke_report()
    evaluation = report["fair_evaluation_report"]
    baseline = evaluation["baseline_summaries"]

    assert evaluation["main_comparison_uses_projected_baselines"] is True
    assert evaluation["raw_baselines_are_diagnostic_only"] is True
    assert evaluation["same_assembler_used_for_actors_and_projected_baselines"] is True
    assert "full_graph_projected" in baseline
    assert "greedy_reliability_projected" in baseline
    assert "random_projected" in baseline
    assert "full_graph_raw" in baseline


def test_stage21_policy_gradient_runs_when_all_hard_gates_pass() -> None:
    # Corrected-budget (rsu 4->8, src/marl_topology/budgets.py) reality: the RSU
    # consensus hub (degree 5-7) survives projection instead of being truncated at
    # degree 4, so the actor genuinely reaches feasibility. best_actor tau rises
    # 0.5143 (36/70) -> 0.7571 (53/70), beating full_graph_projected (0.7143) AND
    # crossing the frozen Stage20 baseline 0.5593 -> every hard gate passes and the
    # PG pilot is allowed to run. (Not a vacuous all-feasible regime: the actor
    # beats full_graph_projected at BOTH budgets.)
    report = _stage21_smoke_report()
    gates = report["stage21_gates"]

    assert all(gate["passed"] for gate in gates.values())
    assert report["stage21_passed"] is True
    assert report["policy_gradient_pilot_ran"] is True
    assert report["failure_review"] is None


def test_stage21_policy_gradient_is_skipped_when_a_hard_gate_fails(monkeypatch) -> None:
    # Gating-logic coverage (budget-independent): since every gate now passes under
    # the corrected budget, force the actor-performance gate to fail by raising the
    # frozen Stage20 baseline above any achievable tau, and assert the PG skip
    # branch still fires. This preserves the conditional-gating coverage the
    # corrected-budget run no longer exercises naturally.
    import marl_topology.evaluation.stage21_fair_evaluation as stage21_fair_evaluation

    monkeypatch.setattr(
        stage21_fair_evaluation, "STAGE20_BEST_ACTOR_TAU_FEASIBLE_RATE", 0.99
    )
    report = run_stage21_objective_stack_aligned_assembler_aware_rerun(
        config=Stage21FairEvaluationConfig(
            supervised_epochs=1,
            run_policy_gradient_if_gates_pass=True,
        )
    )
    gates = report["stage21_gates"]

    assert any(not gate["passed"] for gate in gates.values())
    assert report["stage21_passed"] is False
    assert report["policy_gradient_pilot_ran"] is False
    assert report["failure_review"]["policy_gradient_skipped"] is True
    assert report["failure_review"]["failed_gates"] == [
        "actor_performance_sufficient_for_pilot"
    ]


def test_stage21_boundaries_remain_closed_in_smoke_when_gate_fails() -> None:
    report = _stage21_smoke_report()

    assert report["coma_allowed"] is False
    assert report["transformer_allowed"] is False
    assert report["reward_weight_tuning_performed"] is False
    assert report["final_tau_selected"] is False
    assert report["checkpoint_written"] is False
    assert report["artifact_written"] is False
    assert report["v5_modified"] is False
