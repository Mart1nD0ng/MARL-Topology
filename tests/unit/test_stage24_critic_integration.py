from functools import lru_cache

from marl_topology.training.mappo.trainer import (
    STAGE24_FAIL_VERDICT,
    run_stage24_critic_integrated_micro_loop,
)


@lru_cache(maxsize=1)
def _smoke_report() -> dict[str, object]:
    return run_stage24_critic_integrated_micro_loop(mode="smoke", project_root=".")


def test_stage24_smoke_loop_uses_centralized_critic_for_advantages() -> None:
    report = _smoke_report()

    # Budget fix (rsu 4->8, src/marl_topology/budgets.py) consequence: under the
    # corrected budget the budget-aware endpoint sampler wins the stage24 A/B, so the
    # live winner != the registered active sampler (Plackett-Luce) and the loop
    # correctly returns blocked_awaiting_owner_decision. Owner decision (2026-06):
    # KEEP Plackett-Luce active -- a route-B A/B validation
    # (logs/stage34_sampler_validation.py) showed the endpoint sampler over-selects
    # (7 edges vs the feasible 5) and tanks feasibility under interference; the real
    # fix is a variable proposal size, deferred. The centralized-critic integration
    # this test verifies holds for BOTH samplers regardless.
    assert report["verdict"] == STAGE24_FAIL_VERDICT
    assert report["pass_gate"] is False
    assert report["owner_decision_required"] is True
    for sampler_report in report["sampler_reports"].values():
        integration = sampler_report["critic_integration"]
        assert integration["centralized_critic_used_for_values"] is True
        assert integration["critic_values_used_in_advantage"] is True
        assert integration["critic_value_loss_optimized"] is True
        assert integration["actor_received_gradient"] is True
        assert integration["critic_received_gradient"] is True
        assert sampler_report["update_count"] > 0


def test_stage24_smoke_loop_records_required_training_diagnostics() -> None:
    report = _smoke_report()
    winner = report["sampler_selection"]["selected_sampler_id"]
    updates = report["sampler_reports"][winner]["updates"]
    first = updates[0]

    required = {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "ratio_mean",
        "ratio_max",
        "advantage_mean",
        "advantage_std",
        "grad_norm",
        "value_prediction_mean",
        "return_mean",
    }
    assert required <= set(first)
    assert report["reward_config_unchanged"] is True
    assert report["checkpoint_written"] is False
    assert report["artifact_written"] is False
