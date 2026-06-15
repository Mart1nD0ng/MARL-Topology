import pytest

from marl_topology.data.stage21_objective_stack_evidence import STAGE21_TAU_REQUIREMENT_MIN
from marl_topology.training.production_mappo_adapter import (
    Stage33GNNStabilityConfig,
    Stage33LearningRateScheduler,
    Stage33ProductionMappoViolation,
    stage33_fixed_gnn_stability_configs,
)


def test_stage33_fixed_protocol_has_only_three_allowed_configs() -> None:
    configs = stage33_fixed_gnn_stability_configs()

    assert [config.config_id for config in configs] == [
        "low_lr_no_warmup",
        "low_lr_with_warmup",
        "low_lr_with_warmup_stronger_grad_clip",
    ]
    assert all(config.actor_lr <= 1e-3 for config in configs)
    assert all(config.max_grad_norm <= 0.5 for config in configs)
    assert all(len(config.seeds) >= 5 for config in configs)
    assert all(config.tau_requirement_min == STAGE21_TAU_REQUIREMENT_MIN for config in configs)


def test_stage33_warmup_scheduler_records_lr_and_ramps_then_decays() -> None:
    config = Stage33GNNStabilityConfig(
        config_id="low_lr_with_warmup",
    )
    scheduler = Stage33LearningRateScheduler(config)

    assert config.warmup_updates >= 1
    assert scheduler.factor(1) <= 1.0
    assert scheduler.factor(config.warmup_updates) == pytest.approx(1.0)
    assert scheduler.factor(config.total_opt_steps) >= 0.1


def test_stage33_config_rejects_reward_or_loop_boundary_changes() -> None:
    with pytest.raises(Stage33ProductionMappoViolation):
        Stage33GNNStabilityConfig(actor_lr=0.01)
    with pytest.raises(Stage33ProductionMappoViolation):
        Stage33GNNStabilityConfig(max_grad_norm=0.75)
    with pytest.raises(Stage33ProductionMappoViolation):
        Stage33GNNStabilityConfig(tau_requirement_min=0.89)
    with pytest.raises(Stage33ProductionMappoViolation):
        Stage33GNNStabilityConfig(eval_scenarios=1)

    payload = Stage33GNNStabilityConfig().to_payload()
    assert payload["reward_weights_tuned"] is False
    assert payload["tau_requirement_min"] == STAGE21_TAU_REQUIREMENT_MIN
