from pathlib import Path

import pytest

from marl_topology.training.mappo.stage25_pilot import (
    STAGE25_BASE_CONFIG_ID,
    Stage25PilotBaseConfig,
    Stage25PilotViolation,
    run_stage25_preflight,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage25_base_training_protocol_is_fixed_and_valid() -> None:
    cfg = Stage25PilotBaseConfig()
    payload = cfg.to_payload()

    assert payload["config_id"] == STAGE25_BASE_CONFIG_ID
    assert payload["train_scenarios"] == 16
    assert payload["eval_scenarios"] == 8
    assert payload["seeds"] == [2501, 2502, 2503]
    assert payload["rollout_steps"] == 16
    assert payload["transitions_per_update"] == 256
    assert cfg.transitions_per_update == cfg.train_scenarios * cfg.rollout_steps
    assert payload["minibatch_size"] == 64
    assert cfg.transitions_per_update // cfg.minibatch_size >= 2
    assert payload["update_epochs"] == 4
    assert payload["eval_every"] <= payload["max_updates"]
    assert payload["gamma"] == 0.99
    assert payload["gae_lambda"] == 0.95
    assert payload["clip_eps"] == 0.2
    assert payload["actor_lr"] == 1e-4
    assert payload["critic_lr"] == 1e-4
    assert payload["entropy_coef"] == 0.01
    assert payload["value_coef"] == 0.5
    assert payload["max_grad_norm"] == 0.5
    assert payload["base_config_fixed_not_tuned"] is True
    assert payload["diagnostic_probe"] is False


def test_stage25_base_training_protocol_rejects_downgrade_or_tuning_shape() -> None:
    invalid_cases = [
        {"train_scenarios": 8},
        {"minibatch_size": 512},
        {"minibatch_size": 256},
        {"update_epochs": 1},
        {"eval_every": 25},
        {"seeds": (2501,)},
        {"tau_requirement_min": 0.85},
    ]

    for overrides in invalid_cases:
        with pytest.raises(Stage25PilotViolation):
            Stage25PilotBaseConfig(**overrides)


def test_stage25_preflight_confirms_stage24_sampler_reward_and_manifest() -> None:
    preflight = run_stage25_preflight(project_root=ROOT)
    gates = preflight["gates"]

    assert preflight["preflight_passed"] is True
    assert gates["stage24_complete_and_preflight_passed"]["passed"] is True
    assert gates["active_sampler_is_plackett_luce"]["passed"] is True
    assert gates["base_config_fixed_protocol_valid"]["passed"] is True
    assert gates["train_eval_split_available"]["passed"] is True
    assert gates["reward_weights_unchanged"]["passed"] is True
    assert gates["reward_weights_unchanged"]["weight_tuning_performed"] is False
    assert gates["run_manifest_validator_available"]["passed"] is True
