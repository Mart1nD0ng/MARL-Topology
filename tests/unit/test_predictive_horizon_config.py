"""Predictive-horizon config knob (opt-in): default off (0), serialized in the payload,
non-negative guard. The collector behavior + leakage/PPO safety are covered by the
logs/verify_predictive_horizon.py smoke; here we pin the config contract.
"""

import pytest

from marl_topology.training.production_mappo_adapter import (
    Stage33GNNStabilityConfig,
    Stage33ProductionMappoViolation,
)


def test_predictive_horizon_defaults_to_zero_myopic() -> None:
    cfg = Stage33GNNStabilityConfig()
    assert cfg.predictive_horizon == 0


def test_predictive_horizon_in_payload() -> None:
    cfg = Stage33GNNStabilityConfig(trajectory_mode=True, predictive_horizon=3)
    payload = cfg.to_payload()
    assert payload["predictive_horizon"] == 3
    # the frozen reward contract stays intact -- the horizon changes WHICH frame is scored,
    # not the reward function.
    assert payload["reward_weights_tuned"] is False
    assert payload["tau_requirement_min"] == cfg.tau_requirement_min


def test_predictive_horizon_rejects_negative() -> None:
    with pytest.raises(Stage33ProductionMappoViolation):
        Stage33GNNStabilityConfig(predictive_horizon=-1)
