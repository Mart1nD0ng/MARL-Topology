import pytest

from marl_topology.data import (
    DEPLOYMENT_ACTOR_INPUT_COLUMNS,
    REWARD_TRAINING_ONLY_COLUMNS,
    ReplayColumnViolation,
    classify_replay_columns,
    project_actor_input_row,
    validate_deployment_actor_input_columns,
    validate_registered_replay_columns,
)
from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.objectives import (
    SURROGATE_SIGNAL_INPUT_METRICS,
    SURROGATE_TRAINING_COLUMNS,
    SurrogateSignalConfig,
    SurrogateSignalInput,
    evaluate_reward_surrogate,
)


def _config(
    *,
    reliability_weight: float = 100.0,
    latency_weight: float = 1.0,
    energy_weight: float = 1.0,
) -> SurrogateSignalConfig:
    return SurrogateSignalConfig(
        tau=0.9,
        reliability_weight=reliability_weight,
        latency_weight=latency_weight,
        energy_weight=energy_weight,
        latency_reference_s=0.2,
        energy_reference_j=2.0,
    )


def _input(
    *,
    consensus_success_probability: float,
    latency: float = 0.1,
    energy: float = 1.0,
) -> SurrogateSignalInput:
    return SurrogateSignalInput(
        consensus_success_probability=consensus_success_probability,
        latency=latency,
        energy=energy,
        topology_diagnostics={"fixture": "unit"},
    )


def test_surrogate_uses_only_registered_metric_inputs() -> None:
    require_registered_metrics(SURROGATE_SIGNAL_INPUT_METRICS)

    assert set(SURROGATE_SIGNAL_INPUT_METRICS) == {
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    }
    assert "reward_surrogate" not in REGISTERED_METRICS


def test_reliability_plateau_above_tau_has_no_bonus() -> None:
    config = _config()
    at_tau = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.9),
        config,
    )
    above_tau = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.99),
        config,
    )

    assert at_tau.reliability_penalty == 0.0
    assert above_tau.reliability_penalty == 0.0
    assert at_tau.training_signal_value == above_tau.training_signal_value
    assert above_tau.constraint_satisfied is True


def test_unreliable_low_cost_topology_is_penalized_below_feasible_case() -> None:
    config = _config(reliability_weight=1000.0)
    unreliable_low_cost = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.5, latency=0.0, energy=0.0),
        config,
    )
    feasible_higher_cost = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.95, latency=0.2, energy=2.0),
        config,
    )

    assert unreliable_low_cost.constraint_satisfied is False
    assert feasible_higher_cost.constraint_satisfied is True
    assert unreliable_low_cost.training_signal_value < feasible_higher_cost.training_signal_value


def test_lower_latency_and_energy_improve_signal_when_reliability_matches() -> None:
    config = _config()
    slow = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.95, latency=0.2, energy=1.0),
        config,
    )
    fast = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.95, latency=0.1, energy=1.0),
        config,
    )
    high_energy = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.95, latency=0.1, energy=2.0),
        config,
    )
    low_energy = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.95, latency=0.1, energy=1.0),
        config,
    )

    assert fast.training_signal_value > slow.training_signal_value
    assert low_energy.training_signal_value > high_energy.training_signal_value


def test_config_and_inputs_fail_fast_on_invalid_values() -> None:
    with pytest.raises(ValueError, match="latency_reference_s"):
        SurrogateSignalConfig(
            tau=0.9,
            reliability_weight=1.0,
            latency_weight=1.0,
            energy_weight=1.0,
            latency_reference_s=0.0,
            energy_reference_j=1.0,
        )
    with pytest.raises(ValueError, match="energy_reference_j"):
        SurrogateSignalConfig(
            tau=0.9,
            reliability_weight=1.0,
            latency_weight=1.0,
            energy_weight=1.0,
            latency_reference_s=1.0,
            energy_reference_j=-1.0,
        )
    with pytest.raises(ValueError, match="consensus_success_probability"):
        SurrogateSignalInput(
            consensus_success_probability=1.1,
            latency=0.0,
            energy=0.0,
        )
    with pytest.raises(ValueError, match="latency"):
        SurrogateSignalInput(
            consensus_success_probability=0.9,
            latency=-1.0,
            energy=0.0,
        )


def test_payload_columns_are_training_only_not_actor_inputs() -> None:
    payload = evaluate_reward_surrogate(
        _input(consensus_success_probability=0.95),
        _config(),
    ).to_training_payload()

    assert tuple(payload) == SURROGATE_TRAINING_COLUMNS
    assert set(payload) == REWARD_TRAINING_ONLY_COLUMNS
    validate_registered_replay_columns(payload.keys())

    classification = classify_replay_columns(payload.keys())
    assert set(classification["reward_training_only"]) == set(payload)
    assert not classification["deployment_actor_input"]

    with pytest.raises(ReplayColumnViolation):
        validate_deployment_actor_input_columns(payload.keys())

    mixed_row = {column: None for column in DEPLOYMENT_ACTOR_INPUT_COLUMNS}
    mixed_row.update(payload)
    actor_row = project_actor_input_row(mixed_row)
    assert set(actor_row) == DEPLOYMENT_ACTOR_INPUT_COLUMNS
    assert not (set(actor_row) & set(payload))
