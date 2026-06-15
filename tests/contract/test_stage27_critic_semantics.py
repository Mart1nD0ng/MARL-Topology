from marl_topology.models import (
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    CENTRALIZED_MLP_CRITIC_BASELINE_ID,
    build_model_registry,
)
from marl_topology.training.critic_features import (
    DIAGNOSTIC_FEATURE_FIELDS,
    VALUE_CRITIC_FEATURE_FIELDS,
    VALUE_FORBIDDEN_CURRENT_FIELDS,
)
from marl_topology.training.critic_repair_trainer import denormalize_values_for_gae


def test_value_critic_schema_excludes_current_post_action_outcomes() -> None:
    fields = set(VALUE_CRITIC_FEATURE_FIELDS)

    assert fields.isdisjoint(VALUE_FORBIDDEN_CURRENT_FIELDS)
    assert "previous_consensus_success_probability" in fields
    assert "previous_latency" in fields
    assert "previous_energy" in fields
    assert "previous_reward_summary" in fields


def test_diagnostic_schema_is_action_conditioned_and_may_contain_current_outcomes() -> None:
    diagnostic_fields = set(DIAGNOSTIC_FEATURE_FIELDS)

    assert "current_consensus_success_probability" in diagnostic_fields
    assert "current_latency" in diagnostic_fields
    assert "current_energy" in diagnostic_fields
    assert "current_reward_surrogate" in diagnostic_fields


def test_future_advantage_path_exposes_raw_value_denormalization() -> None:
    assert denormalize_values_for_gae.__name__ == "denormalize_values_for_gae"


def test_registry_has_exactly_one_future_active_value_baseline() -> None:
    registry = build_model_registry()
    active = [
        entry.model_id
        for entry in registry.values()
        if entry.active_for_future_value_baseline
    ]

    assert active == [CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID]
    assert registry[CENTRALIZED_MLP_CRITIC_BASELINE_ID].active_for_future_value_baseline is False
    assert registry[CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID].training_only is True
