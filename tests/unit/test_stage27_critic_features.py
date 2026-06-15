import pytest
import torch

from marl_topology.training.critic_features import (
    CriticFeatureViolation,
    CriticValueFeatureRecord,
    DIAGNOSTIC_FEATURE_FIELDS,
    VALUE_CRITIC_FEATURE_FIELDS,
    VALUE_FORBIDDEN_CURRENT_FIELDS,
    build_feature_batch,
    value_feature_health,
)

from stage27_synthetic import make_stage27_synthetic_dataset


@pytest.fixture(scope="module")
def synthetic_dataset():
    return make_stage27_synthetic_dataset()


def test_value_feature_records_are_finite_stable_and_pre_action(synthetic_dataset) -> None:
    row = synthetic_dataset.train_rows[0]
    record = row.value_features

    assert len(record.values) == len(VALUE_CRITIC_FEATURE_FIELDS)
    assert record.role == "mappo_value_pre_action"
    assert record.action_independent is True
    assert record.actor_input_allowed is False
    assert set(record.metadata).isdisjoint(VALUE_FORBIDDEN_CURRENT_FIELDS)
    assert torch.isfinite(torch.tensor(record.values)).all().item()


def test_diagnostic_records_are_action_conditioned_and_tagged(synthetic_dataset) -> None:
    row = synthetic_dataset.train_rows[0]
    diagnostic = row.diagnostic_features

    assert len(diagnostic.values) == len(DIAGNOSTIC_FEATURE_FIELDS)
    assert diagnostic.role == "action_conditioned_diagnostic"
    assert diagnostic.action_independent is False
    assert diagnostic.actor_input_allowed is False
    assert "current_reward_surrogate" in diagnostic.metadata
    assert "current_consensus_success_probability" in diagnostic.metadata


def test_feature_batch_and_health_report_are_finite(synthetic_dataset) -> None:
    rows = synthetic_dataset.train_rows[:8]
    batch = build_feature_batch(
        (row.value_features for row in rows),
        (row.diagnostic_features for row in rows),
    )
    health = value_feature_health(row.value_features for row in rows)

    assert batch.batch_size == 8
    assert batch.value_features.shape == (8, len(VALUE_CRITIC_FEATURE_FIELDS))
    assert batch.diagnostic_features.shape == (8, len(DIAGNOSTIC_FEATURE_FIELDS))
    assert health["all_finite"] is True
    assert health["actor_input_allowed"] is False


def test_value_feature_record_rejects_post_action_current_fields() -> None:
    values = tuple(0.0 for _ in VALUE_CRITIC_FEATURE_FIELDS)

    with pytest.raises(CriticFeatureViolation):
        CriticValueFeatureRecord(
            values=values,
            metadata={"current_reward_surrogate": -1.0},
        )
