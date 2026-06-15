from marl_topology.training.critic_dataset import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    STAGE27_FROZEN_ACTOR_POLICY_ID,
    Stage27CriticDatasetConfig,
)

from stage27_synthetic import make_stage27_synthetic_dataset


def test_stage27_dataset_config_declares_fixed_critic_only_protocol() -> None:
    config = Stage27CriticDatasetConfig()
    payload = config.to_payload()

    assert config.train_transitions == 32 * 16 * 4
    assert config.eval_transitions == 16 * 16 * 4
    assert payload["actor_update_performed"] is False
    assert payload["policy_gradient_update_performed"] is False
    assert payload["train_transitions"] >= 512
    assert payload["eval_transitions"] >= 128


def test_critic_dataset_health_enforces_minimum_evidence_without_actor_update() -> None:
    dataset = make_stage27_synthetic_dataset()
    health = dataset.health_report()

    assert len(dataset.train_rows) == 512
    assert len(dataset.eval_rows) == 128
    assert dataset.actor_checksum_before == dataset.actor_checksum_after
    assert dataset.actor_update_performed is False
    assert dataset.policy_gradient_update_performed is False
    assert health["minimum_train_transition_gate"] is True
    assert health["minimum_eval_transition_gate"] is True
    assert health["return_variance"] > 0.0
    assert health["reward_variance"] > 0.0
    assert health["feature_health"]["all_finite"] is True
    assert health["sampler_id"] == ACTIVE_POLICY_GRADIENT_SAMPLER_ID
    assert health["actor_policy_id"] == STAGE27_FROZEN_ACTOR_POLICY_ID


def test_critic_dataset_batches_have_stable_feature_and_graph_shapes() -> None:
    dataset = make_stage27_synthetic_dataset()
    features = dataset.feature_batch("train")
    graph = dataset.graph_batch("eval")

    assert features.value_features.shape[0] == 512
    assert features.diagnostic_features.shape[0] == 512
    assert graph.node_features.shape == (128, 3, 8)
    assert graph.edge_features.shape == (128, 3, 8)
    assert graph.edge_index.shape == (128, 3, 2)
    assert graph.node_mask.all().item()
    assert graph.edge_mask.all().item()
