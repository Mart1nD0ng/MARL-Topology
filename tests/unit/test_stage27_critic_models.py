import inspect

import torch

from marl_topology.models import (
    CentralizedMessagePassingGraphCritic,
    CentralizedMessagePassingGraphCriticConfig,
    EnrichedCentralizedMLPCritic,
)
from marl_topology.models.centralized_message_passing_graph_critic import GraphCriticBatch
from marl_topology.models.centralized_message_passing_graph_critic import (
    CentralizedMessagePassingGraphCritic as GraphCriticClass,
)
from marl_topology.models.enriched_centralized_mlp_critic import (
    EnrichedCentralizedMLPCritic as MLPCriticClass,
)
from marl_topology.training.critic_features import VALUE_CRITIC_FEATURE_FIELDS


def test_enriched_mlp_critic_forward_outputs_finite_normalized_values() -> None:
    torch.manual_seed(27)
    model = EnrichedCentralizedMLPCritic()
    features = torch.randn(6, len(VALUE_CRITIC_FEATURE_FIELDS))

    output = model(features)

    output.assert_shapes(6)
    assert torch.isfinite(output.normalized_value).all().item()
    assert model.boundary_report()["deployment_actor_receives_critic_output"] is False


def test_graph_critic_forward_outputs_finite_values() -> None:
    torch.manual_seed(27)
    model = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(hidden_dim=32)
    )
    batch = _graph_batch()

    output = model(batch)

    output.assert_shapes(1)
    assert torch.isfinite(output.normalized_value).all().item()
    assert model.boundary_report()["message_layers"] >= 2


def test_graph_critic_is_structure_sensitive() -> None:
    torch.manual_seed(27)
    model = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(hidden_dim=32)
    )
    original = _graph_batch()
    changed = GraphCriticBatch(
        node_features=original.node_features,
        edge_features=original.edge_features,
        edge_index=torch.tensor([[[0, 1], [0, 1], [0, 1]]], dtype=torch.long),
        node_mask=original.node_mask,
        edge_mask=original.edge_mask,
    )

    original_value = model(original).normalized_value
    changed_value = model(changed).normalized_value

    assert not torch.allclose(original_value, changed_value)


def test_graph_critic_is_permutation_invariant_for_node_order() -> None:
    torch.manual_seed(27)
    model = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(hidden_dim=32)
    )
    original = _graph_batch()
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(3)
    permuted_edges = inverse[original.edge_index]
    permuted = GraphCriticBatch(
        node_features=original.node_features[:, permutation],
        edge_features=original.edge_features,
        edge_index=permuted_edges,
        node_mask=original.node_mask[:, permutation],
        edge_mask=original.edge_mask,
    )

    original_value = model(original).normalized_value
    permuted_value = model(permuted).normalized_value

    assert torch.allclose(original_value, permuted_value, atol=1e-5)


def test_stage27_critic_models_do_not_define_transformer_recurrent_or_coma_classes() -> None:
    source = inspect.getsource(GraphCriticClass) + inspect.getsource(MLPCriticClass)

    forbidden = ("class Transformer", "class LSTM", "class GRU", "class COMA")
    assert all(pattern not in source for pattern in forbidden)


def _graph_batch() -> GraphCriticBatch:
    return GraphCriticBatch(
        node_features=torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.2, 1.1, 1.9, 0.5, 0.0],
                    [0.0, 1.0, 0.0, 0.4, 0.8, 1.2, 0.7, 0.0],
                    [0.0, 0.0, 1.0, 0.6, 0.2, 1.0, 0.3, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
        edge_features=torch.tensor(
            [
                [
                    [0.9, 0.9, 0.02, 0.1, 0.0, 10.0, 0.25, 0.5],
                    [0.7, 0.7, 0.04, 0.3, 0.0, 12.0, 0.5, 0.75],
                    [0.6, 0.6, 0.06, 0.4, 0.0, 15.0, 0.25, 0.75],
                ]
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[[0, 1], [1, 2], [0, 2]]], dtype=torch.long),
        node_mask=torch.tensor([[True, True, True]]),
        edge_mask=torch.tensor([[True, True, True]]),
    )
