import pytest
import torch

from marl_topology.models import (
    ACTIVE_STAGE33_GNN_MODEL_ID,
    LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID,
    LocalGNNV3ResidualNormConfig,
    LocalMessagePassingGNNV3ResidualNorm,
    LocalRoleResourceAwareGNNV3,
)
from marl_topology.models.local_gnn_edge_scorer import LocalGNNEdgeScorer


def _features(model) -> torch.Tensor:
    torch.manual_seed(33)
    return torch.randn(6, model.config.input_dim)


def test_stage33_v3_residual_norm_boundary_is_full_message_passing() -> None:
    model = LocalMessagePassingGNNV3ResidualNorm(LocalGNNV3ResidualNormConfig())
    report = model.boundary_report()

    assert report["model_id"] == ACTIVE_STAGE33_GNN_MODEL_ID
    assert report["full_message_passing_gnn"] is True
    assert report["uses_edge_to_node_messages"] is True
    assert report["uses_node_to_edge_updates"] is True
    assert report["residual_connections"] is True
    assert report["normalization"] == "layer_norm_per_message_passing_layer"
    assert report["global_topology_used"] is False
    assert report["critic_outputs_used"] is False


def test_stage33_v3_outputs_change_when_local_graph_structure_changes() -> None:
    model = LocalMessagePassingGNNV3ResidualNorm()
    model.eval()
    features = _features(model)
    grouped = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    separated = torch.arange(features.shape[0], dtype=torch.long)

    grouped_logits = model.forward_with_groups(features, grouped)
    separated_logits = model.forward_with_groups(features, separated)

    assert grouped_logits.shape == (features.shape[0],)
    assert not torch.allclose(grouped_logits, separated_logits)


def test_stage33_v3_node_and_edge_features_affect_output() -> None:
    model = LocalMessagePassingGNNV3ResidualNorm()
    model.eval()
    features = _features(model)
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    baseline = model.forward_with_groups(features, group_ids)

    changed = features.clone()
    changed[0, 0] += 2.0
    changed[1, -1] -= 2.0
    assert not torch.allclose(baseline, model.forward_with_groups(changed, group_ids))


def test_stage33_role_resource_variant_uses_role_and_resource_columns() -> None:
    model = LocalRoleResourceAwareGNNV3()
    model.eval()
    features = _features(model)
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    baseline = model.forward_with_groups(features, group_ids)

    changed = features.clone()
    changed[:, 0] = 1.0 - changed[:, 0]
    changed[:, -1] += 1.5

    assert model.boundary_report()["model_id"] == LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID
    assert model.boundary_report()["role_features_encoded"] is True
    assert model.boundary_report()["resource_context_features_encoded"] is True
    assert not torch.allclose(baseline, model.forward_with_groups(changed, group_ids))


def test_stage33_v3_is_permutation_invariant_within_grouped_graphs() -> None:
    model = LocalMessagePassingGNNV3ResidualNorm()
    model.eval()
    features = _features(model)
    group_ids = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.long)
    logits = model.forward_with_groups(features, group_ids)

    permutation = torch.tensor([2, 3, 4, 0, 1, 5], dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted_logits = model.forward_with_groups(features[permutation], group_ids[permutation])

    assert torch.allclose(logits, permuted_logits[inverse], atol=1e-6)


def test_stage33_v3_mask_support_and_forbidden_field_guard() -> None:
    model = LocalMessagePassingGNNV3ResidualNorm()
    model.eval()
    features = _features(model).reshape(2, 3, model.config.input_dim)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    logits = model.forward_padded(features, mask)

    assert logits.shape == mask.shape
    assert torch.equal(logits[~mask], torch.zeros_like(logits[~mask]))
    with pytest.raises(ValueError):
        LocalGNNEdgeScorer.validate_input_field_names(("edge_id", "global_topology"))
