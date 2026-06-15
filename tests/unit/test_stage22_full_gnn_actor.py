import torch

from marl_topology.models import LocalGNNEdgeScorer, LocalGNNEdgeScorerConfig
from marl_topology.models.local_gnn_edge_scorer import FULL_GNN_FORBIDDEN_INPUT_FIELDS


def test_full_gnn_output_is_permutation_invariant_within_local_group() -> None:
    torch.manual_seed(22)
    model = LocalGNNEdgeScorer(LocalGNNEdgeScorerConfig(input_dim=4, hidden_dim=8))
    model.eval()
    features = torch.randn(3, 4)
    groups = torch.zeros(3, dtype=torch.long)

    logits = model.forward_with_groups(features, groups)
    order = torch.tensor([2, 0, 1])
    reordered = model.forward_with_groups(features[order], groups)

    restored = torch.empty_like(reordered)
    restored[order] = reordered
    assert torch.allclose(logits, restored, atol=1e-6)


def test_full_gnn_message_passing_changes_group_context() -> None:
    torch.manual_seed(23)
    model = LocalGNNEdgeScorer(LocalGNNEdgeScorerConfig(input_dim=4, hidden_dim=8))
    feature = torch.randn(1, 4)
    neighbor = torch.randn(1, 4)

    isolated = model.forward_with_groups(feature, torch.tensor([0]))
    grouped = model.forward_with_groups(torch.cat((feature, neighbor)), torch.tensor([0, 0]))[0:1]

    assert not torch.allclose(isolated, grouped)


def test_full_gnn_edge_features_node_features_and_mask_affect_scores() -> None:
    torch.manual_seed(24)
    model = LocalGNNEdgeScorer(LocalGNNEdgeScorerConfig(input_dim=4, hidden_dim=8))
    features = torch.zeros(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    features[0, 0, 0] = 1.0
    features[0, 1, 1] = 1.0
    features[1, 0, 2] = 1.0

    logits = model.forward_padded(features, mask)
    changed = features.clone()
    changed[0, 0, 0] = 2.0
    changed_logits = model.forward_padded(changed, mask)

    assert logits.shape == mask.shape
    assert logits[0, 2].item() == 0.0
    assert not torch.allclose(logits[mask], changed_logits[mask])
    assert model.boundary_report()["message_passing_layers"] >= 2
    assert model.boundary_report()["full_message_passing_gnn"] is True


def test_full_gnn_rejects_forbidden_actor_input_fields() -> None:
    assert "global_topology" in FULL_GNN_FORBIDDEN_INPUT_FIELDS
    try:
        LocalGNNEdgeScorer.validate_input_field_names(
            ("agent_kind_is_vehicle", "global_topology")
        )
    except ValueError as exc:
        assert "forbidden GNN actor input fields" in str(exc)
    else:
        raise AssertionError("forbidden global actor field was accepted")
