"""LocalAttentionGNNEdgeScorer: v3 message-passing GNN with GAT-style attention ego-message
aggregation (segment-softmax over each ego's incident edges) instead of the symmetric mean.
Drop-in [E] logits; the attention path genuinely differs from the mean once trained.
"""

import torch

from marl_topology.models.local_attention_gnn_edge_scorer import (
    LocalAttentionGNNEdgeScorer,
    _attention_by_group,
)
from marl_topology.models.local_gnn_edge_scorer import (
    LocalGNNV3ResidualNormConfig,
    LocalMessagePassingGNNV3ResidualNorm,
)
from marl_topology.models.tensorizers import (
    ACTOR_EDGE_FEATURE_FIELDS,
    ActorEdgeRecordRef,
    ActorEdgeTensorBatch,
)

F = len(ACTOR_EDGE_FEATURE_FIELDS)


def _batch(edge_features: torch.Tensor, group_ids: torch.Tensor) -> ActorEdgeTensorBatch:
    edge_count = edge_features.shape[0]
    records = tuple(
        ActorEdgeRecordRef(
            sample_index=0, agent_id="rsu0", neighbor_id=f"v{i}",
            edge_id=f"rsu0--v{i}", directed_edge_id=f"rsu0->v{i}", time_step=0,
        )
        for i in range(edge_count)
    )
    return ActorEdgeTensorBatch(
        edge_features=edge_features,
        edge_mask=torch.ones(edge_count, dtype=torch.bool),
        group_ids=group_ids,
        records=records,
    )


def test_attention_by_group_is_a_valid_segment_softmax() -> None:
    # groups: edges {0,1} -> group 0, edge {2} -> group 1.
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    attn = torch.tensor([[0.0], [0.0], [5.0]])  # equal within group 0 -> mean of edges 0,1
    group_ids = torch.tensor([0, 0, 1])
    out = _attention_by_group(values, attn, group_ids)
    # group 0 equal logits -> 0.5/0.5 weights -> mean = [0.5, 0.5], broadcast to edges 0,1.
    assert torch.allclose(out[0], torch.tensor([0.5, 0.5]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([0.5, 0.5]), atol=1e-6)
    # group 1 single edge -> weight 1 -> equals that edge's value.
    assert torch.allclose(out[2], torch.tensor([2.0, 2.0]), atol=1e-6)


def test_attention_weights_concentrate_on_high_logit_edge() -> None:
    values = torch.tensor([[1.0], [10.0]])
    attn = torch.tensor([[0.0], [100.0]])  # edge 1 dominates
    out = _attention_by_group(values, attn, torch.tensor([0, 0]))
    # near-hard attention on edge 1 -> aggregated ~= 10.0 (not the mean 5.5).
    assert out[0].item() > 9.9


def test_attention_actor_produces_edge_logits_and_handles_empty() -> None:
    actor = LocalAttentionGNNEdgeScorer().eval()
    edge_count = 5
    feats = torch.randn(edge_count, F)
    out = actor.score_tensor_batch(_batch(feats, torch.zeros(edge_count, dtype=torch.long)))
    assert out.shape == (edge_count,)
    # empty ego-graph -> [0] logits (forward returns early before aggregation).
    empty = actor.forward_with_groups(torch.zeros(0, F), torch.zeros(0, dtype=torch.long))
    assert empty.shape == (0,)


def test_attention_differs_from_mean_after_training_signal() -> None:
    torch.manual_seed(0)
    actor = LocalAttentionGNNEdgeScorer().eval()
    plain = LocalMessagePassingGNNV3ResidualNorm(LocalGNNV3ResidualNormConfig()).eval()
    # share the v3 backbone so the ONLY difference is mean-vs-attention aggregation.
    plain.load_state_dict({k: v for k, v in actor.state_dict().items() if k in plain.state_dict()})
    feats = torch.randn(6, F)
    group_ids = torch.zeros(6, dtype=torch.long)
    batch = _batch(feats, group_ids)
    with torch.no_grad():
        # a training step would move the attention scorers off near-uniform -> attention
        # aggregation diverges from the mean.
        for scorer in actor.attention_scorers:
            scorer.weight.add_(torch.randn_like(scorer.weight) * 1.0)
        attn_out = actor.score_tensor_batch(batch)
        mean_out = plain.score_tensor_batch(batch)
    assert attn_out.shape == (6,)
    assert not torch.allclose(attn_out, mean_out, atol=1e-4)


def test_attention_actor_reports_boundary() -> None:
    report = LocalAttentionGNNEdgeScorer().boundary_report()
    assert report["ego_message_aggregation"] == "gat_segment_softmax_attention"
    assert report["outputs_edge_scores_only"] is True
