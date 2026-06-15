"""Part B: LocalTemporalGNNEdgeScorer -- GRU temporal encoder over edge history fused
into the v3 message-passing GNN. Drop-in [E] logits; static fallback (history None)
== plain v3; zero-init residual fusion so it STARTS byte-identical to v3 even with
history, and only earns temporal influence through training.
"""

import torch

from marl_topology.models.local_gnn_edge_scorer import (
    LocalGNNV3ResidualNormConfig,
    LocalMessagePassingGNNV3ResidualNorm,
)
from marl_topology.models.local_temporal_gnn_edge_scorer import LocalTemporalGNNEdgeScorer
from marl_topology.models.tensorizers import (
    ACTOR_EDGE_FEATURE_FIELDS,
    ActorEdgeRecordRef,
    ActorEdgeTensorBatch,
)

F = len(ACTOR_EDGE_FEATURE_FIELDS)


def _batch(edge_features: torch.Tensor, history: torch.Tensor | None = None) -> ActorEdgeTensorBatch:
    edge_count = edge_features.shape[0]
    records = tuple(
        ActorEdgeRecordRef(
            sample_index=0,
            agent_id="rsu0",
            neighbor_id=f"v{i}",
            edge_id=f"rsu0--v{i}",
            directed_edge_id=f"rsu0->v{i}",
            time_step=0,
        )
        for i in range(edge_count)
    )
    return ActorEdgeTensorBatch(
        edge_features=edge_features,
        edge_mask=torch.ones(edge_count, dtype=torch.bool),
        group_ids=torch.zeros(edge_count, dtype=torch.long),
        records=records,
        history=history,
    )


def test_temporal_actor_starts_byte_identical_to_v3() -> None:
    torch.manual_seed(0)
    actor = LocalTemporalGNNEdgeScorer().eval()
    edge_count, window = 5, 4
    feats = torch.randn(edge_count, F)
    history = torch.randn(edge_count, window, F)
    with torch.no_grad():
        static = actor.score_tensor_batch(_batch(feats, history=None))
        temporal = actor.score_tensor_batch(_batch(feats, history=history))
    assert static.shape == (edge_count,)
    assert temporal.shape == (edge_count,)
    # zero-init residual fusion -> the temporal path starts byte-identical to static.
    assert torch.allclose(static, temporal, atol=1e-6)


def test_static_fallback_equals_plain_v3() -> None:
    torch.manual_seed(0)
    actor = LocalTemporalGNNEdgeScorer().eval()
    plain = LocalMessagePassingGNNV3ResidualNorm(LocalGNNV3ResidualNormConfig()).eval()
    # share the v3 weights so the comparison isolates architecture, not random init.
    plain.load_state_dict({k: v for k, v in actor.state_dict().items() if k in plain.state_dict()})
    batch = _batch(torch.randn(6, F), history=None)
    with torch.no_grad():
        assert torch.allclose(
            actor.score_tensor_batch(batch), plain.score_tensor_batch(batch), atol=1e-6
        )


def test_temporal_path_engages_after_training_signal() -> None:
    torch.manual_seed(0)
    actor = LocalTemporalGNNEdgeScorer().eval()
    edge_count, window = 5, 4
    feats = torch.randn(edge_count, F)
    history = torch.randn(edge_count, window, F)
    with torch.no_grad():
        # a training step would move the fusion off zero -> the temporal path influences.
        actor.temporal_fusion.weight.add_(torch.randn_like(actor.temporal_fusion.weight) * 0.5)
        static = actor.score_tensor_batch(_batch(feats, history=None))
        temporal = actor.score_tensor_batch(_batch(feats, history=history))
    assert not torch.allclose(static, temporal, atol=1e-5)


def test_temporal_actor_handles_empty_edges() -> None:
    actor = LocalTemporalGNNEdgeScorer().eval()
    out = actor.forward_with_history(
        torch.zeros(0, 3, F), torch.zeros(0, F), torch.zeros(0, dtype=torch.long)
    )
    assert out.shape == (0,)


def test_temporal_actor_reports_temporal_boundary() -> None:
    report = LocalTemporalGNNEdgeScorer().boundary_report()
    assert report["temporal_history_consumed"] is True
    assert report["outputs_edge_scores_only"] is True
    assert report["full_message_passing_gnn"] is True
