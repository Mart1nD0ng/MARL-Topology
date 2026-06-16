"""Unit tests for the K-hop local message-passing actor (Task 1 trunk finalisation).

The headline test is `test_khop_logit_depends_on_k_hop_neighbour`: it demonstrates
empirically that the receptive field grows one hop per round, so a K=2 actor reasons
about a 2-hop backbone that a K=1 (ego-graph v3-class) actor structurally cannot see.
The locality tests prove the growth is bounded -- no global-state shortcut.
"""

import pytest
import torch

from marl_topology.models import (
    KHOP_GNN_FORBIDDEN_INPUT_FIELDS,
    LocalKHopGNNEdgeScorer,
    LocalKHopGNNEdgeScorerConfig,
)


def _path_graph(num_nodes: int, *, node_dim: int, edge_dim: int, seed: int = 0):
    """A directed path 0<->1<->2<->...<->(num_nodes-1) with random features."""

    generator = torch.Generator().manual_seed(seed)
    node_features = torch.randn(num_nodes, node_dim, generator=generator)
    pairs = []
    for i in range(num_nodes - 1):
        pairs.append((i, i + 1))
        pairs.append((i + 1, i))
    edge_index = torch.tensor(pairs, dtype=torch.long)
    edge_features = torch.randn(edge_index.shape[0], edge_dim, generator=generator)
    return node_features, edge_index, edge_features


def _edge_row(edge_index: torch.Tensor, src: int, dst: int) -> int:
    for row in range(edge_index.shape[0]):
        if int(edge_index[row, 0]) == src and int(edge_index[row, 1]) == dst:
            return row
    raise AssertionError(f"directed edge ({src},{dst}) not found")


def _build(rounds: int, **overrides) -> LocalKHopGNNEdgeScorer:
    torch.manual_seed(1234)
    config = LocalKHopGNNEdgeScorerConfig(rounds=rounds, **overrides)
    model = LocalKHopGNNEdgeScorer(config)
    model.eval()
    return model


def test_khop_logit_depends_on_k_hop_neighbour() -> None:
    # Path a(0)-b(1)-c(2)-d(3). The logit on directed edge a->b has its nearest endpoint
    # b at hop distance 2 from d. So a K=1 actor cannot see d (logit unchanged when d is
    # perturbed), but a K=2 actor can (logit changes). This is the multi-hop capability.
    node_dim, edge_dim = 7, 11
    node_features, edge_index, edge_features = _path_graph(4, node_dim=node_dim, edge_dim=edge_dim)
    ab_row = _edge_row(edge_index, 0, 1)

    perturbed = node_features.clone()
    perturbed[3] += 3.0  # perturb node d only

    model_k1 = _build(1)
    base_k1 = model_k1.forward_graph(node_features, edge_index, edge_features)
    pert_k1 = model_k1.forward_graph(perturbed, edge_index, edge_features)
    assert torch.allclose(base_k1[ab_row], pert_k1[ab_row], atol=1e-6), (
        "K=1 receptive field must NOT reach a 2-hop node (ego-graph behaviour)"
    )

    model_k2 = _build(2)
    base_k2 = model_k2.forward_graph(node_features, edge_index, edge_features)
    pert_k2 = model_k2.forward_graph(perturbed, edge_index, edge_features)
    assert not torch.allclose(base_k2[ab_row], pert_k2[ab_row], atol=1e-6), (
        "K=2 receptive field MUST reach a 2-hop node (multi-hop backbone reasoning)"
    )


def test_khop_one_hop_neighbour_is_seen_at_k1() -> None:
    # Contrast: directed edge b->c has endpoint c at hop distance 1 from d, so even K=1
    # sees a perturbation of d.
    node_features, edge_index, edge_features = _path_graph(4, node_dim=7, edge_dim=11)
    bc_row = _edge_row(edge_index, 1, 2)
    perturbed = node_features.clone()
    perturbed[3] += 3.0

    model_k1 = _build(1)
    base = model_k1.forward_graph(node_features, edge_index, edge_features)
    pert = model_k1.forward_graph(perturbed, edge_index, edge_features)
    assert not torch.allclose(base[bc_row], pert[bc_row], atol=1e-6)


def test_khop_strict_batch_locality_no_cross_scene_leak() -> None:
    # Two disconnected components in one tensor: {0,1} and {2,3}. Perturbing a node in
    # component 2 must NOT change any logit in component 1 (no global-state shortcut).
    torch.manual_seed(7)
    node_features = torch.randn(4, 7)
    edge_index = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long)
    edge_features = torch.randn(4, 11)
    node_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    model = _build(3)  # deep enough that any leak would show
    base = model.forward_graph(node_features, edge_index, edge_features, node_batch=node_batch)

    perturbed = node_features.clone()
    perturbed[2] += 5.0  # component-2 node
    pert = model.forward_graph(perturbed, edge_index, edge_features, node_batch=node_batch)

    # Component-1 edges are rows 0 and 1.
    assert torch.allclose(base[:2], pert[:2], atol=1e-6), "disconnected component leaked"
    # Sanity: component-2 edges DID change.
    assert not torch.allclose(base[2:], pert[2:], atol=1e-6)


def test_khop_permutation_equivariant() -> None:
    torch.manual_seed(11)
    node_features = torch.randn(5, 7)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 4], [4, 3]], dtype=torch.long
    )
    edge_features = torch.randn(edge_index.shape[0], 11)
    model = _build(3)
    logits = model.forward_graph(node_features, edge_index, edge_features)

    # Relabel node i -> slot perm[i]; remap edge_index; keep edge order/features fixed.
    perm = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
    permuted_nodes = torch.empty_like(node_features)
    permuted_nodes[perm] = node_features
    permuted_edge_index = perm[edge_index]
    permuted_logits = model.forward_graph(permuted_nodes, permuted_edge_index, edge_features)

    assert torch.allclose(logits, permuted_logits, atol=1e-5)


def test_khop_directed_head_breaks_symmetry() -> None:
    # u->v and v->u must be able to differ (directed head): essential for mutual acceptance.
    torch.manual_seed(3)
    node_features = torch.randn(2, 7)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # Give the two directions DIFFERENT edge features (kind flags swap in real data).
    edge_features = torch.randn(2, 11)
    model = _build(2)
    logits = model.forward_graph(node_features, edge_index, edge_features)
    assert not torch.allclose(logits[0], logits[1], atol=1e-6)


def test_khop_boundary_report_is_full_message_passing() -> None:
    model = _build(3)
    report = model.boundary_report()
    assert report["full_message_passing_gnn"] is True
    assert report["uses_edge_to_node_messages"] is True
    assert report["uses_node_to_edge_updates"] is True
    assert report["directed_edge_head"] is True
    assert report["residual_connections"] is True
    assert report["normalization"] == "layer_norm_per_message_passing_layer"
    assert report["global_topology_used"] is False
    assert report["critic_outputs_used"] is False
    assert report["receptive_field_hops"] == 3


def test_khop_rejects_forbidden_actor_input_fields() -> None:
    assert "global_topology" in KHOP_GNN_FORBIDDEN_INPUT_FIELDS
    with pytest.raises(ValueError):
        LocalKHopGNNEdgeScorer.validate_input_field_names(("edge_id", "global_topology"))
    # A clean field set must pass.
    LocalKHopGNNEdgeScorer.validate_input_field_names(("distance_3d_m", "link_success_probability"))


def test_khop_anti_oversmoothing_preserves_logit_spread_at_depth() -> None:
    # A moderately connected random graph at K=5 must not collapse every edge to the same
    # logit (residual + per-node LayerNorm + jumping-knowledge guard against oversmoothing).
    torch.manual_seed(5)
    num_nodes = 12
    node_features = torch.randn(num_nodes, 7)
    pairs = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and torch.rand(1).item() < 0.35:
                pairs.append((i, j))
    edge_index = torch.tensor(pairs, dtype=torch.long)
    edge_features = torch.randn(edge_index.shape[0], 11)
    model = _build(5, jumping_knowledge="concat")
    logits = model.forward_graph(node_features, edge_index, edge_features).detach()
    assert torch.isfinite(logits).all()
    assert float(logits.max() - logits.min()) > 1e-3, "edge logits collapsed (oversmoothing)"


def test_khop_jumping_knowledge_modes_run() -> None:
    node_features, edge_index, edge_features = _path_graph(5, node_dim=7, edge_dim=11)
    for mode in ("last", "concat", "max"):
        model = _build(3, jumping_knowledge=mode)
        logits = model.forward_graph(node_features, edge_index, edge_features)
        assert logits.shape == (edge_index.shape[0],)
        assert torch.isfinite(logits).all()


def test_khop_empty_graph_returns_empty() -> None:
    model = _build(3)
    node_features = torch.randn(2, 7)
    edge_index = torch.empty((0, 2), dtype=torch.long)
    edge_features = torch.empty((0, 11), dtype=torch.float32)
    logits = model.forward_graph(node_features, edge_index, edge_features)
    assert logits.shape == (0,)


def test_khop_config_validates() -> None:
    with pytest.raises(ValueError):
        LocalKHopGNNEdgeScorerConfig(rounds=0)
    with pytest.raises(ValueError):
        LocalKHopGNNEdgeScorerConfig(jumping_knowledge="bogus")
    with pytest.raises(ValueError):
        LocalKHopGNNEdgeScorerConfig(dropout_probability=1.0)
