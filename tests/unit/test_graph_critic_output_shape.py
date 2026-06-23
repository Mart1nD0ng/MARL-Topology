"""Phase 7: the centralized graph value critic produces a scalar V(scene) per graph.

A B-scene batch -> V of shape [B] (scalar head) in reward units, finite, with gradients to the
critic's own params; vector_heads=True -> [B,3]; variable-N / padding handled by masks (a masked
node's features cannot change V).
"""

import torch

from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic


def _batch(B=2, N=4, E=5, dn=8, de=8, seed=0):
    torch.manual_seed(seed)
    nf = torch.randn(B, N, dn)
    ef = torch.randn(B, E, de)
    ei = torch.randint(0, N, (B, E, 2))
    node_mask = torch.ones(B, N)
    node_mask[0, 3] = 0.0   # scene 0 has only 3 real nodes
    edge_mask = torch.ones(B, E)
    edge_mask[0, 4] = 0.0   # scene 0 has only 4 real edges
    return nf, ef, ei, node_mask, edge_mask


def test_scalar_value_shape_finite_and_grad():
    nf, ef, ei, nm, em = _batch()
    critic = CentralizedGraphCritic(node_dim=8, edge_dim=8, hidden=16, rounds=2)
    value = critic(nf, ef, ei, nm, em)
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert all(p.grad is not None for p in critic.parameters() if p.requires_grad)


def test_vector_heads_shape():
    nf, ef, ei, nm, em = _batch()
    critic = CentralizedGraphCritic(node_dim=8, edge_dim=8, hidden=16, rounds=2, vector_heads=True)
    value = critic(nf, ef, ei, nm, em)
    assert value.shape == (2, 3)


def test_masked_padding_does_not_change_value():
    nf, ef, ei, nm, em = _batch()
    critic = CentralizedGraphCritic(node_dim=8, edge_dim=8, hidden=16, rounds=2).eval()
    with torch.no_grad():
        v1 = critic(nf, ef, ei, nm, em)
        nf2 = nf.clone()
        nf2[0, 3] = 99.0  # change a MASKED node's features
        ef2 = ef.clone()
        ef2[0, 4] = 99.0  # change a MASKED edge's features
        v2 = critic(nf2, ef2, ei, nm, em)
    assert torch.allclose(v1, v2, atol=1e-5)
