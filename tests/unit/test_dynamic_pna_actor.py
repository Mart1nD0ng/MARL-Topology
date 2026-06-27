"""D12: the dynamic PNA actor -- PNA aggregation + cross-frame hidden + preference conditioning, a
decentralized drop-in for DynamicRecurrentActor (Plan §14)."""

from __future__ import annotations

import torch


def _actor(node_dim=5, edge_dim=3, hidden=8):
    from marl_topology.models.dynamic_pna_actor import DynamicPNAActor
    torch.manual_seed(0)
    return DynamicPNAActor(node_dim, edge_dim, hidden=hidden, delta=1.0)


def _graph(n=4, node_dim=5, edge_dim=3):
    nf = torch.randn(n, node_dim)
    ei = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    ef = torch.randn(ei.shape[0], edge_dim)
    return nf, ef, ei


def test_dynamic_pna_actor_signature_compatible() -> None:
    # drop-in for DynamicRecurrentActor: forward(nf, ef, ei, hidden=None) -> (logits[E], h[N,H]).
    from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor
    nd, ed, H = 5, 3, 8
    pna = _actor(nd, ed, H)
    nf, ef, ei = _graph(4, nd, ed)
    logits, h = pna(nf, ef, ei, hidden=None)
    assert logits.shape == (ei.shape[0],) and h.shape == (4, H)
    mlp = DynamicRecurrentActor(nd, ed, hidden=H)
    ml, mh = mlp(nf, ef, ei, hidden=None)
    assert ml.shape == logits.shape and mh.shape == h.shape       # identical I/O contract
    assert hasattr(pna, "model_id") and hasattr(pna, "gru")       # the trunk/tests rely on these
    # softly bounded logits (the BCSP sampler can't overflow)
    assert torch.isfinite(logits).all() and float(logits.abs().max()) <= 10.0 + 1e-4


def test_dynamic_pna_cross_frame_hidden_changes_output() -> None:
    # the carried per-node hidden (cross-frame memory) changes the logits -> recurrence is real.
    pna = _actor()
    nf, ef, ei = _graph()
    l_none, h = pna(nf, ef, ei, hidden=None)
    l_carry, _ = pna(nf, ef, ei, hidden=torch.randn(4, pna.hidden))
    assert not torch.allclose(l_none, l_carry)


def test_dynamic_pna_preference_changes_output() -> None:
    # the deployment preference omega = (omega_E, omega_L) is an actor input -> it changes the logits
    # (a single policy can sweep the energy-latency trade-off; omega is a PUBLIC param, deploy-legal).
    pna = _actor()
    nf, ef, ei = _graph()
    pna.set_preference(1.0, 0.0)
    l1, _ = pna(nf, ef, ei, hidden=None)
    pna.set_preference(0.0, 1.0)
    l2, _ = pna(nf, ef, ei, hidden=None)
    assert not torch.allclose(l1, l2)
    pna.set_preference(0.0, 0.0)                                   # omega=0 is a fixed (ignorable) input
    l0, _ = pna(nf, ef, ei, hidden=None)
    assert torch.isfinite(l0).all()


def test_dynamic_pna_backward_no_nan_isolated() -> None:
    # an ISOLATED node (degree 0) must not inject NaN/inf gradients through the PNA std / attenuation
    # scaler (the D1 gradient-safe guards in the static primitives).
    pna = _actor(node_dim=5, edge_dim=3, hidden=8)
    nf = torch.randn(4, 5)
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)         # node 3 is isolated (no incident edge)
    ef = torch.randn(ei.shape[0], 3)
    logits, h = pna(nf, ef, ei, hidden=None)
    (logits.sum() + h.sum()).backward()
    grads = [p.grad for p in pna.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_dynamic_pna_is_decentralized() -> None:
    # the actor declares no global state / global decoder (deployment-decentralization, D1).
    pna = _actor()
    rep = pna.boundary_report()
    assert rep["global_topology_used"] is False
    assert rep["decentralized_with_communication"] is True
    assert rep["activation_owned_by_decoder"] is True
