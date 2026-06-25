"""Episode-recurrent actor: per-node hidden state carried across frames (Spec S7.12 temporal actor).

Pins: forward returns (per-edge logits, per-node hidden); the carried hidden ACTUALLY changes the
output (cross-frame memory is real, not inert); gradients flow across frames (BPTT); the edge logit
is symmetric in its endpoints; permutation-equivariant in node order; isolated nodes / no edges give
no NaN; device/dtype preserved.
"""

from __future__ import annotations

import torch

from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor

ND, ED, H = 8, 6, 16


def _actor(seed=0):
    torch.manual_seed(seed)
    return DynamicRecurrentActor(ND, ED, hidden=H)


def _graph(n=4, e=5, seed=1):
    torch.manual_seed(seed)
    nf = torch.randn(n, ND)
    ei = torch.randint(0, n, (e, 2))
    ef = torch.randn(e, ED)
    return nf, ef, ei


def test_shapes_and_hidden_returned() -> None:
    a = _actor()
    nf, ef, ei = _graph()
    logits, h = a(nf, ef, ei)
    assert logits.shape == (ei.shape[0],)
    assert h.shape == (nf.shape[0], H)


def test_carried_hidden_changes_output() -> None:
    # the SAME frame observation, run with a fresh (zero) hidden vs a carried non-zero hidden, must
    # give DIFFERENT logits -> the cross-frame memory genuinely affects the policy.
    a = _actor()
    nf, ef, ei = _graph()
    logits_fresh, h1 = a(nf, ef, ei, hidden=None)
    # carry a hidden produced by a different previous frame
    nf_prev, ef_prev, ei_prev = _graph(seed=9)
    _lp, h_prev = a(nf_prev, ef_prev, ei_prev, hidden=None)
    logits_carried, _ = a(nf, ef, ei, hidden=h_prev)
    assert not torch.allclose(logits_fresh, logits_carried, atol=1e-5)


def test_memoryless_mode_is_deterministic_in_obs() -> None:
    # with hidden reset each call (memoryless control), identical obs -> identical logits.
    a = _actor()
    nf, ef, ei = _graph()
    l1, _ = a(nf, ef, ei, hidden=None)
    l2, _ = a(nf, ef, ei, hidden=None)
    assert torch.allclose(l1, l2, atol=1e-6)


def test_bptt_gradients_flow_across_frames() -> None:
    a = _actor()
    h = None
    total = 0.0
    for t in range(3):                       # roll 3 frames carrying the hidden state
        nf, ef, ei = _graph(seed=t + 1)
        logits, h = a(nf, ef, ei, hidden=h)
        total = total + logits.sum()
    total.backward()
    grads = [p.grad for p in a.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # the GRU (cross-frame) must receive gradient -> recurrence is trained, not inert
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in a.gru.parameters())


def test_cross_frame_carry_discriminates_recurrent_from_memoryless() -> None:
    # The decisive property: under a loss on ONLY the LAST frame, an EARLY frame's input influences
    # the last frame's output iff the hidden state carries. Recurrent -> nonzero grad of the last
    # logits w.r.t. the first frame's node features; memoryless (hidden reset) -> exactly zero.
    a = _actor()
    frames = [(_graph(seed=t + 1)) for t in range(3)]

    def last_logit_grad_wrt_first_nf(recurrent: bool) -> float:
        nf0 = frames[0][0].clone().requires_grad_(True)
        h = None
        logits = None
        for t, (nf, ef, ei) in enumerate(frames):
            x = nf0 if t == 0 else nf
            logits, h_next = a(x, ef, ei, hidden=h)
            h = h_next if recurrent else None
        logits.sum().backward()
        return 0.0 if nf0.grad is None else float(nf0.grad.abs().sum())

    g_rec = last_logit_grad_wrt_first_nf(recurrent=True)
    g_mem = last_logit_grad_wrt_first_nf(recurrent=False)
    assert g_rec > 1e-6, "recurrent: the first frame must influence the last (cross-frame carry)"
    assert g_mem == 0.0, "memoryless: the last frame must be independent of the first (no carry)"


def test_edge_logit_symmetric_in_endpoints() -> None:
    a = _actor()
    nf = torch.randn(3, ND)
    ef = torch.randn(1, ED)
    l_uv, _ = a(nf, ef, torch.tensor([[0, 1]]))
    l_vu, _ = a(nf, ef, torch.tensor([[1, 0]]))
    assert torch.allclose(l_uv, l_vu, atol=1e-5)   # activation is mutual / undirected


def test_permutation_equivariant() -> None:
    a = _actor()
    nf, ef, ei = _graph(n=5, e=6, seed=3)
    base, _ = a(nf, ef, ei)
    perm = torch.tensor([2, 0, 4, 1, 3])
    inv = torch.argsort(perm)
    ei_p = torch.stack([inv[ei[:, 0]], inv[ei[:, 1]]], dim=1)
    out_p, _ = a(nf[perm], ef, ei_p)
    assert torch.allclose(out_p, base, atol=1e-5)   # edge order unchanged -> logits unchanged


def test_isolated_nodes_and_no_edges_no_nan() -> None:
    a = _actor()
    nf = torch.randn(3, ND)
    out, h = a(nf, torch.randn(1, ED), torch.tensor([[0, 1]]))   # node 2 isolated
    assert torch.isfinite(out).all() and torch.isfinite(h).all()
    out0, h0 = a(nf, torch.zeros(0, ED), torch.zeros(0, 2, dtype=torch.long))   # no edges
    assert out0.shape == (0,) and torch.isfinite(h0).all()


def test_device_dtype() -> None:
    if torch.cuda.is_available():
        a = _actor().cuda()
        nf, ef, ei = _graph()
        logits, h = a(nf.cuda(), ef.cuda(), ei.cuda())
        assert logits.device.type == "cuda" and h.device.type == "cuda"
        assert torch.isfinite(logits).all()
