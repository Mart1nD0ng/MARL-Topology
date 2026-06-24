"""Phase 11b (Spec §7.12 items 2 & 4): directional message passing + recurrent shared update.

Pins: scatter_directional_pna == per-node PNA aggregators; the layer is DIRECTIONAL (edge direction
decides who receives); the recurrent update SHARES one GRUCell across all rounds (param count independent
of rounds); rounds=0 is encoder-only; permutation-equivariant in node order; isolated nodes give no NaN;
device/dtype preserved.
"""

from __future__ import annotations

import torch

from marl_topology.models.pna_aggregation import pna_aggregators
from marl_topology.models.recurrent_directional_pna import (
    RecurrentDirectionalPNA,
    scatter_directional_pna,
)

ND, ED, H = 6, 4, 16


def _module(rounds=3, delta=1.2, seed=0):
    torch.manual_seed(seed)
    return RecurrentDirectionalPNA(ND, ED, hidden=H, rounds=rounds, delta=delta)


def test_scatter_pna_matches_per_node_aggregators() -> None:
    torch.manual_seed(1)
    n, e, f = 4, 9, 5
    messages = torch.randn(e, f)
    dst = torch.randint(0, n, (e,))
    got = scatter_directional_pna(messages, dst, n)             # [n, 4, f]
    for node in range(n):
        incoming = messages[dst == node]
        expect = pna_aggregators(incoming)                     # [4, f]
        assert torch.allclose(got[node], expect, atol=1e-5), f"node {node}"


def test_layer_is_directional() -> None:
    # single directed edge: with 0->1 node 1 receives; reversing to 1->0 makes node 0 receive instead.
    mod = _module(rounds=2)
    nf = torch.randn(2, ND)
    ef = torch.randn(1, ED)
    h_fwd = mod(nf, torch.tensor([[0, 1]]), ef)     # 0 -> 1
    h_rev = mod(nf, torch.tensor([[1, 0]]), ef)     # 1 -> 0 (same feature, reversed direction)
    # node 1 received a message in fwd but not in rev -> its embedding must differ (direction matters)
    assert not torch.allclose(h_fwd[1], h_rev[1], atol=1e-5)
    assert not torch.allclose(h_fwd[0], h_rev[0], atol=1e-5)


def test_recurrent_shares_one_gru_across_rounds() -> None:
    # parameter count is INDEPENDENT of the number of rounds -> the GRUCell + msg MLP are shared, not
    # stacked per-round.
    p2 = sum(p.numel() for p in _module(rounds=2).parameters())
    p5 = sum(p.numel() for p in _module(rounds=5).parameters())
    assert p2 == p5
    assert len([m for m in _module().modules() if isinstance(m, torch.nn.GRUCell)]) == 1


def test_rounds_zero_is_encoder_only() -> None:
    mod = _module(rounds=0)
    nf = torch.randn(3, ND)
    ei = torch.tensor([[0, 1], [1, 2]])
    ef = torch.randn(2, ED)
    out = mod(nf, ei, ef)
    assert torch.allclose(out, mod.encoder(nf), atol=1e-6)   # no message passing at rounds=0


def test_more_rounds_change_output_and_propagate() -> None:
    mod1, mod2 = _module(rounds=1, seed=3), _module(rounds=3, seed=3)
    # identical weights so the only difference is the round count
    mod2.load_state_dict(mod1.state_dict(), strict=False)
    nf = torch.randn(4, ND)
    ei = torch.tensor([[0, 1], [1, 2], [2, 3]])   # a path 0->1->2->3
    ef = torch.randn(3, ED)
    h1, h3 = mod1(nf, ei, ef), mod2(nf, ei, ef)
    assert not torch.allclose(h1, h3, atol=1e-5)   # more rounds -> larger receptive field -> different


def test_permutation_equivariant() -> None:
    mod = _module(rounds=2)
    nf = torch.randn(5, ND)
    ei = torch.tensor([[0, 1], [1, 2], [3, 4], [2, 0]])
    ef = torch.randn(4, ED)
    base = mod(nf, ei, ef)
    perm = torch.tensor([2, 0, 4, 1, 3])
    inv = torch.argsort(perm)
    nf_p = nf[perm]
    ei_p = torch.stack([inv[ei[:, 0]], inv[ei[:, 1]]], dim=1)   # relabel edges under the permutation
    out_p = mod(nf_p, ei_p, ef)
    assert torch.allclose(out_p[inv], base, atol=1e-5)          # equivariant: permuting nodes permutes output


def test_isolated_node_and_no_edges_no_nan() -> None:
    mod = _module(rounds=2)
    nf = torch.randn(3, ND)
    # node 2 isolated (no incoming); node 0,1 connected
    out = mod(nf, torch.tensor([[0, 1]]), torch.randn(1, ED))
    assert torch.isfinite(out).all()
    out0 = mod(nf, torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, ED))   # no edges at all
    assert torch.isfinite(out0).all()   # no NaN/inf; the shared GRU still self-updates from a zero message
    # (no edges != encoder-only: only rounds=0 is encoder-only -- see test_rounds_zero_is_encoder_only)


def test_device_dtype() -> None:
    if torch.cuda.is_available():
        mod = _module(rounds=2).cuda()
        nf = torch.randn(3, ND, device="cuda")
        out = mod(nf, torch.tensor([[0, 1], [1, 2]], device="cuda"), torch.randn(2, ED, device="cuda"))
        assert out.device.type == "cuda" and torch.isfinite(out).all()
