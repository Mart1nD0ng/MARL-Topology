"""T4 (Temporal Recovery) — edge-level recurrent state (task 3.4).

The CSI dynamics are EDGE attributes (ef cols 0-3), but the actor's recurrence is a per-NODE GRUCell. T4 adds
an opt-in EDGE-level GRU (``edge_recurrent=True``) that carries a per-edge hidden [E,H] across frames (edges are
a fixed candidate set per scene) and an ``edge_belief_head`` reading that edge hidden directly. edge_recurrent
defaults to False -> byte-identical to the frozen R1-R8 / T2 / T3 actor (the T4 modules are appended last and
allocated only when enabled). Fails on HEAD: BeliefResidualActor has no edge_recurrent / edge_gru / belief_edge.
"""

from __future__ import annotations

import pytest
import torch

from marl_topology.models.belief_residual_actor import BeliefResidualActor

_REC = (torch.nn.GRUCell, torch.nn.LSTMCell, torch.nn.GRU, torch.nn.LSTM)


def test_edge_recurrent_adds_edge_gru() -> None:
    a0 = BeliefResidualActor(4, 6, hidden=8)                             # default: per-node only
    assert not hasattr(a0, "edge_gru")
    a1 = BeliefResidualActor(4, 6, hidden=8, edge_recurrent=True)        # T4: per-node + per-edge
    assert isinstance(a1.edge_gru, torch.nn.GRUCell)
    assert len([m for m in a1.modules() if isinstance(m, _REC)]) == 2    # node gru + edge gru
    br = a1.boundary_report()
    assert br["edge_recurrent"] is True and br["recurrence_locus"] == "node+edge"


def test_belief_edge_returns_edge_hidden_and_carries() -> None:
    """belief_edge returns a per-edge hidden [E,H]; carrying it across a call CHANGES the belief -> the edge
    recurrence is load-bearing (not inert)."""
    a = BeliefResidualActor(4, 6, hidden=8, edge_recurrent=True, belief_extra_dim=0)
    nf = torch.randn(5, 4)
    ei = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    ef = torch.rand(3, 6)
    bel, he = a.belief_edge(nf, ef, ei)
    assert bel.shape == (3,) and he.shape == (3, 8)                      # edge hidden is per-EDGE [E,H]
    bel2, _he2 = a.belief_edge(nf, ef, ei, edge_hidden=he)
    assert not torch.allclose(bel, bel2)                                 # carried edge memory changes the belief


def test_belief_edge_requires_flag() -> None:
    a = BeliefResidualActor(4, 6, hidden=8)                              # edge_recurrent=False
    with pytest.raises(RuntimeError):
        a.belief_edge(torch.randn(5, 4), torch.rand(3, 6), torch.tensor([[0, 1], [1, 2]], dtype=torch.long))


def test_edge_modules_appended_last_preserve_init() -> None:
    """edge_recurrent=True allocates the edge modules LAST -> the node gru + belief_head init RNG is unchanged,
    so the R2/T3 belief() path is byte-identical whether or not edge recurrence is enabled."""
    torch.manual_seed(0)
    a0 = BeliefResidualActor(4, 6, hidden=8, belief_extra_dim=2)
    torch.manual_seed(0)
    a1 = BeliefResidualActor(4, 6, hidden=8, belief_extra_dim=2, edge_recurrent=True)
    # EVERY shared parameter is bit-identical (not just belief_head/gru) -> the whole non-edge actor is preserved
    sd0, sd1 = a0.state_dict(), a1.state_dict()
    for k, v in sd0.items():
        assert k in sd1 and torch.equal(v, sd1[k]), f"shared param {k} changed when edge_recurrent enabled"
    # a1 adds ONLY the edge modules on top
    extra = set(sd1) - set(sd0)
    assert extra and all("edge" in k for k in extra)
