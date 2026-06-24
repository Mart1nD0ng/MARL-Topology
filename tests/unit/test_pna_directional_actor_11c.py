"""Phase 11c (Spec §7.12): the deployable Preference-conditioned Directional PNA Actor.

Pins: it is a signature-compatible drop-in for MessagePassingGraphEdgeScorer (forward -> [B,E], and
forward_logits works unchanged); the omega preference genuinely changes the logits and sweeps a range;
the activation logit is SYMMETRIC in an edge's endpoints (undirected activation); permutation-equivariant;
D1 boundary report; device + variable N/E; no critic/global-state import (deployment-clean).
"""

from __future__ import annotations

import torch

from marl_topology.models.pna_directional_actor import PNADirectionalActor
from marl_topology.models.message_passing_graph_edge_scorer import MessagePassingGraphEdgeScorer

ND, ED, N, E = 6, 4, 5, 7


def _actor(seed=0, rounds=3):
    torch.manual_seed(seed)
    return PNADirectionalActor(ND, ED, hidden=16, rounds=rounds, delta=1.2)


def _scene(seed=0):
    torch.manual_seed(seed + 100)
    nf = torch.randn(1, N, ND)
    ef = torch.randn(1, E, ED)
    ei = torch.randint(0, N, (1, E, 2))
    return nf, ef, ei, torch.ones(1, N), torch.ones(1, E)


def test_forward_matches_actor_signature_shape() -> None:
    actor = _actor()
    nf, ef, ei, nm, em = _scene()
    out = actor(nf, ef, ei, nm, em)
    assert out.shape == (1, E)                          # [B, E] per-edge logits, like the MLP actor
    assert torch.isfinite(out).all()


def test_forward_logits_compatible() -> None:
    # the trunk's forward_logits calls actor(nf, ef, ei, ones, ones)[0] -> [E]; must work unchanged.
    from marl_topology.training.decentralized_distillation import forward_logits
    actor = _actor()
    nf = torch.randn(N, ND); ef = torch.randn(E, ED); ei = torch.randint(0, N, (E, 2))
    sample = {"nf": nf, "ef": ef, "ei": ei}
    mean = (torch.zeros(ND), torch.zeros(ED)); std = (torch.ones(ND), torch.ones(ED))
    logits = forward_logits(actor, sample, mean, std)
    assert logits.shape == (E,) and torch.isfinite(logits).all()


def test_omega_preference_changes_and_sweeps_logits() -> None:
    actor = _actor()
    nf, ef, ei, nm, em = _scene()
    o_energy = actor(nf, ef, ei, nm, em, omega=torch.tensor([1.0, 0.0]))
    o_bal = actor(nf, ef, ei, nm, em, omega=torch.tensor([0.5, 0.5]))
    o_lat = actor(nf, ef, ei, nm, em, omega=torch.tensor([0.0, 1.0]))
    # preference genuinely conditions the policy (a single net spanning the Pareto front)
    assert not torch.allclose(o_energy, o_lat, atol=1e-5)
    assert not torch.allclose(o_energy, o_bal, atol=1e-5)
    # default (no omega) == neutral (0.5, 0.5)
    assert torch.allclose(actor(nf, ef, ei, nm, em), o_bal, atol=1e-6)


def test_edge_logit_symmetric_in_endpoints() -> None:
    # an undirected candidate edge's activation logit must not depend on the (u,v) vs (v,u) ORDER in ei
    actor = _actor()
    nf = torch.randn(1, N, ND); ef = torch.randn(1, E, ED)
    ei = torch.randint(0, N, (1, E, 2))
    ei_swapped = ei.clone(); ei_swapped[..., [0, 1]] = ei[..., [1, 0]]   # flip every edge's endpoints
    a = actor(nf, ef, ei, torch.ones(1, N), torch.ones(1, E))
    b = actor(nf, ef, ei_swapped, torch.ones(1, N), torch.ones(1, E))
    assert torch.allclose(a, b, atol=1e-5)             # symmetric (directional backbone, symmetric head)


def test_permutation_equivariant_in_nodes() -> None:
    actor = _actor()
    nf = torch.randn(N, ND); ef = torch.randn(E, ED); ei = torch.randint(0, N, (E, 2))
    base = actor(nf.unsqueeze(0), ef.unsqueeze(0), ei.unsqueeze(0), torch.ones(1, N), torch.ones(1, E))[0]
    perm = torch.randperm(N); inv = torch.argsort(perm)
    ei_p = inv[ei]                                       # relabel endpoints (edge order preserved)
    out_p = actor(nf[perm].unsqueeze(0), ef.unsqueeze(0), ei_p.unsqueeze(0),
                  torch.ones(1, N), torch.ones(1, E))[0]
    assert torch.allclose(out_p, base, atol=1e-5)       # node relabeling leaves per-edge logits unchanged


def test_d1_boundary_report_and_no_critic_import() -> None:
    rep = _actor().boundary_report()
    assert rep["global_topology_used"] is False
    assert rep["decentralized_with_communication"] is True
    assert rep["preference_conditioned"] is True
    assert rep["outputs"] == "per_edge_activation_logit"
    # deployment-clean: the actor module must not import the centralized critic / training-only symbols
    import marl_topology.models.pna_directional_actor as m
    src = __import__("inspect").getsource(m)
    for banned in ("centralized_graph_critic", "graph_mappo", "counterfactual", "scq_"):
        assert banned not in src


def test_variable_n_e_and_no_edges() -> None:
    actor = _actor()
    for n, e in [(3, 2), (8, 15), (4, 0)]:
        nf = torch.randn(1, n, ND); ef = torch.randn(1, e, ED)
        ei = torch.randint(0, n, (1, e, 2)) if e else torch.zeros(1, 0, 2, dtype=torch.long)
        out = actor(nf, ef, ei, torch.ones(1, n), torch.ones(1, e))
        assert out.shape == (1, e) and torch.isfinite(out).all()


def test_device_if_cuda() -> None:
    if not torch.cuda.is_available():
        return
    actor = _actor().cuda()
    nf, ef, ei, nm, em = (t.cuda() for t in _scene())
    out = actor(nf, ef, ei, nm, em)
    assert out.device.type == "cuda" and torch.isfinite(out).all()
