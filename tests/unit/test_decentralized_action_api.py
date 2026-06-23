"""Phase 6 contract: the decentralized per-agent action API (action_i / logp_i / entropy_i).

These are the invariants MAPPO/COMA/SCQ depend on: the joint log-prob decomposes into a sum of
per-agent log-probs, an edge activates iff both endpoints accept (mutual symmetry), each node
respects its own budget using only its own incident-edge logits (local observability, no global
sort), the per-agent entropy is the real Plackett-Luce policy entropy, and the temperature->0
limit reproduces the deployed decoder.
"""

import math

import torch

from marl_topology.training.decentralized_action import (
    deterministic_decentralized_action,
    sample_decentralized_action,
)

EDGES = {"AB": ("A", "B"), "BC": ("B", "C"), "AC": ("A", "C")}
EDGE_IDS = ["AB", "BC", "AC"]


def _sample(logits, budgets, temperature=1.0, seed=0, compute_entropy=True):
    torch.manual_seed(seed)
    return sample_decentralized_action(
        torch.tensor(logits, dtype=torch.float64),
        EDGE_IDS,
        edges=EDGES,
        budgets=budgets,
        temperature=temperature,
        compute_entropy=compute_entropy,
    )


def test_joint_logp_is_sum_of_per_agent_logp():
    act = _sample([0.5, 0.2, -0.1], {"A": 1, "B": 1, "C": 1})
    per_agent_sum = sum((pa.logp for pa in act.per_agent), torch.zeros((), dtype=torch.float64))
    assert torch.allclose(act.joint_logp, per_agent_sum)


def test_joint_entropy_is_sum_of_per_agent_entropy():
    act = _sample([0.5, 0.2, 0.8], {"A": 2, "B": 2, "C": 2})
    per_agent_sum = sum((pa.entropy for pa in act.per_agent), torch.zeros((), dtype=torch.float64))
    assert torch.allclose(act.joint_entropy, per_agent_sum)


def test_mutual_symmetry():
    act = _sample([1.0, 1.0, 1.0], {"A": 2, "B": 2, "C": 2})
    accept = {pa.node_id: set(pa.accepted_order) for pa in act.per_agent}
    for i, eid in enumerate(EDGE_IDS):
        u, v = EDGES[eid]
        assert (i in act.active_edge_indices) == (i in accept[u] and i in accept[v])


def test_budget_feasibility():
    act = _sample([1.0, 1.0, 1.0], {"A": 1, "B": 1, "C": 1})
    for pa in act.per_agent:
        assert len(pa.accepted_order) <= pa.budget


def test_local_observability_far_edge_does_not_change_node():
    # Edge BC is not incident to A; changing its logit must not change A's candidate set or entropy.
    a1 = _sample([0.5, 0.3, 0.4], {"A": 1, "B": 1, "C": 1})
    a2 = _sample([0.5, -9.0, 0.4], {"A": 1, "B": 1, "C": 1})
    pa1 = next(p for p in a1.per_agent if p.node_id == "A")
    pa2 = next(p for p in a2.per_agent if p.node_id == "A")
    assert pa1.gated_edge_indices == pa2.gated_edge_indices == (0, 2)
    assert torch.allclose(pa1.entropy, pa2.entropy)


def test_entropy_budget1_equals_categorical_entropy():
    # Only A acts (budget 1) over its two gated edges AB(0.5), AC(0.4): H = H(softmax([0.5, 0.4])).
    act = _sample([0.5, -9.0, 0.4], {"A": 1, "B": 0, "C": 0})
    pa = next(p for p in act.per_agent if p.node_id == "A")
    z = torch.tensor([0.5, 0.4], dtype=torch.float64)
    p = torch.softmax(z, 0)
    expected = -(p * torch.log(p)).sum()
    assert torch.allclose(pa.entropy, expected, atol=1e-9)


def test_entropy_is_nonnegative_and_rises_with_temperature():
    low = _sample([0.9, 0.2, 0.5], {"A": 2, "B": 2, "C": 2}, temperature=0.5)
    high = _sample([0.9, 0.2, 0.5], {"A": 2, "B": 2, "C": 2}, temperature=4.0)
    assert float(low.joint_entropy) >= 0.0
    assert float(high.joint_entropy) >= float(low.joint_entropy)


def test_entropy_matches_monte_carlo():
    logits, budgets, temp = [0.5, 0.2, 0.8], {"A": 2, "B": 2, "C": 2}, 1.0
    ref = _sample(logits, budgets, temperature=temp)
    h_analytic = float(ref.joint_entropy)
    total, n = 0.0, 4000
    for s in range(n):
        a = _sample(logits, budgets, temperature=temp, seed=1000 + s, compute_entropy=False)
        total += float(a.joint_logp)
    h_mc = -total / n
    assert abs(h_analytic - h_mc) < 0.06, (h_analytic, h_mc)


def test_deterministic_limit_matches_local_mutual_decoder():
    logits = torch.tensor([0.5, -0.3, 0.4], dtype=torch.float64)
    budgets = {"A": 1, "B": 1, "C": 1}
    det = deterministic_decentralized_action(logits, EDGE_IDS, edges=EDGES, budgets=budgets)
    # A picks AB(0.5)>AC(0.4); B picks AB (BC gated out, <0); C picks AC. AB active (A&B), AC not (A picked AB).
    assert set(det) == {0}
    # the temperature->0 sampler reproduces it
    low = _sample([0.5, -0.3, 0.4], budgets, temperature=1e-4)
    assert set(low.active_edge_indices) == {0}


def test_logp_and_entropy_carry_gradient():
    logits = torch.tensor([0.5, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    torch.manual_seed(0)
    act = sample_decentralized_action(logits, EDGE_IDS, edges=EDGES, budgets={"A": 2, "B": 2, "C": 2}, temperature=1.0)
    (act.joint_logp + act.joint_entropy).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
