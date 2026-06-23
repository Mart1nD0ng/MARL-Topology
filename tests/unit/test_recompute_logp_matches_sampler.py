"""Phase 7 spine: recompute_logp re-scores a recorded PL order over the FROZEN gated set.

The PPO importance ratio re-scores the recorded ordered acceptance set against fresh logits. It
MUST re-score over the gated set frozen at sample time (not re-derive the logit>=0 gate, which
drifts as logits move across inner epochs). This pins recompute_logp to the genuine Phase-6
Plackett-Luce log-prob produced by the sampler.
"""

import torch

from marl_topology.training.decentralized_action import (
    recompute_logp,
    sample_decentralized_action,
)

EDGES = {"AB": ("A", "B"), "BC": ("B", "C"), "AC": ("A", "C")}
EID = ["AB", "BC", "AC"]


def test_recompute_logp_matches_per_agent_and_joint():
    logits = torch.tensor([0.5, 0.2, 0.8], dtype=torch.float64)
    temp = 1.3
    torch.manual_seed(0)
    act = sample_decentralized_action(logits, EID, edges=EDGES, budgets={"A": 2, "B": 2, "C": 2}, temperature=temp)
    rejoint = torch.zeros((), dtype=torch.float64)
    scored = 0
    for pa in act.per_agent:
        if not pa.accepted_order:
            continue
        scored += 1
        lp = recompute_logp(logits, pa.gated_edge_indices, pa.accepted_order, temp)
        assert torch.allclose(lp, pa.logp, atol=1e-10), (lp, pa.logp)
        rejoint = rejoint + lp
    assert scored >= 1
    assert torch.allclose(rejoint, act.joint_logp, atol=1e-10)


def test_recompute_logp_is_differentiable():
    logits = torch.tensor([0.5, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    torch.manual_seed(0)
    act = sample_decentralized_action(logits, EID, edges=EDGES, budgets={"A": 2, "B": 2, "C": 2}, temperature=1.0)
    pa = next(p for p in act.per_agent if p.accepted_order)
    recompute_logp(logits, pa.gated_edge_indices, pa.accepted_order, 1.0).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_recompute_logp_uses_frozen_gate_not_redirived():
    # An edge whose logit later drops below 0 must STILL be scored if it was in the frozen gated
    # set + recorded order -- recompute_logp must not silently drop it by re-deriving the gate.
    logits = torch.tensor([0.5, 0.2, 0.8], dtype=torch.float64)
    torch.manual_seed(0)
    act = sample_decentralized_action(logits, EID, edges=EDGES, budgets={"A": 2, "B": 2, "C": 2}, temperature=1.0)
    pa = next(p for p in act.per_agent if len(p.accepted_order) >= 1)
    moved = logits.clone()
    for g in pa.gated_edge_indices:
        moved[g] = -3.0  # push the whole frozen gated set below the gate
    lp = recompute_logp(moved, pa.gated_edge_indices, pa.accepted_order, 1.0)
    assert torch.isfinite(lp)  # still a well-defined PL log-prob over the frozen set
