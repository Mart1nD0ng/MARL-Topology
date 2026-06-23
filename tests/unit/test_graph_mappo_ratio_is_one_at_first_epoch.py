"""Phase 7 load-bearing guard: the PPO importance ratio is exactly 1 before any actor update.

logp_old is captured at sampling; logp_new re-scores the recorded order over the FROZEN gated set
on the UNCHANGED actor. ratio must be exactly 1 (so L_clip == -mean(A), approx_kl == 0) at inner
epoch 0, and depart from 1 only after a logit step. This only holds because the gated set is
frozen at sample time -- it doubles as the regression test for that correctness fix.
"""

import torch

from marl_topology.training.decentralized_action import (
    recompute_logp,
    sample_decentralized_action,
)
from marl_topology.training.graph_mappo import ppo_clip_actor_loss

EDGES = {"AB": ("A", "B"), "BC": ("B", "C"), "AC": ("A", "C")}
EID = ["AB", "BC", "AC"]


def _joint_recompute(logits, action, temperature):
    j = logits.new_zeros(())
    for pa in action.per_agent:
        if pa.accepted_order:
            j = j + recompute_logp(logits, pa.gated_edge_indices, pa.accepted_order, temperature)
    return j


def test_ratio_is_one_at_epoch0_then_departs_after_step():
    logits = torch.tensor([0.5, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    temp = 1.0
    torch.manual_seed(0)
    act = sample_decentralized_action(logits, EID, edges=EDGES, budgets={"A": 2, "B": 2, "C": 2}, temperature=temp)
    logp_old = act.joint_logp.detach()

    logp_new = _joint_recompute(logits, act, temp)  # actor unchanged
    ratio = torch.exp(logp_new - logp_old)
    assert torch.allclose(ratio, torch.ones((), dtype=torch.float64), atol=1e-9)

    advantage = torch.tensor([0.3], dtype=torch.float64)
    loss, info = ppo_clip_actor_loss(logp_new.reshape(1), logp_old.reshape(1), advantage)
    assert torch.allclose(info["approx_kl"], torch.zeros((), dtype=torch.float64), atol=1e-9)
    assert torch.allclose(loss, -advantage.mean(), atol=1e-9)  # ratio==1 -> L_clip == -mean(A)
    assert float(info["clip_fraction"]) == 0.0

    # A NON-uniform logit step departs the ratio (a uniform shift would not: the PL log-prob is
    # shift-invariant by the softmax property, so the ratio only moves with relative logit changes).
    moved = logits.detach().clone()
    moved[0] = moved[0] + 1.5
    ratio_after = torch.exp(_joint_recompute(moved, act, temp) - logp_old)
    assert not torch.allclose(ratio_after, torch.ones((), dtype=torch.float64), atol=1e-3)
