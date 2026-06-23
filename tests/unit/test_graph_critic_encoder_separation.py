"""Phase 7 item-5 + D1: the centralized critic encoder is fully separate from the actor.

The critic shares ZERO nn.Parameter objects with MessagePassingGraphEdgeScorer, holds no reference
to the actor, and a backward through the critic loss leaves the actor's params' .grad None -- they
are trained by separate optimizers and the centralized critic can never leak into the deployed
actor.
"""

import torch

from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic
from marl_topology.models.message_passing_graph_edge_scorer import MessagePassingGraphEdgeScorer


def test_critic_shares_no_parameters_with_actor():
    critic = CentralizedGraphCritic(node_dim=8, edge_dim=8, hidden=16, rounds=2)
    actor = MessagePassingGraphEdgeScorer(node_dim=8, edge_dim=8, hidden=16, rounds=2)
    critic_param_ids = {id(p) for p in critic.parameters()}
    actor_param_ids = {id(p) for p in actor.parameters()}
    assert critic_param_ids.isdisjoint(actor_param_ids)
    assert not any(m is actor for m in critic.modules())


def test_critic_backward_does_not_touch_actor_grad():
    torch.manual_seed(0)
    nf = torch.randn(2, 4, 8)
    ef = torch.randn(2, 5, 8)
    ei = torch.randint(0, 4, (2, 5, 2))
    nm = torch.ones(2, 4)
    em = torch.ones(2, 5)
    critic = CentralizedGraphCritic(node_dim=8, edge_dim=8, hidden=16, rounds=2)
    actor = MessagePassingGraphEdgeScorer(node_dim=8, edge_dim=8, hidden=16, rounds=2)
    value = critic(nf, ef, ei, nm, em)
    critic_loss = (value - 0.5).pow(2).mean()  # (reward - V)^2 surrogate
    critic_loss.backward()
    assert all(p.grad is None for p in actor.parameters())
    assert any(p.grad is not None for p in critic.parameters())
