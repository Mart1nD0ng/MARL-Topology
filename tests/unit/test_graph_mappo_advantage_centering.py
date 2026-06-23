"""Phase 7: the single-step (T=1) advantage is exactly r - V(scene).detach().

No bootstrap, no gamma/lambda -- this IS GAE at T=1. The detached value means the actor's PPO
gradient never flows into the critic through the advantage; the critic replaces the EMA/RLOO
baseline.
"""

import torch

from marl_topology.training.graph_mappo import graph_mappo_advantage


def test_advantage_is_reward_minus_detached_value():
    value = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    adv = graph_mappo_advantage(0.7, value)
    assert torch.allclose(adv, torch.tensor(0.3, dtype=torch.float64))
    assert not adv.requires_grad  # value is detached -> no grad path into the critic via the actor


def test_advantage_centers_when_value_predicts_mean_reward():
    rewards = [0.2, 0.5, 0.8]
    mean = sum(rewards) / len(rewards)
    advs = [float(graph_mappo_advantage(r, torch.tensor(mean, dtype=torch.float64))) for r in rewards]
    assert abs(sum(advs) / len(advs)) < 1e-12
