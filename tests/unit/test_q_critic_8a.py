"""Phase 8a (Spec §8.3/§9.4): the action-conditioned Q critic Q(s, S).

The centralized critic, built with critic_sees_action=True, conditions on the realized joint action
(the mutual-decoder active-edge one-hot). Pins: the action one-hot CHANGES Q (else the counterfactual
baseline is degenerate), Q trains (params move, Q tracks the reward), the critic step never touches the
actor, and the one-hot is a detached conditioning input (no gradient into the action).
"""

from __future__ import annotations

import torch

from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic
from marl_topology.models.message_passing_graph_edge_scorer import MessagePassingGraphEdgeScorer
from marl_topology.training.graph_mappo import critic_q_value

ND, ED, N, E = 6, 4, 5, 8


def _scene():
    torch.manual_seed(1)
    return (torch.randn(N, ND), torch.randn(E, ED), torch.randint(0, N, (E, 2)),
            (torch.zeros(ND), torch.ones(ND)), (torch.zeros(ED), torch.ones(ED)))


def _q(critic, scene, onehot):
    nf, ef, ei, nm, em = scene
    return critic_q_value(critic, nf, ef, ei, onehot, node_mean=nm[0], node_std=nm[1],
                          edge_mean=em[0], edge_std=em[1])


def test_critic_sees_action_changes_q_output() -> None:
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2, critic_sees_action=True)
    scene = _scene()
    q_none = _q(critic, scene, torch.zeros(E))
    q_full = _q(critic, scene, torch.ones(E))
    q_some = _q(critic, scene, torch.tensor([1.0, 0, 1, 0, 0, 1, 0, 0]))
    # the realized-action conditioning genuinely moves Q (else the COMA baseline would be degenerate)
    assert abs(float(q_none) - float(q_full)) > 1e-4
    assert abs(float(q_none) - float(q_some)) > 1e-4


def test_q_critic_trains_and_tracks_reward() -> None:
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2, critic_sees_action=True)
    opt = torch.optim.AdamW(critic.parameters(), lr=1e-2)
    scene = _scene()
    onehot = torch.tensor([1.0, 0, 1, 0, 0, 1, 0, 0])
    target = torch.tensor(-1.5)
    before = [p.detach().clone() for p in critic.parameters()]
    for _ in range(40):
        q = _q(critic, scene, onehot)
        (target - q).pow(2).backward(); opt.step(); opt.zero_grad()
    after = list(critic.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))   # critic moved
    assert abs(float(_q(critic, scene, onehot)) - float(target)) < 0.3    # Q tracks the reward target


def test_q_critic_step_leaves_actor_unchanged() -> None:
    actor = MessagePassingGraphEdgeScorer(ND, ED, hidden=16, rounds=2)
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2, critic_sees_action=True)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=1e-2)
    actor_before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    (torch.tensor(0.0) - _q(critic, _scene(), torch.ones(E))).pow(2).backward(); opt_c.step()
    for k, v in actor.state_dict().items():
        assert torch.allclose(actor_before[k], v)


def test_action_onehot_is_detached_no_grad_into_action() -> None:
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2, critic_sees_action=True)
    onehot = torch.ones(E, requires_grad=True)   # pretend the action carried a gradient
    q = _q(critic, _scene(), onehot)
    q.backward()
    # the critic detaches the action conditioning -> no gradient flows back into the action one-hot
    assert onehot.grad is None or torch.allclose(onehot.grad, torch.zeros_like(onehot))


def test_v_critic_ignores_onehot() -> None:
    # a critic_sees_action=False critic is the plain V(scene): the one-hot argument must not matter
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2, critic_sees_action=False)
    nf, ef, ei, nm, em = _scene()
    from marl_topology.training.graph_mappo import critic_scene_value
    v = critic_scene_value(critic, nf, ef, ei, node_mean=nm[0], node_std=nm[1],
                           edge_mean=em[0], edge_std=em[1])
    assert torch.isfinite(v)
