"""Q9 (POMDP-QP-FAR): potential-based reward shaping (PBRS) -- the Ng-Harada guarantees.

Pins: the shaping telescopes to -lam*Phi_0 with terminal Phi_T=0, and (the key theorem) it does NOT
change the optimal policy of a small MDP -- so D_quorum can enter the TRAINING reward as shaping without
altering the true objective. Fails on HEAD (the module is new).
"""

from __future__ import annotations

import pytest

from marl_topology.training.potential_shaping import (
    discounted_shaping_sum, episode_pbrs, pbrs_term)


def test_pbrs_telescopes_with_terminal_zero() -> None:
    potentials = [1.5, -0.3, 2.1, 0.7]                       # Phi_0..Phi_3 (terminal Phi_4 = 0)
    gamma, lam = 0.9, 0.5
    shaping = episode_pbrs(potentials, gamma, lam)
    assert len(shaping) == len(potentials)
    total = discounted_shaping_sum(shaping, gamma)
    assert total == pytest.approx(lam * (-potentials[0]), abs=1e-9)   # telescopes to -lam*Phi_0


def test_pbrs_terminal_potential_zero() -> None:
    potentials = [2.0, 1.0, 0.5]
    gamma, lam = 0.95, 1.0
    shaping = episode_pbrs(potentials, gamma, lam)
    # the LAST term uses the terminal Phi_T = 0
    assert shaping[-1] == pytest.approx(lam * (gamma * 0.0 - potentials[-1]), abs=1e-12)
    assert shaping[0] == pytest.approx(pbrs_term(potentials[0], potentials[1], gamma, lam), abs=1e-12)


def test_pbrs_preserves_small_mdp_optimum() -> None:
    # A tiny deterministic MDP. The shaped reward r'(s,a) = r(s,a) + gamma*Phi(s') - Phi(s) must yield the
    # SAME optimal policy as r(s,a) for ANY potential (Ng-Harada). A sign error in the shaping form would
    # flip an argmax -> this test would fail.
    states = [0, 1, 2]
    actions = [0, 1]
    gamma = 0.9
    trans = {(0, 0): 1, (0, 1): 2, (1, 0): 2, (1, 1): 0, (2, 0): 2, (2, 1): 1}
    rew = {(0, 0): 0.0, (0, 1): 0.6, (1, 0): 0.5, (1, 1): 0.2, (2, 0): 0.1, (2, 1): 0.4}

    def optimal_policy(reward_fn):
        V = {s: 0.0 for s in states}
        for _ in range(500):
            V = {s: max(reward_fn(s, a) + gamma * V[trans[(s, a)]] for a in actions) for s in states}
        return {s: max(actions, key=lambda a: reward_fn(s, a) + gamma * V[trans[(s, a)]]) for s in states}

    base = optimal_policy(lambda s, a: rew[(s, a)])
    for phi in ({0: 0.3, 1: -0.5, 2: 1.0}, {0: -2.0, 1: 3.1, 2: 0.0}, {0: 5.0, 1: 5.0, 2: -4.0}):
        shaped = optimal_policy(lambda s, a, p=phi: rew[(s, a)] + gamma * p[trans[(s, a)]] - p[s])
        assert shaped == base, phi                          # PBRS preserves the optimal policy
