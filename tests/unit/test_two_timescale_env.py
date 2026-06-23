"""R5 (v2 Engineering-Plan §R5, Spec §3.3-3.6): two-timescale dynamic env + Temporal Value Test.

A macro topology decision is held for H_PBFT micro-rounds; the per-step reward subtracts the
reconfiguration cost of switching topology (a real physical cost, Spec §3.5). The Temporal Value
Test Δ_H = J_myopic − J_horizon decides whether the task needs temporal modeling (Δ_H>0) or the
static T=1 bandit suffices (Δ_H≈0).
"""

from __future__ import annotations

import pytest

from marl_topology.training.two_timescale_env import (
    ReconfigCost,
    TwoTimescaleTopologyEnv,
    temporal_value_test,
)

A = frozenset({"e1"})
B = frozenset({"e2"})


def _alternating_cost(frame, topology):
    # frame 0 prefers A (1.0 < 2.0); frame 1 prefers B -- the per-frame optima ALTERNATE.
    table = {0: {A: 1.0, B: 2.0}, 1: {A: 2.0, B: 1.0}}
    return table[frame][frozenset(topology)]


def test_episode_length_gt_one() -> None:
    env = TwoTimescaleTopologyEnv([0, 1, 2], _const_cost(1.0))
    state = env.reset()
    steps = 0
    done = False
    while not done:
        result = env.step(state, A)
        state, done = result.next_state, result.done
        steps += 1
    assert steps == 3  # a 3-frame trajectory is a 3-step episode (not T=1)


def test_topology_persists_for_hold_interval() -> None:
    env = TwoTimescaleTopologyEnv([0], _const_cost(2.0), hold_interval=5)
    result = env.step(env.reset(), A)
    assert result.base_objective == pytest.approx(2.0 * 5)  # held for 5 micro-rounds


def test_action_changes_future_cost() -> None:
    env = TwoTimescaleTopologyEnv([0, 1], _const_cost(1.0), ReconfigCost(e_edge=1.0))
    s0 = env.reset()
    # choosing A vs B at frame 0 changes the reconfiguration cost of switching to A at frame 1
    after_a = env.step(s0, A).next_state
    after_b = env.step(s0, B).next_state
    cost_from_a = env.step(after_a, A).reconfig_cost   # A->A: no switch
    cost_from_b = env.step(after_b, A).reconfig_cost   # B->A: a 2-edge switch
    assert cost_from_a == pytest.approx(0.0)
    assert cost_from_b > 0.0


def test_reconfiguration_cost_nonzero() -> None:
    env = TwoTimescaleTopologyEnv([0, 1], _const_cost(1.0), ReconfigCost(e_edge=0.5, l_edge=0.5))
    s1 = env.step(env.reset(), A).next_state
    assert env.step(s1, A).reconfig_cost == pytest.approx(0.0)        # hold -> no cost
    assert env.step(s1, B).reconfig_cost == pytest.approx(1.0 * 2)    # switch -> (0.5+0.5)*|A△B|=2


def test_state_fork_isolation() -> None:
    env = TwoTimescaleTopologyEnv([0, 1, 2], _const_cost(1.0), ReconfigCost(e_edge=1.0))
    base = env.step(env.reset(), A).next_state
    fork1 = env.fork(base)
    fork2 = env.fork(base)
    r1 = env.step(fork1, A)
    r2 = env.step(fork2, B)
    # stepping one fork does not affect the other or the base state (immutable env state)
    assert r1.next_state.previous_topology == A
    assert r2.next_state.previous_topology == B
    assert base.previous_topology == A and base.frame_index == 1


def test_T1_matches_static_mode() -> None:
    env = TwoTimescaleTopologyEnv([0], _const_cost(3.0), ReconfigCost(e_edge=9.0), hold_interval=1)
    result = env.step(env.reset(), A)
    # a single frame with no prior topology pays exactly the static per-frame objective (no reconfig)
    assert result.reconfig_cost == pytest.approx(0.0)
    assert -result.reward == pytest.approx(3.0)


def test_temporal_value_test_zero_reconfig_is_zero_delta() -> None:
    env = TwoTimescaleTopologyEnv([0, 1], _alternating_cost, ReconfigCost())  # reconfig = 0
    out = temporal_value_test(env, [[A, B], [A, B]])
    assert out["delta_h"] == pytest.approx(0.0)  # no switch cost -> myopic == horizon


def test_temporal_value_test_positive_when_reconfig_dominates() -> None:
    env = TwoTimescaleTopologyEnv([0, 1], _alternating_cost, ReconfigCost(e_edge=1.0))  # r=1/edge
    out = temporal_value_test(env, [[A, B], [A, B]])
    # myopic switches A->B (cost 1+1+2) = 4; horizon keeps a stable topology (cost 3) -> Δ_H = 1.
    assert out["j_myopic"] == pytest.approx(4.0)
    assert out["j_horizon"] == pytest.approx(3.0)
    assert out["delta_h"] == pytest.approx(1.0)


def _const_cost(value):
    return lambda _frame, _topology: value
