"""D11: episode-level chance / CVaR / Pareto / preference for the dynamic task (Contract §9, Plan §13)."""

from __future__ import annotations

import pytest


def test_dynamic_chance_dual_up_down() -> None:
    # the chance dual is SIGN-FLEXIBLE: it RISES when the frame failure rate exceeds delta and FALLS
    # (toward 0) when it is met -- a real chance constraint, not a one-sided ascent.
    from marl_topology.training.dynamic_reliability import episode_chance_dual_step
    lam_up, res_up = episode_chance_dual_step(0.0, [False, False, True], delta=0.1, lr=0.5)
    assert res_up > 0.0 and lam_up > 0.0                  # 2/3 below tau > 0.1 -> violated -> lam rises
    lam_dn, res_dn = episode_chance_dual_step(1.0, [True, True, True], delta=0.1, lr=0.5)
    assert res_dn < 0.0 and lam_dn < 1.0                  # 0 below tau < 0.1 -> met -> lam falls
    lam_floor, _ = episode_chance_dual_step(0.0, [True, True, True], delta=0.1, lr=0.5)
    assert lam_floor == 0.0                               # never negative (projected to >= 0)


def test_dynamic_cvar_metric_matches_bruteforce() -> None:
    # CVaR of the per-frame shortfalls matches the brute-force tail-mean oracle.
    from marl_topology.training.dynamic_reliability import cvar_bruteforce, episode_cvar_shortfall
    margins = [0.0, 0.1, 0.3, 0.0, 0.5]                   # D_t = (tau - C_t)_+
    cv = episode_cvar_shortfall(margins, alpha=0.6)       # worst ceil(0.4*5)=2 -> (0.5+0.3)/2 = 0.4
    assert abs(cv - 0.4) < 1e-9
    assert abs(cv - cvar_bruteforce(margins, alpha=0.6)) < 1e-9
    for a in (0.0, 0.6, 0.8):                            # (1-a)*5 integral -> CVaR == tail-mean exactly
        assert abs(episode_cvar_shortfall(margins, alpha=a) - cvar_bruteforce(margins, alpha=a)) < 1e-9
    assert episode_cvar_shortfall([], alpha=0.5) == 0.0


def test_dynamic_pareto_archive_uses_val() -> None:
    # the dynamic Pareto archive selects the checkpoint from VALIDATION metrics by reliability-risk ->
    # min violation -> non-dominated, NEVER by raw feasibility, and refuses a non-val-seeded archive.
    from marl_topology.training.dynamic_reliability import dynamic_pareto_select
    archive = [
        {"split": "val", "reliability_violation": 0.20, "energy": 1.0, "latency": 1.0, "update": 5},
        {"split": "val", "reliability_violation": 0.05, "energy": 2.0, "latency": 2.0, "update": 10},
        {"split": "val", "reliability_violation": 0.05, "energy": 1.0, "latency": 1.0, "update": 15},
    ]
    chosen = dynamic_pareto_select(archive)
    assert chosen["split"] == "val"
    assert chosen["reliability_violation"] == 0.05        # min violation, not raw feasibility
    assert chosen["update"] == 15                         # the non-dominated (lower energy+latency) one
    with pytest.raises(ValueError):                       # held/train must NOT seed the archive (§3.4)
        dynamic_pareto_select([{"split": "held", "reliability_violation": 0.0, "energy": 1.0, "latency": 1.0}])


def test_dynamic_preference_changes_actions() -> None:
    # a preference over the energy-latency trade-off changes which topology is reward-best (the action).
    from marl_topology.training.dynamic_reliability import preference_weighted_objective
    # topology A: lower feasibility margin but cheap; topology B: higher margin but expensive.
    a_e, b_e = 0.5, 2.0
    rA0 = preference_weighted_objective(0.40, a_e, 1.0, omega_e=0.0, omega_l=0.0)
    rB0 = preference_weighted_objective(0.50, b_e, 1.0, omega_e=0.0, omega_l=0.0)
    assert rB0 > rA0                                      # no preference -> B (higher base) wins
    rA1 = preference_weighted_objective(0.40, a_e, 1.0, omega_e=1.0, omega_l=0.0)
    rB1 = preference_weighted_objective(0.50, b_e, 1.0, omega_e=1.0, omega_l=0.0)
    assert rA1 > rB1                                      # energy preference -> A (cheaper) wins (action changed)
    # omega = 0 is byte-identical to the plain objective
    assert preference_weighted_objective(0.5, 9.0, 9.0, omega_e=0.0, omega_l=0.0) == 0.5
