"""Phase 10 (Spec §5.4/§6.2/§6.3/§6.4): distribution-level reliability constraint primitives.

Pins the chance residual (signed fraction-below-tau minus delta) + its sign-flexible projected dual,
the CVaR shortfall against an independent worst-tail-mean oracle (and monotonicity in alpha), and the
Pareto checkpoint selection order (reliability-risk -> min violation -> non-dominated -> hypervolume),
which must NOT reduce to raw feasibility.
"""

from __future__ import annotations

import pytest

from marl_topology.training.reliability_constraints import (
    chance_dual_update,
    chance_residual,
    cvar_shortfall,
    pareto_archive_select,
    tail_mean_shortfall,
)

TAU = 0.9


def test_chance_residual_is_frac_below_minus_delta() -> None:
    c = [0.95, 0.92, 0.80, 0.70, 0.99]      # 2 of 5 below tau=0.9 -> frac 0.4
    assert chance_residual(c, TAU, delta=0.2) == pytest.approx(0.4 - 0.2)
    assert chance_residual(c, TAU, delta=0.4) == pytest.approx(0.0)   # exactly met
    assert chance_residual([0.95, 0.99], TAU, delta=0.1) == pytest.approx(-0.1)  # none below -> negative
    assert chance_residual([], TAU, delta=0.1) == 0.0


def test_chance_dual_rises_when_violated_falls_when_satisfied_nonneg() -> None:
    # violated (residual > 0) -> lambda rises
    assert chance_dual_update(0.5, residual=0.3, lr=1.0) == pytest.approx(0.8)
    # satisfied (residual < 0) -> lambda falls
    assert chance_dual_update(0.5, residual=-0.3, lr=1.0) == pytest.approx(0.2)
    # cannot go negative (projection onto >= 0)
    assert chance_dual_update(0.1, residual=-1.0, lr=1.0) == 0.0
    # respects lam_max
    assert chance_dual_update(0.9, residual=1.0, lr=1.0, lam_max=1.0) == 1.0


def test_cvar_matches_independent_tail_mean_when_exact() -> None:
    # n=10, alpha=0.8 -> worst (1-alpha)*n = 2 shortfalls; CVaR == mean of the 2 largest deficits
    c = [0.99, 0.97, 0.95, 0.93, 0.91, 0.88, 0.85, 0.80, 0.70, 0.60]
    assert cvar_shortfall(c, TAU, alpha=0.8) == pytest.approx(tail_mean_shortfall(c, TAU, alpha=0.8))
    # n=20, alpha=0.9 -> worst 2
    c2 = [0.9 + 0.005 * i for i in range(15)] + [0.5, 0.55, 0.6, 0.65, 0.7]
    assert cvar_shortfall(c2, TAU, alpha=0.9) == pytest.approx(tail_mean_shortfall(c2, TAU, alpha=0.9))


def test_cvar_is_tail_not_mean_and_monotone_in_alpha() -> None:
    c = [0.99, 0.95, 0.92, 0.88, 0.80, 0.70, 0.50]   # mixed feasible/infeasible
    mean_shortfall = sum(TAU - x for x in c) / len(c)
    cvar_high = cvar_shortfall(c, TAU, alpha=0.9)     # extreme tail (worst deficit)
    cvar_low = cvar_shortfall(c, TAU, alpha=0.0)      # alpha=0 -> CVaR == mean shortfall
    assert cvar_low == pytest.approx(mean_shortfall)
    assert cvar_high >= cvar_low                      # the tail is at least the mean
    assert cvar_shortfall(c, TAU, 0.9) >= cvar_shortfall(c, TAU, 0.5)   # monotone non-decreasing in alpha


def test_cvar_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        cvar_shortfall([0.9], TAU, alpha=1.0)


def test_pareto_select_prefers_reliability_then_nondominated() -> None:
    archive = [
        {"id": "A", "reliability_violation": 0.0, "energy": 1.0, "latency": 1.0, "hypervolume": 2.0},
        {"id": "B", "reliability_violation": 0.0, "energy": 2.0, "latency": 2.0, "hypervolume": 1.0},  # dominated
        {"id": "C", "reliability_violation": 0.3, "energy": 0.1, "latency": 0.1, "hypervolume": 9.0},  # great E/L but RISKY
    ]
    sel = pareto_archive_select(archive, risk_budget=0.05)
    assert sel["id"] == "A"   # C excluded (reliability risk), B dominated -> A wins


def test_pareto_not_raw_feasibility() -> None:
    # the entry with the BEST raw feasibility (highest, encoded as lowest violation? no -- here "raw
    # feasibility" = great E/L) but a reliability-risk violation must NOT be chosen over a safe one.
    archive = [
        {"id": "risky_but_efficient", "reliability_violation": 0.5, "energy": 0.01, "latency": 0.01},
        {"id": "safe", "reliability_violation": 0.0, "energy": 5.0, "latency": 5.0},
    ]
    assert pareto_archive_select(archive, risk_budget=0.05)["id"] == "safe"
    # with NO safe entry, falls back to min-violation (not best raw E/L)
    archive2 = [
        {"id": "less_risky", "reliability_violation": 0.2, "energy": 9.0, "latency": 9.0},
        {"id": "more_risky_efficient", "reliability_violation": 0.6, "energy": 0.01, "latency": 0.01},
    ]
    assert pareto_archive_select(archive2, risk_budget=0.05)["id"] == "less_risky"


def test_pareto_empty_archive() -> None:
    assert pareto_archive_select([]) is None


def test_pareto_archive_trunk_entry_shape_carries_state() -> None:
    # mirrors the trunk's per-eval archive entry (Phase 10c keep-best): the selector must work on the
    # full dict (incl. an opaque 'state' payload) and return the right checkpoint to restore.
    archive = [
        {"update": 20, "reliability_violation": 0.4, "energy": 0.1, "latency": 0.1,
         "hypervolume": 0.0, "stability": 0.6, "state": "ckpt@20"},     # risky
        {"update": 40, "reliability_violation": 0.0, "energy": 0.3, "latency": 0.3,
         "hypervolume": 0.0, "stability": 0.7, "state": "ckpt@40"},     # safe, non-dominated
        {"update": 60, "reliability_violation": 0.0, "energy": 0.5, "latency": 0.5,
         "hypervolume": 0.0, "stability": 0.9, "state": "ckpt@60"},     # safe but dominated by @40
    ]
    sel = pareto_archive_select(archive, risk_budget=0.05)
    assert sel["state"] == "ckpt@40"     # reliability-satisfied + non-dominated wins over risky @20 / dominated @60
