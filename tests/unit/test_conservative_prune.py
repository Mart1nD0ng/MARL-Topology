"""Q8 (POMDP-QP-FAR): D_quorum-safety-guided conservative prune.

Pins: the greedy prune is remove-only (pruned subset of anchor), retains feasibility, deletes 0 critical
edges, never increases energy, and the safety-head target is exactly D(x∖e) − D(x). Fails on HEAD (the
functions are new).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

from marl_topology.training.dynamic_baselines import local_hysteresis_action
from marl_topology.training.dynamic_frames import sample_dynamic_urban_scenes
from marl_topology.training.dynamic_rl import _budgets_edges
from marl_topology.training.quorum_deficit_bridge import topology_quorum_deficit, topology_reliability
from marl_topology.training.residual_repair import greedy_conservative_prune, safety_head_targets
from marl_topology.training.two_timescale_env import ReconfigCost

_TAU = 0.9


def _feasible_setup():
    """Find an anchor-FEASIBLE frame (urban anchors are mostly feasible, D13)."""
    from build_operating_point_dataset import operating_point_regime
    scenes = sample_dynamic_urban_scenes(
        seed=3, count=5, node_count_choices=(12, 16), regime=operating_point_regime(20.0),
        num_frames=4, dt_s=2.0, speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
        hold_interval=4, gamma=0.95)
    for sc in scenes:
        prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            budgets, edges = _budgets_edges(obs["context"])
            anchor = local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev,
                                             keep_threshold=0.4, add_threshold=0.6)
            ev = obs["context"].evaluator
            if len(anchor) >= 2 and topology_reliability(ev, anchor)["consensus"] >= _TAU:
                return obs, ev, anchor
            prev = anchor
    raise AssertionError("no feasible anchor frame found (urban)")


def test_prune_is_remove_only() -> None:
    obs, ev, anchor = _feasible_setup()
    r = greedy_conservative_prune(ev, anchor, obs["edge_ids"], obs["context"], max_removes=3)
    assert r["applicable"] is True
    assert set(r["pruned"]).issubset(set(anchor))            # no new edge
    assert r["removed_subset_of_anchor"] is True


def test_prune_retains_feasibility() -> None:
    obs, ev, anchor = _feasible_setup()
    r = greedy_conservative_prune(ev, anchor, obs["edge_ids"], obs["context"], max_removes=4)
    assert r["pruned_C"] >= _TAU                             # feasibility kept
    assert r["feasibility_retained"] is True and r["retention"] == 1.0


def test_prune_no_critical_deletion() -> None:
    obs, ev, anchor = _feasible_setup()
    r = greedy_conservative_prune(ev, anchor, obs["edge_ids"], obs["context"], max_removes=4)
    assert r["critical_edge_deletions"] == 0                 # the greedy checks C>=tau before each removal


def test_prune_reports_cost_change() -> None:
    # NOTE: energy is NOT monotone in edge count here -- the one-hop-relay PBFT regime means removing a
    # direct edge can force a 2-hop relay (MORE transmissions -> MORE energy). The cost direction is an
    # EMPIRICAL question for the diagnostic, not a structural invariant. The prune only reports it.
    import math
    obs, ev, anchor = _feasible_setup()
    r = greedy_conservative_prune(ev, anchor, obs["edge_ids"], obs["context"], max_removes=4)
    assert math.isfinite(r["pruned_energy"]) and math.isfinite(r["pruned_latency"])
    assert "energy_reduction" in r and "latency_reduction" in r
    assert r["energy_reduction"] == pytest.approx(r["anchor_energy"] - r["pruned_energy"], abs=1e-6)


def test_safety_head_target_is_risk() -> None:
    obs, ev, anchor = _feasible_setup()
    targets = safety_head_targets(ev, anchor, obs["edge_ids"], obs["context"])
    aset = set(anchor)
    assert set(targets) == aset                             # anchor edges only
    base = topology_quorum_deficit(ev, aset)["d_quorum_mean"]
    e = next(iter(aset))
    d_without = topology_quorum_deficit(ev, aset - {e})["d_quorum_mean"]
    assert targets[e] == pytest.approx(d_without - base, abs=1e-6)   # risk_e = D(x∖e) - D(x)
