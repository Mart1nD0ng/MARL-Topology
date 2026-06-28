"""Q7 (POMDP-QP-FAR): D_quorum-guided add-only repair.

Pins: the greedy repair is add-only (repaired superset of anchor, budgets respected), it never increases
D_quorum, retention is total, and the repair-head target is exactly D(x) - D(x+e). Fails on HEAD (the
module is new).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

from marl_topology.training.dynamic_baselines import local_hysteresis_action
from marl_topology.training.dynamic_frames import sample_dynamic_scenes
from marl_topology.training.dynamic_rl import _budgets_edges
from marl_topology.training.quorum_deficit_bridge import topology_quorum_deficit
from marl_topology.training.residual_repair import (
    greedy_dquorum_add_repair, repair_head_targets)
from marl_topology.training.two_timescale_env import ReconfigCost


def _setup(nodes=(12,)):
    from build_operating_point_dataset import operating_point_regime
    scene = sample_dynamic_scenes(
        seed=9, count=1, node_count_choices=nodes, regime=operating_point_regime(20.0),
        num_frames=1, dt_s=2.0, speed_min_mps=15.0, speed_max_mps=30.0,
        reconfig=ReconfigCost(), hold_interval=1, gamma=0.95)[0]
    obs = scene.observation(0, [])
    ev = obs["context"].evaluator
    budgets, edges = _budgets_edges(obs["context"])
    anchor = local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, [],
                                     keep_threshold=0.4, add_threshold=0.6)
    return obs, ev, anchor, budgets, edges


def test_greedy_repair_is_add_only() -> None:
    obs, ev, anchor, budgets, edges = _setup()
    r = greedy_dquorum_add_repair(ev, anchor, obs["edge_ids"], obs["context"], max_adds=3)
    assert set(anchor).issubset(set(r["repaired"]))          # never removes an anchor edge
    deg: dict = {}
    for eid in r["repaired"]:
        u, v = edges[eid]
        deg[u] = deg.get(u, 0) + 1; deg[v] = deg.get(v, 0) + 1
    for node, d in deg.items():
        assert d <= int(budgets.get(node, 0))                # budgets respected


def test_greedy_repair_reduces_dquorum() -> None:
    obs, ev, anchor, _b, _e = _setup()
    r = greedy_dquorum_add_repair(ev, anchor, obs["edge_ids"], obs["context"], max_adds=4)
    assert r["d_quorum_final"] <= r["d_quorum_init"] + 1e-9   # greedy only accepts deficit-reducing adds
    assert r["d_quorum_reduction"] >= -1e-9


def test_repair_retention_is_total() -> None:
    obs, ev, anchor, _b, _e = _setup()
    r = greedy_dquorum_add_repair(ev, anchor, obs["edge_ids"], obs["context"], max_adds=3)
    assert r["retention"] == 1.0 and r["anchor_retained"] is True


def test_repair_head_target_is_delta_dquorum() -> None:
    obs, ev, anchor, _b, _e = _setup()
    targets = repair_head_targets(ev, anchor, obs["edge_ids"], obs["context"])
    aset = set(anchor)
    base = topology_quorum_deficit(ev, aset)["d_quorum_mean"]
    cand = next(e for e in obs["edge_ids"] if e not in aset)
    d_e = topology_quorum_deficit(ev, aset | {cand})["d_quorum_mean"]
    assert targets[cand] == pytest.approx(base - d_e, abs=1e-6)   # r_e = D(x) - D(x+e)
    assert all(e not in aset for e in targets)                    # non-anchor edges only
